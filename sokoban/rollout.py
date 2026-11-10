"""Episode workers: (optional) deterministic prefix replay, then the policy-environment loop (text Sokoban).

A job is a dict:
  gamefile ('<split>/<seed>/level'), prefix (list of actions to replay, may be empty), avoid_first (optional: resample
  the first generated turn up to k_alt times to get an action different from this one), url (sglang /generate base),
  lora (adapter name or None), temperature, max_new_tokens, max_steps, model (HF path for the tokenizer), tag.
The result records every turn the policy generated (turns of the replayed prefix are not included):
  {'t', 'prompt_ids', 'resp_ids', 'lps', 'action', 'valid', 'energy', 'obs'} with energy = sum (1 - p_token)^2.
WM-PPO additions: with K > 1 every turn draws K samples in one batched request and executes the first (so the behaviour
policy is unchanged) and records them in turn['samples']; with wm_url set, a turn with >= 2 distinct sampled actions gets
turn['wm_child'] = {action: policy-format prompt of the predicted successor} and turn['wm_term'] = {action: 1 / None},
1 when the predicted grid is solved, and the first turn of a branch gets turn['calib'] = the chain's action and the
branch's action with both predicted successors (and term_c / term_b). With job['record_wm'] the result also carries
'wm_trans', every executed transition in the world model's record format (online world model). With job['mc'] every
turn carries 'mc', ProAct's Monte-Carlo value of the current state (env.mc_value).
"""
import os, re, time, json, multiprocessing as mp
import requests
from env import AlfEnv
from prompts import messages, parse_action, split_task

_TOK = None
_SES = None


def _tok(path):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(path)
    return _TOK


def _post(url, payload, timeout=600, retries=4):
    global _SES
    if _SES is None:
        _SES = requests.Session()
    last = None
    for a in range(retries):
        try:
            r = _SES.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"POST {url}: {last}")


def generate(job, ids):
    payload = {"input_ids": ids,
               "sampling_params": {"temperature": job['temperature'], "top_p": 1.0, "top_k": -1,
                                   "max_new_tokens": job['max_new_tokens'], "skip_special_tokens": False,
                                   "no_stop_trim": True, "spaces_between_special_tokens": False},
               "return_logprob": True}
    if job.get('lora'):
        payload['lora_path'] = job['lora']
    o = _post(job['url'].rstrip('/') + '/generate', payload)
    meta = o.get('meta_info', {})
    items = [it for it in meta.get('output_token_logprobs', []) if it[0] is not None]
    return o.get('text', ''), [int(it[1]) for it in items], [float(it[0]) for it in items]


def wm_terminal(o_hat):
    """Sokoban: 1 if the predicted grid is solved (no box off a target, at least one box on a target: the simulator
    ends the episode with reward 1), else None. The trainer uses this as the value of the predicted successor instead
    of the critic's value, as for ScienceWorld's predicted endings."""
    return 1 if ('X' not in o_hat and '\u221a' in o_hat) else None


def generate_k(job, ids, k):
    """k i.i.d. samples for the same prompt in ONE batched sglang request (~900-token responses, so the K = 4 samples of
    WM-PPO are decoded concurrently instead of one after another; same distribution)."""
    payload = {"input_ids": [ids] * k,
               "sampling_params": {"temperature": job['temperature'], "top_p": 1.0, "top_k": -1,
                                   "max_new_tokens": job['max_new_tokens'], "skip_special_tokens": False,
                                   "no_stop_trim": True, "spaces_between_special_tokens": False},
               "return_logprob": True}
    if job.get('lora'):
        payload['lora_path'] = [job['lora']] * k
    outs = _post(job['url'].rstrip('/') + '/generate', payload)
    res = []
    for o in outs:
        items = [it for it in o.get('meta_info', {}).get('output_token_logprobs', []) if it[0] is not None]
        res.append((o.get('text', ''), [int(it[1]) for it in items], [float(it[0]) for it in items]))
    return res


def wm_child_ids(job, tok, task, room, hist, cur, adm, action, term=None):
    """Policy-format prompt of the state the WM predicts after `action` (greedy WM; admissible list as before the
    action), i.e. the input the critic scores in ct_infer.py --rule critic_wm. With a dict `term`, also records
    term[action] = wm_terminal(prediction)."""
    from wm_prompt import SYSTEM as WM_SYSTEM, wm_user
    r = {'task': task, 'room': room, 'hist': [list(x) for x in hist], 'obs': cur, 'adm': adm, 'action': action}
    ids = tok.apply_chat_template([{"role": "system", "content": WM_SYSTEM}, {"role": "user", "content": wm_user(r)}],
                                  tokenize=True, add_generation_prompt=True, return_dict=False)
    ids = ids if isinstance(ids, list) else ids['input_ids']
    text = generate({'url': job['wm_url'], 'lora': job.get('wm_lora'), 'temperature': 0.0, 'max_new_tokens': 1024}, ids)[0]
    m = re.search(r"<obs>(.*?)</obs>", text, re.S)
    o_hat = (m.group(1) if m else text.replace('<|im_end|>', '')).strip()
    if term is not None:
        term[action] = wm_terminal(o_hat)
    c = tok.apply_chat_template(messages(task, room, hist + [(cur, action)], o_hat, adm), tokenize=True,
                                add_generation_prompt=True, return_dict=False)
    return c if isinstance(c, list) else c['input_ids']


def run_episode(job):
    t_start = time.time()
    tok = _tok(job['model'])
    env = AlfEnv(job['gamefile'], max_steps=job['max_steps'])
    obs, adm = env.reset()
    task, room = split_task(obs)
    hist, cur, r, done = [], room, 0.0, False
    for a in job.get('prefix', []):
        hist.append((cur, a))
        cur, adm, r, done = env.step(a)
        if done:
            break
    turns = []
    n_tok = 0
    wm_trans = []
    while not done:
        msgs = messages(task, room, hist, cur, adm)
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
        ids = ids if isinstance(ids, list) else ids['input_ids']
        branch_first = not turns and bool(job.get('avoid_first'))
        samples = None
        if job.get('K', 1) > 1 and not branch_first:
            outs = generate_k(job, ids, job['K'])
            samples = [{'resp_ids': o[1], 'lps': o[2], 'action': parse_action(o[0], adm)[0]} for o in outs]
            text, rid, lps = outs[0]                         # execute the first sample: behaviour policy unchanged
        else:
            text, rid, lps = generate(job, ids)
        action, valid = parse_action(text, adm)
        # branch: at the first generated turn, prefer an action different from the one the source chain took
        if branch_first:
            tries = 1
            while action == job['avoid_first'] and tries < job.get('k_alt', 4):
                text, rid, lps = generate(job, ids)
                action, valid = parse_action(text, adm)
                tries += 1
        energy = float(sum((1.0 - pow(2.718281828, lp)) ** 2 for lp in lps))
        if job.get('mc'):                        # mcppo: ProAct Monte-Carlo critic of the current (cloned) state
            mc_v = env.mc_value(job['mc']['M'], job['mc']['T'], job['mc']['gamma'])
        turn = {'t': len(hist), 'prompt_ids': ids, 'resp_ids': rid, 'lps': lps, 'action': action,
                'valid': valid, 'energy': energy, 'obs': cur, **({'mc': mc_v} if job.get('mc') else {})}
        if samples is not None:
            turn['samples'] = samples
            distinct = list(dict.fromkeys(x['action'] for x in samples))
            if len(distinct) > 1 and job.get('wm_url'):
                term = {}
                turn['wm_child'] = {a: wm_child_ids(job, tok, task, room, hist, cur, adm, a, term) for a in distinct}
                turn['wm_term'] = term
        if branch_first and job.get('wm_url') and action != job['avoid_first']:
            term = {}
            turn['calib'] = {'a_c': job['avoid_first'], 'a_b': action,
                             'child_c': wm_child_ids(job, tok, task, room, hist, cur, adm, job['avoid_first'], term),
                             'child_b': wm_child_ids(job, tok, task, room, hist, cur, adm, action, term)}
            turn['calib'].update(term_c=term[job['avoid_first']], term_b=term[action])
        turns.append(turn)
        n_tok += len(rid)
        prev = (cur, adm, [list(x) for x in hist])
        hist.append((cur, action))
        cur, adm, r, done = env.step(action)
        if job.get('record_wm'):                 # online world model: the executed transition, in the record format of train_wm.py
            wm_trans.append({'task': task, 'room': room, 'hist': prev[2], 'obs': prev[0], 'adm': prev[1], 'action': action,
                             'next_obs': cur, 'next_adm': adm, 'done': done, 'won': bool(done and r > 0), 'executed': True,
                             'count': 1, 'game': job['gamefile'], 't': len(prev[2])})
    for x in wm_trans:
        x['episode_reward'] = r
    try:
        env.close()                    # no resources to release (same interface as the ScienceWorld env)
    except Exception:  # noqa: BLE001
        pass
    return {'gamefile': job['gamefile'], 'tag': job.get('tag'), 'prefix': list(job.get('prefix', [])),
            'actions': [a for _, a in hist], 'turns': turns, 'reward': r, 'steps': len(hist), 'final_obs': cur,
            'gen_tokens': n_tok, 'replay_steps': len(job.get('prefix', [])), 'wall_s': time.time() - t_start,
            **({'wm_trans': wm_trans} if job.get('record_wm') else {})}


def _safe(job):
    try:
        return run_episode(job)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {'gamefile': job['gamefile'], 'tag': job.get('tag'), 'error': repr(e),
                'trace': traceback.format_exc(limit=4)}


class Pool:
    def __init__(self, n):
        ctx = mp.get_context('fork')
        # workers are recycled every few episodes so that per-process resources are released regularly
        self.pool = ctx.Pool(n, maxtasksperchild=4)

    def run(self, jobs):
        return self.pool.map(_safe, jobs, chunksize=1)

    def close(self):
        self.pool.close(); self.pool.join()
