"""Episode workers: (optional) deterministic prefix replay, then the policy-environment loop.

A job is a dict:
  gamefile, prefix (list of actions to replay, may be empty), avoid_first (optional: resample the first generated
  turn up to k_alt times to get an action different from this one), url (sglang /generate base), lora (adapter name
  or None), temperature, max_new_tokens, max_steps, model (HF path for the tokenizer), tag (free metadata).
The result records every turn the policy generated (turns of the replayed prefix are not included):
  {'t', 'prompt_ids', 'resp_ids', 'lps', 'action', 'valid', 'energy', 'obs'} with energy = sum (1 - p_token)^2.
WM-PPO additions: with K > 1 every turn draws K samples and executes the first (so the behaviour policy is unchanged)
and records them in turn['samples']; with wm_url set, a turn with >= 2 distinct sampled actions gets
turn['wm_child'] = {action: policy-format prompt of the predicted successor}, and the first turn of a branch gets
turn['calib'] = the chain's action and the branch's action with both predicted successors. With job['record_wm'] the
result also carries 'wm_trans', every executed transition in the world model's record format (online world model).
Baselines: job['opd'] (teacher log-probs of the executed response), job['ocsd'] (OCSD teacher scores) and job['paw']
(the next observation of each executed action).
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


def wm_child_ids(job, tok, task, room, hist, cur, adm, action):
    """Policy-format prompt of the state the WM predicts after `action` (greedy WM; admissible list as before the
    action), i.e. the input the critic scores in ct_infer.py --rule critic_wm."""
    from wm_prompt import SYSTEM as WM_SYSTEM, wm_user
    r = {'task': task, 'room': room, 'hist': [list(x) for x in hist], 'obs': cur, 'adm': adm, 'action': action}
    ids = tok.apply_chat_template([{"role": "system", "content": WM_SYSTEM}, {"role": "user", "content": wm_user(r)}],
                                  tokenize=True, add_generation_prompt=True, return_dict=False)
    ids = ids if isinstance(ids, list) else ids['input_ids']
    text = generate({'url': job['wm_url'], 'lora': job.get('wm_lora'), 'temperature': 0.0, 'max_new_tokens': 300}, ids)[0]
    m = re.search(r"<obs>(.*?)</obs>", text, re.S)
    o_hat = (m.group(1) if m else text.replace('<|im_end|>', '')).strip()
    c = tok.apply_chat_template(messages(task, room, hist + [(cur, action)], o_hat, adm), tokenize=True,
                                add_generation_prompt=True, return_dict=False)
    return c if isinstance(c, list) else c['input_ids']


def opd_teacher_lps(job, tok, task, room, hist, cur, adm, rid):
    """Standard OPD (arm ppo160opd): log-probs of the executed response tokens under the teacher given exactly the
    student's prompt, built with the student's chat template. Teacher = job['opd_url'] / job['opd_lora'] (a stronger
    model with the same tokenizer, Qwen3-32B with base weights) or, without opd_url, the policy itself. One prefill."""
    msgs = messages(task, room, hist, cur, adm)
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
    ids = (ids if isinstance(ids, list) else ids['input_ids']) + list(job.get('opd_prefix_ids') or [])
    payload = {"input_ids": ids + list(rid), "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
               "return_logprob": True, "logprob_start_len": len(ids) - 1}
    lora = job['opd_lora'] if 'opd_lora' in job else job.get('lora')
    if lora:
        payload['lora_path'] = lora
    il = _post((job.get('opd_url') or job['url']).rstrip('/') + '/generate', payload)['meta_info']['input_token_logprobs'][1:1 + len(rid)]
    assert len(il) == len(rid) and all(int(x[1]) == t for x, t in zip(il, rid)), 'teacher scoring misaligned'
    return [float(x[0]) for x in il]


# OCSD (Observation-Calibrated Self-Distillation), arm ppo160ocsd. The four helpers below are copied from the released
# code, github.com/aim-uofa/Self-Distillation-AgentRL (commit 35a44f5, Apache-2.0), verl/trainer/ppo/ocsd_utils.py
# (extract_action_text, build_action_schema, build_future_block, insert_future_evidence_before_response); the future
# evidence follows ocsd_rollout.py (next one / two observations the agent saw, the schema of the next action, none for
# the last step) and the step selection follows select_uncertain_steps (top 20% mean-token NLL per trajectory).
_OCSD_MARKERS = ("<|im_start|>assistant", "<|start_header_id|>assistant<|end_header_id|>", "<|assistant|>",
                 "[ASSISTANT]", "### Assistant:", "Assistant:")


def _ocsd_extract_action_text(response):
    for pattern in (r"<action>(.*?)</action>", r"(<search>.*?</search>)", r"(<answer>.*?</answer>)"):
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    match = re.search(r"(?i)^action:\s*(.+)", response, re.MULTILINE)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return response.strip() or "a future action"


def _ocsd_action_schema(action, env_name):
    action = action.strip(); env_name = env_name.lower(); lowered = action.lower()
    if "alfworld" in env_name:
        for pattern, schema in ((r"^take .+ from .+", "take an item from a receptacle"), (r"^move .+ to .+", "move an item to a location"),
                                (r"^heat .+ with .+", "heat an item with an appliance"), (r"^cool .+ with .+", "cool an item with an appliance"),
                                (r"^clean .+ with .+", "clean an item with a fixture"), (r"^slice .+ with .+", "slice an item with a tool"),
                                (r"^go to .+", "go to a location"), (r"^open .+", "open a receptacle"), (r"^close .+", "close a receptacle"),
                                (r"^use .+", "use a lamp"), (r"^examine .+", "examine a target")):
            if re.match(pattern, lowered):
                return schema
        return "a future action"
    if env_name == "scienceworld":
        for pattern, schema in ((r"^focus on .+", "focus on a task object"), (r"^use .+ on .+", "use a tool on a target"),
                                (r"^connect .+ to .+", "connect two electrical terminals"), (r"^move .+ to .+", "move an object to a container"),
                                (r"^pour .+ into .+", "pour a liquid into a container"), (r"^pick up .+", "pick up an object"),
                                (r"^go to .+", "go to a room"), (r"^(open|close|activate|deactivate|mix|read|look at|look in) .+", None)):
            match = re.match(pattern, lowered)
            if match:
                return schema or f"{match.group(1)} a target"
        return lowered if lowered in {"wait", "wait1", "look around", "inventory", "task"} else "a future action"
    return "a future action"


def _ocsd_future_block(available_future_len, obs_1, action_1, obs_2, env_name, full):
    shown_obs_1 = obs_1.strip() if full and obs_1 else "not provided"
    if available_future_len <= 1:
        return "Future evidence:\nAfter current action:\n" f"Observation: {shown_obs_1}"
    shown_obs_2 = obs_2.strip() if full and obs_2 else "not provided"
    schema = _ocsd_action_schema(action_1, env_name)
    return ("Future evidence:\nAfter current action:\n" f"Observation: {shown_obs_1}\n\n" "Next action:\n" f"{schema}\n\n"
            "After next action:\n" f"Observation: {shown_obs_2}")


def _ocsd_insert(prompt, future_block):
    insert_at = max(prompt.rfind(marker) for marker in _OCSD_MARKERS)
    block = "\n\n" + future_block.strip() + "\n\n"
    if insert_at < 0:
        return prompt.rstrip() + block
    return prompt[:insert_at].rstrip() + block + prompt[insert_at:].lstrip()


def _teacher_lps(url, lora, ids, rid):
    payload = {"input_ids": ids + list(rid), "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
               "return_logprob": True, "logprob_start_len": len(ids) - 1}
    if lora:
        payload['lora_path'] = lora
    il = _post(url.rstrip('/') + '/generate', payload)['meta_info']['input_token_logprobs'][1:1 + len(rid)]
    assert len(il) == len(rid) and all(int(x[1]) == t for x, t in zip(il, rid)), 'teacher scoring misaligned'
    return [float(x[0]) for x in il]


def ocsd_scores(job, tok, turns):
    """turn['ocsd'] = {'full', 'abl'}: teacher log-probs of the executed response with the Full / Ablated ('not provided')
    future evidence, on the selected steps. Teacher = job['ocsd_url'] (Qwen3-32B base, non-thinking prefix) or the policy
    itself = theta_old (the sampler's LoRA; OCSDRayTrainer scores with the actor right after old_log_prob)."""
    import math
    elig = [k for k, tu in enumerate(turns) if tu['resp_ids'] and k + 1 < len(turns)]   # available_future_len > 0
    if not elig:
        return
    nll = {k: -sum(turns[k]['lps']) / max(1, len(turns[k]['lps'])) for k in elig}
    n = max(1, math.ceil(len(elig) * job.get('ocsd_top', 0.2)))
    url, lora = (job['ocsd_url'], None) if job.get('ocsd_url') else (job['url'], job.get('lora'))
    for k in sorted(elig, key=lambda x: -nll[x])[:n]:
        avail = min(job.get('ocsd_k', 2), 2 if k + 2 < len(turns) else 1)
        o1 = turns[k + 1]['obs']; o2 = turns[k + 2]['obs'] if avail >= 2 else ''
        a1 = _ocsd_extract_action_text(tok.decode(turns[k + 1]['resp_ids'])) if avail >= 2 else ''
        p = tok.decode(turns[k]['prompt_ids'])
        out = {'avail': avail}
        try:                                   # one failed step (e.g. over the 8k context) leaves that step unmodulated
            for name, full in (('full', True), ('abl', False)):
                text = _ocsd_insert(p, _ocsd_future_block(avail, o1, a1, o2, job['ocsd_env'], full))
                ids = tok.encode(text, add_special_tokens=False) + list(job.get('ocsd_prefix_ids') or [])
                out[name] = _teacher_lps(url, lora, ids, turns[k]['resp_ids'])
            turns[k]['ocsd'] = out
        except Exception as e:  # noqa: BLE001
            turns[k]['ocsd_err'] = repr(e)[:200]


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
            outs = [generate(job, ids) for _ in range(job['K'])]
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
        turn = {'t': len(hist), 'prompt_ids': ids, 'resp_ids': rid, 'lps': lps, 'action': action,
                'valid': valid, 'energy': energy, 'obs': cur}      # obs: anchor state of GiGPO / HGPO
        if samples is not None:
            turn['samples'] = samples
            distinct = list(dict.fromkeys(x['action'] for x in samples))
            if len(distinct) > 1 and job.get('wm_url'):
                turn['wm_child'] = {a: wm_child_ids(job, tok, task, room, hist, cur, adm, a) for a in distinct}
        if branch_first and job.get('wm_url') and action != job['avoid_first']:
            turn['calib'] = {'a_c': job['avoid_first'], 'a_b': action,
                             'child_c': wm_child_ids(job, tok, task, room, hist, cur, adm, job['avoid_first']),
                             'child_b': wm_child_ids(job, tok, task, room, hist, cur, adm, action)}
        turns.append(turn)
        n_tok += len(rid)
        prev = (cur, adm, [list(x) for x in hist])
        hist.append((cur, action))
        cur, adm, r, done = env.step(action)
        if job.get('paw'):                       # ppo160paw: the observation this action produced (PaW target)
            turn['next_obs'] = cur
        if job.get('opd') and rid:               # ppo160opd: teacher log-probs of the executed response
            try:
                h_ = [tuple(x) for x in prev[2]]
                turn['opd_lps'] = opd_teacher_lps(job, tok, task, room, h_, prev[0], prev[1], rid)
            except Exception as e:  # noqa: BLE001
                turn['opd_err'] = repr(e)[:200]
        if job.get('record_wm'):                 # online world model: the executed transition, in the record format of train_wm.py
            wm_trans.append({'task': task, 'room': room, 'hist': prev[2], 'obs': prev[0], 'adm': prev[1], 'action': action,
                             'next_obs': cur, 'next_adm': adm, 'done': done, 'won': bool(done and r > 0), 'executed': True,
                             'count': 1, 'game': job['gamefile'], 't': len(prev[2])})
    if job.get('ocsd') and turns:                # ppo160ocsd: Full / Ablated future-evidence scores (after the episode)
        try:
            ocsd_scores(job, tok, turns)
        except Exception as e:  # noqa: BLE001
            turns[0]['ocsd_err'] = repr(e)[:200]
    for x in wm_trans:
        x['episode_reward'] = r
    try:
        env.env.close()                    # textworld keeps resources per game; release them
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
        # workers are recycled every few episodes: a long-lived worker accumulates storage from the PDDL planner's
        # temporary files that is only released when the process exits
        self.pool = ctx.Pool(n, maxtasksperchild=4)

    def run(self, jobs):
        return self.pool.map(_safe, jobs, chunksize=1)

    def close(self):
        self.pool.close(); self.pool.join()
