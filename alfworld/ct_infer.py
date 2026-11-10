"""Inference with a selection rule on the 274 ALFWorld evaluation games, one episode per game. At every step the
policy samples K candidates (temperature 1) and a rule picks the action to execute:
  random             the first sample, i.e. plain temperature-1 sampling (the pass@1 runs of the evaluation protocol)
  sc                 the most frequent action (ties: first seen; majority of K)
  peek               execute every distinct candidate in a sandbox copy (fresh env + prefix replay); if one completes
                     the task, take it, else the majority vote (control of Section 2.1)
  critic_only        execute every distinct candidate for ONE step in the sandbox and take the one whose resulting state
                     the critic values highest (true next state); the environment's win/done signal is never used
  critic_parent_adm  as critic_only, but the resulting state is shown with the admissible actions of BEFORE the action
  critic_masked      as critic_parent_adm, and newly revealed contents ('you see ...' not seen earlier) are removed
  critic_wm          the world model predicts the next observation of every distinct candidate (greedy, admissible
                     actions of before the step) and the critic values the predicted states; no sandbox call
The critic rules query a value server (ct_value_server.py, --value-url). Output <out>/infer_<arm>_<rule><tag>.jsonl,
one line per game."""
import argparse
import collections
import json
import re
import os
import random
import sys
import time

import requests

os.environ.setdefault('ALF_SHARED_DOWNWARD', '0')   # sandbox envs coexist with the main env: no shared planner lib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rollout  # noqa: E402
from env import AlfEnv, game_files  # noqa: E402
from prompts import messages, parse_action, split_task  # noqa: E402
from wmppo import paths  # noqa: E402


def prompt(tok, task, room, hist, cur, adm):
    ids = tok.apply_chat_template(messages(task, room, hist, cur, adm), tokenize=True, add_generation_prompt=True,
                                  return_dict=False)
    return ids if isinstance(ids, list) else ids['input_ids']


def sandbox(gf, max_steps, acts):
    """State after replaying acts in a fresh env: (task, room, hist, cur, adm, reward, done)."""
    e = AlfEnv(gf, max_steps=max_steps)
    obs, adm = e.reset(); task, room = split_task(obs)
    hist, cur, r, done = [], room, 0.0, False
    for a in acts:
        hist.append((cur, a)); cur, adm, r, done = e.step(a)
        if done:
            break
    try: e.env.close()
    except Exception: pass  # noqa: E722
    return task, room, hist, cur, adm, r, done


SEE = re.compile(r"((?:On|In) (?:the |it)[^.,]*?,? ?you see|In it, you see|Next to it, you see|you see)[^.]*\.", re.I)


def mask_new_contents(obs, prior_obs):
    """Remove 'you see ...' clauses that do not already appear in an earlier observation of the episode: what a
    world model that cannot know unobserved contents could at best reproduce."""
    seen = " ".join(prior_obs)
    def rep(m):
        return m.group(0) if m.group(0) in seen else ""
    return re.sub(r"\s+", " ", SEE.sub(rep, obs)).strip()


def wm_predict(job, tok, task, room, hist, cur, adm, action):
    """The world model's greedy prediction of the next observation."""
    from wm_prompt import SYSTEM, wm_user
    r = {'task': task, 'room': room, 'hist': [list(x) for x in hist], 'obs': cur, 'adm': adm, 'action': action}
    ids = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": wm_user(r)}],
                                  tokenize=True, add_generation_prompt=True, return_dict=False)
    ids = ids if isinstance(ids, list) else ids['input_ids']
    wj = {'url': job['wm_url'], 'lora': None if job['wm_lora'] == 'none' else job['wm_lora'],   # 'none': the base model
          'temperature': 0.0, 'max_new_tokens': 600}
    text = rollout.generate(wj, ids)[0]
    mo = re.search(r"<obs>(.*?)</obs>", text, re.S)
    return (mo.group(1) if mo else text).strip()


def value(job, ids):
    """Critic values of the given prompts; --value-url may list several servers (comma-separated): one is picked at random."""
    url = random.choice(job['value_url'].split(','))
    return requests.post(url + '/value', json={'ids': ids}, timeout=1800).json()['v']


def episode(job):
    try:
        tok = rollout._tok(job['model'])
        env = AlfEnv(job['gamefile'], max_steps=job['max_steps'])
        obs, adm = env.reset(); task, room = split_task(obs)
        hist, cur, r, done = [], room, 0.0, False
        stat = collections.Counter()
        while not done:
            ids = prompt(tok, task, room, hist, cur, adm)
            samples = [parse_action(rollout.generate(job, ids)[0], adm)[0] for _ in range(job['K'])]
            counts = collections.Counter(samples)
            distinct = list(dict.fromkeys(samples))
            majority = max(distinct, key=lambda a: counts[a])
            choice = majority
            if len(distinct) > 1:
                stat['multi'] += 1
                if job['rule'] == 'random':
                    choice = samples[0]
                elif job['rule'] == 'critic_wm':
                    # the world model replaces the sandbox: no environment call for the candidates at all
                    cids = [prompt(tok, task, room, hist + [(cur, a)], wm_predict(job, tok, task, room, hist, cur, adm, a), adm)
                            for a in distinct]
                    v = value(job, cids)
                    choice = distinct[max(range(len(distinct)), key=lambda i: v[i])]
                    stat['critic_choice'] += 1; stat['critic_changed'] += int(choice != majority)
                elif job['rule'] in ('critic_only', 'critic_parent_adm', 'critic_masked'):
                    acts = [a for _, a in hist]
                    kids = {a: sandbox(job['gamefile'], job['max_steps'], acts + [a]) for a in distinct}
                    if job['rule'] == 'critic_only':
                        cids = [prompt(tok, *kids[a][:5]) for a in distinct]
                    else:
                        prior = [room] + [o for o, _ in hist] + [cur]
                        cids = []
                        for a in distinct:
                            t_, r_, h_, c_, _adm, _, _ = kids[a]
                            if job['rule'] == 'critic_masked':
                                c_ = mask_new_contents(c_, prior)
                            cids.append(prompt(tok, t_, r_, h_, c_, adm))     # admissible list as BEFORE the action
                    v = value(job, cids)
                    choice = distinct[max(range(len(distinct)), key=lambda i: v[i])]
                    stat['critic_choice'] += 1; stat['critic_changed'] += int(choice != majority)
                elif job['rule'] == 'peek':
                    acts = [a for _, a in hist]
                    kids = {a: sandbox(job['gamefile'], job['max_steps'], acts + [a]) for a in distinct}
                    win = [a for a in distinct if kids[a][6] and kids[a][5] > 0]
                    if win:
                        choice = win[0]; stat['peek_win'] += 1
            hist.append((cur, choice))
            cur, adm, r, done = env.step(choice)
        try: env.env.close()
        except Exception: pass  # noqa: E722
        return {'game': job['gamefile'], 'rule': job['rule'], 'reward': r, 'steps': len(hist), **stat}
    except Exception as e:  # noqa: BLE001
        return {'game': job['gamefile'], 'rule': job['rule'], 'error': repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--rule', required=True, choices=['random', 'sc', 'peek', 'critic_only', 'critic_parent_adm', 'critic_masked', 'critic_wm'])
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--lora', required=True)
    ap.add_argument('--sglang', required=True)
    ap.add_argument('--value-url', default=None)
    ap.add_argument('--wm-url', default=None)
    ap.add_argument('--wm-lora', default=None)
    ap.add_argument('--tag', default='', help='suffix of the output file name')
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--max-steps', type=int, default=40)
    ap.add_argument('--model', default=paths.MODEL)
    a = ap.parse_args()
    games = game_files('valid_seen') + game_files('valid_unseen')
    jobs = [{'gamefile': g, 'rule': a.rule, 'url': a.sglang, 'lora': a.lora, 'temperature': a.temperature, 'max_new_tokens': 256,
             'max_steps': a.max_steps, 'model': a.model, 'K': a.K, 'value_url': a.value_url, 'wm_url': a.wm_url,
             'wm_lora': a.wm_lora} for g in games]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context('fork').Pool(a.workers, maxtasksperchild=1) as p:
        res = p.map(episode, jobs, chunksize=1)
    path = os.path.join(a.out, f'infer_{a.arm}_{a.rule}{a.tag}.jsonl')
    with open(path, 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    ok = [r for r in res if 'error' not in r]
    sr = sum(r['reward'] for r in ok) / max(len(ok), 1)
    print(f"{a.arm} {a.rule}{a.tag}: SR {sr:.3f} on {len(ok)} games ({len(res) - len(ok)} errors) {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
