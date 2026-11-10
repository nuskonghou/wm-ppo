"""Temperature-1 sampling runs of the evaluation protocol on the 284 ScienceWorld evaluation variations, one episode per game
(rule 'random': at every step the policy's first sample is executed). Output <out>/infer_<arm>_random<tag>.jsonl, one
line per game; the evaluation protocol (scripts/launch.py eval) runs it three times per checkpoint (pass@1 = mean, pass@3 = any of the three)."""
import argparse
import collections
import json
import re
import os
import random
import sys
import time

import requests

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
            hist.append((cur, choice))
            cur, adm, r, done = env.step(choice)
        try: env.close()
        except Exception: pass  # noqa: E722
        return {'game': job['gamefile'], 'rule': job['rule'], 'reward': r, 'steps': len(hist), **stat}
    except Exception as e:  # noqa: BLE001
        return {'game': job['gamefile'], 'rule': job['rule'], 'error': repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--rule', required=True, choices=['random'])
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--lora', required=True)
    ap.add_argument('--sglang', required=True)
    ap.add_argument('--tag', default='', help='suffix of the output file name')
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--max-steps', type=int, default=40)
    ap.add_argument('--model', default=paths.MODEL)
    a = ap.parse_args()
    games = game_files('valid_seen') + game_files('valid_unseen')
    jobs = [{'gamefile': g, 'rule': a.rule, 'url': a.sglang, 'lora': a.lora, 'temperature': a.temperature, 'max_new_tokens': 256,
             'max_steps': a.max_steps, 'model': a.model, 'K': a.K} for g in games]
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
