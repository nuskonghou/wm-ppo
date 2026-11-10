"""Offline update of Section 2.3, step 1: data for one PPO-clip epoch whose labels come from the world model and the
critic.

The policy itself acts (the first of K = 4 temperature-1 samples is executed). At every step with >= 2 distinct sampled
actions, the K samples are labelled with
  la  the preferred candidate: argmax over distinct actions of the critic's value of the predicted next state
      (the critic_wm rule of ct_infer.py: greedy world model, admissible list as before the action)
  sc  the majority choice (majority-vote labels)
One output line per game: per step with >= 2 distinct actions the prompt ids, the K samples (response ids, old
log-probs, action) and both labels. Training games only (a shard the world model was not trained on)."""
import argparse
import collections
import json
import os
import random
import sys
import time

os.environ.setdefault('ALF_SHARED_DOWNWARD', '0')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rollout  # noqa: E402
from ct_infer import prompt, value, wm_predict  # noqa: E402
from env import AlfEnv, game_files  # noqa: E402
from prompts import parse_action, split_task  # noqa: E402
from wmppo import paths  # noqa: E402


def episode(job):
    try:
        tok = rollout._tok(job['model'])
        env = AlfEnv(job['gamefile'], max_steps=job['max_steps'])
        obs, adm = env.reset(); task, room = split_task(obs)
        hist, cur, r, done = [], room, 0.0, False
        steps = []
        while not done:
            ids = prompt(tok, task, room, hist, cur, adm)
            outs = [rollout.generate(job, ids) for _ in range(job['K'])]
            acts = [parse_action(o[0], adm)[0] for o in outs]
            counts = collections.Counter(acts)
            distinct = list(dict.fromkeys(acts))
            if len(distinct) > 1:
                cids = []
                for a in distinct:
                    o_hat = wm_predict(job, tok, task, room, hist, cur, adm, a)
                    cids.append(prompt(tok, task, room, hist + [(cur, a)], o_hat, adm))
                v = value(job, cids)
                steps.append({'t': len(hist), 'prompt_ids': ids, 'acts': acts,
                              'resp_ids': [o[1] for o in outs], 'old_lp': [o[2] for o in outs],
                              'la': distinct[max(range(len(distinct)), key=lambda i: v[i])],
                              'sc': max(distinct, key=lambda a: counts[a]), 'v': dict(zip(distinct, v))})
            hist.append((cur, acts[0]))
            cur, adm, r, done = env.step(acts[0])
        try: env.env.close()
        except Exception: pass  # noqa: E722
        return {'game': job['gamefile'], 'reward': r, 'steps': steps, 'n_steps': len(hist)}
    except Exception as e:  # noqa: BLE001
        return {'game': job['gamefile'], 'error': repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora', required=True)
    ap.add_argument('--sglang', required=True)
    ap.add_argument('--value-url', required=True)
    ap.add_argument('--wm-url', required=True)
    ap.add_argument('--wm-lora', default='wm_obs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-games', type=int, default=500)
    ap.add_argument('--shard', type=int, default=2, help='train shard (WM was trained on shards 0 and 1)')
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--max-steps', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default=paths.MODEL)
    a = ap.parse_args()
    games = game_files('train')
    random.Random(a.seed).shuffle(games)                 # same shuffle as wm_collect.py, so shards do not overlap
    games = games[a.shard * a.n_games:(a.shard + 1) * a.n_games]
    jobs = [{'gamefile': g, 'url': a.sglang, 'lora': a.lora, 'temperature': 1.0, 'max_new_tokens': 256,
             'max_steps': a.max_steps, 'model': a.model, 'K': a.K, 'value_url': a.value_url, 'wm_url': a.wm_url,
             'wm_lora': a.wm_lora} for g in games]
    import multiprocessing as mp
    t0 = time.time(); n = err = 0
    with open(a.out, 'w', buffering=1) as f, mp.get_context('fork').Pool(a.workers, maxtasksperchild=1) as p:
        for res in p.imap_unordered(episode, jobs):
            if 'error' in res:
                err += 1
            n += 1
            f.write(json.dumps(res) + '\n')
    print(f'shard {a.shard}: {n} games ({err} errors) {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
