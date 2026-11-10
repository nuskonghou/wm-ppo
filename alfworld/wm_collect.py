"""Data of the diagnostic world model (Section 2.2): transitions under one fixed policy, including one-step executions of
candidates the policy did not choose.

The policy plays training games (temperature 1). At every step it samples K candidate turns; the first sample is executed
(plain sampling behaviour). For every distinct candidate action the TRUE consequence is recorded: the executed one from
the main env, the others by a one-step sandbox (fresh env + prefix replay). One record per (state, candidate):
  task, room (initial description), hist [[obs, action], ...] (full history), obs (current), adm (current admissible),
  action, next_obs, next_adm, done, won, executed, count (how many of the K samples chose it), episode_reward.
Run with ALF_SHARED_DOWNWARD=0 (sandbox envs coexist with the main env) and a large TMPDIR."""
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
from ct_infer import prompt, sandbox  # noqa: E402
from env import AlfEnv, game_files  # noqa: E402
from prompts import parse_action, split_task  # noqa: E402
from wmppo import paths  # noqa: E402


def episode(job):
    try:
        tok = rollout._tok(job['model'])
        env = AlfEnv(job['gamefile'], max_steps=job['max_steps'])
        obs, adm = env.reset(); task, room = split_task(obs)
        hist, cur, r, done = [], room, 0.0, False
        recs = []
        while not done:
            ids = prompt(tok, task, room, hist, cur, adm)
            samples = [parse_action(rollout.generate(job, ids)[0], adm)[0] for _ in range(job['K'])]
            counts = collections.Counter(samples)
            executed = samples[0]
            acts = [a for _, a in hist]
            for a in dict.fromkeys(samples):
                if a == executed:
                    continue
                t_, r_, h_, c_, adm_, rew_, done_ = sandbox(job['gamefile'], job['max_steps'], acts + [a])
                recs.append({'action': a, 'next_obs': c_, 'next_adm': adm_, 'done': done_, 'won': bool(done_ and rew_ > 0),
                             'executed': False, 'count': counts[a], 'state': (task, room, [list(x) for x in hist], cur, adm)})
            hist.append((cur, executed))
            cur, adm_next, r, done = env.step(executed)
            recs.append({'action': executed, 'next_obs': cur, 'next_adm': adm_next, 'done': done, 'won': bool(done and r > 0),
                         'executed': True, 'count': counts[executed],
                         'state': (task, room, [list(x) for x in hist[:-1]], hist[-1][0], adm)})
            adm = adm_next
        try: env.env.close()
        except Exception: pass  # noqa: E722
        out = []
        for x in recs:
            task_, room_, h_, o_, adm_ = x.pop('state')
            out.append(dict(x, task=task_, room=room_, hist=h_, obs=o_, adm=adm_, episode_reward=r,
                            game=job['gamefile'], t=len(h_)))
        return out
    except Exception as e:  # noqa: BLE001
        return [{'game': job['gamefile'], 'error': repr(e)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora', required=True)
    ap.add_argument('--sglang', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--n-games', type=int, default=500)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--max-steps', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default=paths.MODEL)
    a = ap.parse_args()
    games = game_files(a.split)
    random.Random(a.seed).shuffle(games)
    games = games[a.shard * a.n_games:(a.shard + 1) * a.n_games] if a.split == 'train' else games
    jobs = [{'gamefile': g, 'url': a.sglang, 'lora': a.lora, 'temperature': 1.0, 'max_new_tokens': 256,
             'max_steps': a.max_steps, 'model': a.model, 'K': a.K} for g in games]
    import multiprocessing as mp
    t0 = time.time(); n = err = 0
    with open(a.out, 'w') as f, mp.get_context('fork').Pool(a.workers, maxtasksperchild=1) as p:
        for recs in p.imap_unordered(episode, jobs):
            for x in recs:
                if 'error' in x:
                    err += 1; continue
                f.write(json.dumps(x) + '\n'); n += 1
    print(f'{a.split} shard {a.shard}: {len(games)} games, {n} transitions, {err} failed games, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
