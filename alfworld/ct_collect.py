"""Pairwise test of Section 2.1, step 1: real sibling states with Monte-Carlo action values, under one fixed policy.

For every evaluation game (valid_seen + valid_unseen):
  A. play one episode with the policy (temperature 1) and pick a step t uniformly at random (--seed);
  B. at that state sample K candidate turns from the same policy; keep the distinct parsed actions;
  C. for each distinct action a, execute it for real (deterministic replay of the prefix, then a) and run R independent
     continuations with the policy from the resulting state; Q_hat(a) = mean reward. A terminal action has an exact Q.
Output <out>/<arm>.json (<arm>_s<seed>.json for seed > 0): one record per state with the parent prompt, and per candidate
the action, its count among the K samples, whether it ended the episode, the R rewards, and the prompt of the state it
leads to (for the critic, ct_score.py)."""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rollout  # noqa: E402
from env import AlfEnv, game_files  # noqa: E402
from prompts import messages, parse_action, split_task  # noqa: E402
from wmppo import paths  # noqa: E402


def sample_candidates(job):
    """Replay the prefix, build the turn prompt, draw K samples, return the parent prompt and the parsed actions."""
    try:
        tok = rollout._tok(job['model'])
        env = AlfEnv(job['gamefile'], max_steps=job['max_steps'])
        obs, adm = env.reset()
        task, room = split_task(obs)
        hist, cur, done = [], room, False
        for a in job['prefix']:
            hist.append((cur, a))
            cur, adm, r, done = env.step(a)
            if done:
                break
        try:
            env.env.close()
        except Exception:  # noqa: BLE001
            pass
        if done:
            return {'tag': job['tag'], 'error': 'prefix ended the episode'}
        ids = tok.apply_chat_template(messages(task, room, hist, cur, adm), tokenize=True, add_generation_prompt=True,
                                      return_dict=False)
        ids = ids if isinstance(ids, list) else ids['input_ids']
        acts = []
        for _ in range(job['K']):
            text, _, _ = rollout.generate(job, ids)
            acts.append(parse_action(text, adm))
        return {'tag': job['tag'], 'parent_prompt_ids': ids, 'samples': acts}
    except Exception as e:  # noqa: BLE001
        return {'tag': job['tag'], 'error': repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--lora', required=True, help='adapter name already loaded on --sglang')
    ap.add_argument('--sglang', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--R', type=int, default=3)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--max-steps', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default=paths.MODEL)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = {'url': a.sglang, 'lora': a.lora, 'temperature': 1.0, 'max_new_tokens': 256, 'max_steps': a.max_steps,
            'model': a.model}
    games = game_files('valid_seen') + game_files('valid_unseen')
    pool = rollout.Pool(a.workers)
    t0 = time.time()
    eps = pool.run([dict(base, gamefile=g, prefix=[], tag=i) for i, g in enumerate(games)])
    rng = random.Random(a.seed)
    states = []
    for e in eps:
        if 'error' in e or not e['actions']:
            continue
        t = rng.randrange(len(e['actions']))
        states.append({'game': e['gamefile'], 'split': 'valid_seen' if '/valid_seen/' in e['gamefile'] else 'valid_unseen',
                       't': t, 'prefix': e['actions'][:t], 'episode_reward': e['reward']})
    print(f'[{time.strftime("%H:%M:%S")}] {a.arm}: {len(states)} states from {len(eps)} episodes ({time.time() - t0:.0f}s)',
          flush=True)
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.workers, maxtasksperchild=4) as p:
        cand = p.map(sample_candidates, [dict(base, gamefile=s['game'], prefix=s['prefix'], tag=i, K=a.K)
                                         for i, s in enumerate(states)], chunksize=1)
    jobs = []
    for i, c in enumerate(cand):
        s = states[i]
        if 'error' in c:
            s['error'] = c['error']; continue
        s['parent_prompt_ids'] = c['parent_prompt_ids']
        counts = {}
        for act, valid in c['samples']:
            counts.setdefault(act, [0, valid]); counts[act][0] += 1
        s['cands'] = [{'action': act, 'count': n, 'valid': v, 'rewards': [], 'terminal': False, 'child_prompt_ids': None}
                      for act, (n, v) in counts.items()]
        if len(s['cands']) < 2:
            continue
        for k, cd in enumerate(s['cands']):
            for r in range(a.R):
                jobs.append(dict(base, gamefile=s['game'], prefix=s['prefix'] + [cd['action']], tag=(i, k, r)))
    print(f'[{time.strftime("%H:%M:%S")}] {a.arm}: {sum(1 for s in states if len(s.get("cands", [])) >= 2)} states '
          f'with >= 2 distinct actions; {len(jobs)} continuations', flush=True)
    for res in pool.run(jobs):
        if 'error' in res:
            continue
        i, k, r = res['tag']
        cd = states[i]['cands'][k]
        cd['rewards'].append(res['reward'])
        if not res['turns']:
            cd['terminal'] = True                  # the action itself ended the episode
        elif cd['child_prompt_ids'] is None:
            cd['child_prompt_ids'] = res['turns'][0]['prompt_ids']
    pool.close()
    json.dump({'arm': a.arm, 'lora': a.lora, 'K': a.K, 'R': a.R, 'states': states}, open(os.path.join(a.out, f'{a.arm}.json' if a.seed == 0 else f'{a.arm}_s{a.seed}.json'), 'w'))
    print(f'[{time.strftime("%H:%M:%S")}] {a.arm}: done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
