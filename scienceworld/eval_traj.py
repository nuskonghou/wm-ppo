"""Greedy evaluation of one checkpoint on the ScienceWorld evaluation variations (dev = valid_seen, test = valid_unseen)
with the full trajectory of every episode saved (the evaluation protocol keeps only per-game success); used
for the behaviour analyses. Same rollout code, same prompts, temperature 0, 40 steps.
   python eval_traj.py --adapter <dir> --name <lora> --sglang <url> --out <jsonl> [--workers 16]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
from env import game_files
import rollout
from wmppo import paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--sglang', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--model', default=paths.MODEL)
    a = ap.parse_args()
    try:
        requests.post(a.sglang + '/load_lora_adapter', json={'lora_name': a.name, 'lora_path': a.adapter}, timeout=600).raise_for_status()
    except requests.HTTPError:
        pass
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['game'] for l in open(a.out)}
    jobs = []
    for sp in ('valid_seen', 'valid_unseen'):
        for gf in game_files(sp):
            gid = sp + '/' + gf
            if gid in done: continue
            jobs.append({'gamefile': gf, 'prefix': [], 'url': a.sglang, 'lora': a.name, 'avoid_first': None, 'k_alt': 8,
                         'temperature': 0.0, 'max_new_tokens': 256, 'max_steps': 40, 'model': a.model, 'tag': gid, 'K': 1})
    tok = rollout._tok(a.model)
    pool = rollout.Pool(a.workers)
    with open(a.out, 'a') as f:
        for i in range(0, len(jobs), 48):
            for e in pool.run(jobs[i:i + 48]):
                if 'error' in e:
                    f.write(json.dumps({'game': e['tag'], 'error': e['error']}) + '\n'); continue
                steps = [{'t': tu['t'], 'obs': tu['obs'], 'action': tu['action'], 'response': tok.decode(tu['resp_ids'])} for tu in e['turns']]
                f.write(json.dumps({'game': e['tag'], 'reward': e['reward'], 'steps': e['steps'], 'final_obs': e['final_obs'],
                                    'trajectory': steps}) + '\n')
            f.flush()
    pool.close()
    print('done', a.out, flush=True)

if __name__ == '__main__':
    main()
