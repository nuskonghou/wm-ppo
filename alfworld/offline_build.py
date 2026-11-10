"""Offline update of Section 2.3, step 2: PPO-clip samples toward a label (la = world model + critic, sc = majority).
At each recorded step (>= 2 distinct sampled actions), sample k gets the advantage 1[action_k == label] - share(label),
which sums to zero over the K samples (Equation 4). The samples are trained for one epoch with train_lora.py.
   python offline_build.py --label la|sc --out samples.jsonl collect_*.jsonl"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument('--label', choices=['la', 'sc'], required=True)
ap.add_argument('--out', required=True)
ap.add_argument('files', nargs='+')
a = ap.parse_args()
n_states = n_rows = n_games = same = 0; won = []
with open(a.out, 'w') as f:
    for fn in a.files:
        for line in open(fn):
            g = json.loads(line)
            if 'error' in g:
                continue
            n_games += 1; won.append(g['reward'])
            for st in g['steps']:
                lab = st[a.label]; K = len(st['acts'])
                share = sum(x == lab for x in st['acts']) / K
                n_states += 1; same += st['la'] == st['sc']
                for k in range(K):
                    adv = float(st['acts'][k] == lab) - share
                    if st['resp_ids'][k] and abs(adv) > 0:
                        f.write(json.dumps({'prompt_ids': st['prompt_ids'], 'resp_ids': st['resp_ids'][k],
                                            'old_lp': st['old_lp'][k], 'adv': adv}) + '\n'); n_rows += 1
print(json.dumps({'label': a.label, 'games': n_games, 'behaviour_SR': sum(won) / max(1, len(won)), 'states': n_states,
                  'rows': n_rows, 'la_equals_sc': same / max(1, n_states)}))
