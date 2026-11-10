#!/usr/bin/env python3
"""Numbers of Section 2 and Figure 1 (ALFWorld diagnostics) from results/diagnostics/.

Policy and critic: the PPO run continued from the warm-up to iteration 24; 274 evaluation games, one episode per game
(ct_infer.py rules; the offline update: two temperature-1 runs per policy).
  (a) candidate selection: first sample, majority of four, the critic's value of the successor predicted by the untrained
      or the trained world model, or of the true successor (plus the controls of Appendix C)
  (b) pairwise test: 195 candidate pairs with different Monte-Carlo estimates (three random states per game, three
      continuations per candidate); accuracy of the critic's order in the lower half and the top quarter of its value
      gaps, and on the pairs where one candidate succeeded in all three continuations and the other in none
  (c) offline update: policy-only success after one PPO-clip epoch with labels from the world model and the critic or
      from majority vote, against the policy before the update
Differences are per-game paired; SE from the per-game differences; p: exact two-sided sign test on the games where the
two differ (ct_paired.py), and the z-test p for comparison.
   python3 analysis/diagnostics.py [--plot figures/diagnostics.pdf] [--runs DIR]
--runs reads the outputs of a new run of the diagnostics instead (docs/REPRODUCE.md): the infer_*.jsonl files of
ct_infer.py and ppo.scored.json, ppo_s1.scored.json, ppo_s2.scored.json of ct_collect.py / ct_score.py in DIR."""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaldata as E  # noqa: E402

D = os.path.join(E.RESULTS, 'diagnostics')
RUNS = None          # set by --runs: directory with the raw outputs of ct_infer.py and ct_score.py
RAW = {'select_first_sample_r1': 'infer_ppo_random', 'select_first_sample_r2': 'infer_ppo_random_r2',
       'select_majority': 'infer_ppo_sc', 'select_peek': 'infer_ppo_peek',
       'select_critic_true_next_state': 'infer_ppo_critic_only',
       'select_critic_true_obs_parent_actions': 'infer_ppo_critic_parent_adm',
       'select_critic_true_obs_masked': 'infer_ppo_critic_masked',
       'select_critic_untrained_wm': 'infer_ppo_critic_wm_untrained_wm', 'select_critic_trained_wm': 'infer_ppo_critic_wm_trained_wm',
       'offline_update_wm_critic_r1': 'infer_offline_la_random_r1', 'offline_update_wm_critic_r2': 'infer_offline_la_random_r2',
       'offline_update_majority_r1': 'infer_offline_sc_random_r1', 'offline_update_majority_r2': 'infer_offline_sc_random_r2'}


def load(name):
    path = os.path.join(RUNS, RAW[name] + '.jsonl') if RUNS else os.path.join(D, f'{name}.jsonl')
    return {x['game']: float(x['reward'] > 0) for x in map(json.loads, open(path)) if 'error' not in x}


def avg(*names):
    ds = [load(n) for n in names]
    return {k: E.mean(d[k] for d in ds) for k in ds[0]}


def sign_p(w, l):
    n, k = w + l, min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def compare(A, B):
    d = [A[k] - B[k] for k in sorted(set(A) & set(B))]
    mu = E.mean(d); se = (sum((v - mu) ** 2 for v in d) / (len(d) - 1) / len(d)) ** 0.5
    w, l = sum(v > 0 for v in d), sum(v < 0 for v in d)
    return mu, se, sign_p(w, l), math.erfc(abs(mu / se) / 2 ** 0.5)


def pairwise():
    if RUNS:
        states = [st for f in ('ppo', 'ppo_s1', 'ppo_s2') for st in json.load(open(os.path.join(RUNS, f'{f}.scored.json')))['states']]
    else:
        states = json.load(open(os.path.join(D, 'pairwise_test_states.json')))
    pairs, top2 = [], []
    for s in states:
        cs = [c for c in s['cands'] if c['rewards']]
        if len(cs) < 2:
            continue
        nt = [c for c in cs if not c['terminal'] and c.get('v') is not None]
        vs = sorted((c['v'] for c in nt), reverse=True)
        if len(vs) >= 2:
            top2.append(vs[0] - vs[1])
        for i in range(len(nt)):
            for j in range(i + 1, len(nt)):
                a, b = nt[i], nt[j]
                qa, qb = E.mean(a['rewards']), E.mean(b['rewards'])
                if qa == qb:
                    continue
                acc = float((a['v'] > b['v']) == (qa > qb)) if a['v'] != b['v'] else 0.5
                extreme = sorted([qa, qb]) == [0.0, 1.0] and len(a['rewards']) == len(b['rewards']) == 3
                pairs.append((abs(a['v'] - b['v']), acc, extreme))
    pairs.sort(key=lambda x: x[0])
    n = len(pairs); acc = [p[1] for p in pairs]; gaps = [p[0] for p in pairs]
    pos = 0.75 * (n - 1); q75 = gaps[int(pos)] + (gaps[min(int(pos) + 1, n - 1)] - gaps[int(pos)]) * (pos - int(pos))
    top2.sort()
    return {'pairs': n, 'accuracy': E.mean(acc), 'lower_half': E.mean(acc[:n // 2]),
            'top_quarter': E.mean(a for g, a in zip(gaps, acc) if g > q75),
            'extreme_correct': int(sum(p[1] for p in pairs if p[2])), 'extreme_n': sum(p[2] for p in pairs),
            'median_top_two_gap': top2[len(top2) // 2] if len(top2) % 2 else (top2[len(top2) // 2 - 1] + top2[len(top2) // 2]) / 2,
            'states_with_two_values': len(top2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plot', default=None, help='write Figure 1 to this file (pdf or png)')
    ap.add_argument('--runs', default=None, help='directory with the outputs of a new run of the diagnostics')
    a = ap.parse_args()
    global RUNS
    RUNS = a.runs
    sel = {'first sample': load('select_first_sample_r1'), 'majority vote': load('select_majority'),
           'critic + untrained WM': load('select_critic_untrained_wm'), 'critic + trained WM': load('select_critic_trained_wm'),
           'critic + true next state': load('select_critic_true_next_state')}
    ctrl = {'peek control (complete the task if possible, else majority)': load('select_peek'),
            'true observation, admissible actions of before the step': load('select_critic_true_obs_parent_actions'),
            'true observation without newly revealed objects': load('select_critic_true_obs_masked')}
    print('(a) candidate selection (success, %)')
    for k, v in list(sel.items()) + list(ctrl.items()):
        print(f'    {k:62s} {100 * E.mean(v.values()):.1f}')
    for x, y in (('critic + true next state', 'majority vote'), ('critic + trained WM', 'majority vote'),
                 ('critic + trained WM', 'critic + untrained WM')):
        mu, se, ps, pz = compare(sel[x], sel[y])
        print(f'    {x} - {y}: {100 * mu:+.1f} points (SE {100 * se:.1f}; sign test p = {ps:.2g}, z-test p = {pz:.2g})')
    pw = pairwise()
    print(f"(b) pairwise test: {pw['pairs']} pairs, critic order correct {100 * pw['accuracy']:.0f}%; lower half of the value "
          f"gaps {100 * pw['lower_half']:.0f}%, top quarter {100 * pw['top_quarter']:.0f}%; 3/3 vs 0/3 pairs "
          f"{pw['extreme_correct']}/{pw['extreme_n']}; median gap between the two highest critic values of a state "
          f"{pw['median_top_two_gap']:.3f} ({pw['states_with_two_values']} states)")
    before = avg('select_first_sample_r1', 'select_first_sample_r2')
    la = avg('offline_update_wm_critic_r1', 'offline_update_wm_critic_r2')
    mv = avg('offline_update_majority_r1', 'offline_update_majority_r2')
    print(f'(c) offline update (policy alone, two runs per game): before {100 * E.mean(before.values()):.1f}, '
          f'world-model + critic labels {100 * E.mean(la.values()):.1f}, majority-vote labels {100 * E.mean(mv.values()):.1f}')
    mu, se, ps, pz = compare(la, before)
    print(f'    world-model + critic labels - before: {100 * mu:+.1f} points (SE {100 * se:.1f}; sign test p = {ps:.2g}, '
          f'z-test p = {pz:.2g})')
    if a.plot:
        plot(a.plot, sel, pw, before, mv, la)


def plot(path, sel, pw, before, mv, la):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3))
    panels = [('a  Candidate selection', [(k, E.mean(v.values())) for k, v in sel.items()]),
              ('b  Critic order vs value gap', [('lower half', pw['lower_half']), ('top quarter', pw['top_quarter']),
                                                ('3/3 vs 0/3', pw['extreme_correct'] / pw['extreme_n'])]),
              ('c  Offline update', [('before', E.mean(before.values())), ('majority vote', E.mean(mv.values())),
                                     ('WM + critic', E.mean(la.values()))])]
    for ax, (title, items) in zip(axes, panels):
        labels, vals = zip(*items)
        y = list(range(len(vals)))[::-1]
        ax.barh(y, [100 * v - 40 for v in vals], left=40, height=0.5, color='#2E7D5B')
        for yi, v in zip(y, vals):
            ax.text(100 * v + 0.5, yi, f'{100 * v:.0f}' if title[0] == 'b' else f'{100 * v:.1f}', va='center')
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(40, 100); ax.set_title(title, loc='left')
        ax.set_xlabel('Success (%)' if title[0] != 'b' else 'Correct order (%)')
    axes[1].axvline(50, ls=':', color='grey')
    axes[2].axvline(100 * E.mean(sel['critic + trained WM'].values()), ls='--', color='grey')
    fig.tight_layout(); fig.savefig(path)
    print('wrote', path)


if __name__ == '__main__':
    main()
