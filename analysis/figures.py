#!/usr/bin/env python3
"""Figures of the paper that are computed from results/ (the same numbers; a plain style with the DejaVu Sans font):
  frontier    Figures 4 and 12: greedy success against the real steps executed since the warm-up (means over runs):
              every method at the end of the protocol (ALFWorld iteration 36, ScienceWorld 60; CAPO on ScienceWorld:
              iteration 18, as in Table 2), and the two WM-PPO variants at every evaluated iteration
  gate        Figure 8: gate weights of the three margin bins during training (mean over the three runs; band: range of
              the largest bin), WM-PPO and the policy-channel-only variant
  ablation    Figure 9: success during training of the gate-ablation runs (ScienceWorld, first run of each method)
  curves      Figure 13: greedy success during training against real steps (runs and their mean)
Figure 1 is made by analysis/diagnostics.py.
   python3 analysis/figures.py [--out figures] [--only frontier,gate,ablation,curves]"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaldata as E  # noqa: E402

NAMES = {'wmppo': 'WM-PPO', 'wmppo_policy_only': 'WM-PPO (policy channel only)', 'ppo160': 'PPO-160', 'ppo256': 'PPO-256',
         'grpo': 'GRPO', 'gigpo': 'GiGPO', 'capo': 'CAPO', 'sopo': 'SO-PPO', 'stppo': 'ST-PPO', 'qppo': 'Q-PPO',
         'wide_tree_ppo': 'Wide-Tree-PPO', 'fixed_wm': 'Fixed WM', 'paw': 'PaW', 'majority_vote': 'Majority-vote labels',
         'ocsd_self': 'OCSD (self)', 'ocsd_32b': 'OCSD (32B)', 'opd_32b': 'OPD (32B)'}
GAMES = {'alfworld': (140, 134), 'scienceworld': (140, 144)}
GREEN, BLUE, GREY = '#2E7D5B', '#2F6DB5', '#9AA5B1'


def plt_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def frontier(out, D):
    plt = plt_setup()
    its = {'alfworld': [24, 36], 'scienceworld': [24, 36, 48, 60]}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for ax, env in zip(axes, ('alfworld', 'scienceworld')):
        for m, label in NAMES.items():
            pts = []
            end = [18] if (env, m) == ('scienceworld', 'capo') else its[env][-1:]   # CAPO on ScienceWorld: iteration 18
            for it in (its[env] if m.startswith('wmppo') else end):
                v = D.method(env, m, it)
                if v and v['steps'] is not None:
                    pts.append((v['steps'] / 1000, 100 * v['greedy'], it))
            if not pts:
                continue
            xs, ys, labs = zip(*pts)
            col = GREEN if m == 'wmppo' else BLUE if m == 'wmppo_policy_only' else GREY
            ax.plot(xs, ys, '-o' if m.startswith('wmppo') else 'o', color=col, ms=3, lw=1.2, label=label if m.startswith('wmppo') else None)
            if not m.startswith('wmppo'):
                ax.annotate(label, (xs[-1], ys[-1]), fontsize=5, xytext=(2, 2), textcoords='offset points', color='#555555')
        ax.set_title(env.capitalize().replace('world', 'World'), loc='left')
        ax.set_xlabel('Real steps since the warm-up (thousands)'); ax.set_ylabel('Greedy success (%)')
    axes[0].legend(frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'frontier.pdf')); print('wrote', os.path.join(out, 'frontier.pdf'))


def gate(out):
    plt = plt_setup()
    W = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(os.path.join(E.RESULTS, 'gate.csv'))):
        if r['method'] in ('wmppo', 'wmppo_policy_only') and r['w'] not in ('', 'null', 'None'):
            W[(r['env'], r['method'])][int(r['it'])].append(json.loads(r['w']))
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.4), sharey=True)
    for j, m in enumerate(('wmppo', 'wmppo_policy_only')):
        for i, (env, last) in enumerate((('scienceworld', 60), ('alfworld', 36))):
            ax = axes[i][j]; d = W[(env, m)]
            its = sorted(k for k in d if k <= last)
            col = GREEN if m == 'wmppo' else BLUE
            for b, ls in zip(range(3), (':', '--', '-')):
                ax.plot(its, [E.mean(w[b] for w in d[t]) for t in its], ls, color=col, lw=1)
            ax.fill_between(its, [min(w[2] for w in d[t]) for t in its], [max(w[2] for w in d[t]) for t in its], color=col, alpha=0.15)
            ax.set_title(f'{NAMES[m]}, {env}', loc='left'); ax.set_ylim(-0.05, 1.05)
            if i == 1:
                ax.set_xlabel('Iteration')
            if j == 0:
                ax.set_ylabel('Gate weight')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'gate_weights.pdf')); print('wrote', os.path.join(out, 'gate_weights.pdf'))


def curves_rows():
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(E.RESULTS, 'curves.csv'))):
        if r['eval_dev'] not in ('', 'None'):
            nd, nt = GAMES[r['env']]
            pooled = (nd * float(r['eval_dev']) + nt * float(r['eval_test'])) / (nd + nt)
            rows[(r['env'], r['run'])].append((int(r['it']), pooled, float(r['cum_exec']) if r['cum_exec'] else None,
                                               float(r['eval_dev']), float(r['eval_test'])))
    return {k: sorted(v) for k, v in rows.items()}


def ablation(out):
    plt = plt_setup()
    C = curves_rows()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    for name, col, ls in (('wmppo_run1', GREEN, '-'), ('gate_ablation_no_gate_run1', '#B5543A', '--'),
                          ('gate_ablation_top_bin_run1', '#8E6BBF', '-.'), ('ppo160_run1', GREY, ':')):
        pts = [p for p in C[('scienceworld', name)] if p[0] >= 12]
        for ax, idx, title in ((axes[0], 3, 'dev'), (axes[1], 4, 'test')):
            ax.plot([p[0] for p in pts], [100 * p[idx] for p in pts], ls, color=col, lw=1.2, label=name.replace('_run1', ''))
            ax.set_title(f'ScienceWorld, {title} games', loc='left'); ax.set_xlabel('Iteration')
    axes[0].set_ylabel('Greedy success (%)'); axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'gate_ablation.pdf')); print('wrote', os.path.join(out, 'gate_ablation.pdf'))


def curves(out):
    plt = plt_setup()
    C = curves_rows()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, env, last in ((axes[0], 'alfworld', 55), (axes[1], 'scienceworld', 60)):
        for m, col in (('wmppo', GREEN), ('wmppo_policy_only', BLUE), ('ppo160', GREY), ('ppo256', '#666666')):
            runs = [C[(env, f'{m}_run{k}')] for k in (1, 2, 3) if (env, f'{m}_run{k}') in C]
            for rr in runs:
                ax.plot([p[2] / 1000 for p in rr if p[2] is not None], [100 * p[1] for p in rr if p[2] is not None], color=col, lw=0.4, alpha=0.4)
            its = sorted(set.intersection(*[{p[0] for p in rr if p[2] is not None and p[0] <= last} for rr in runs]))
            mx = [E.mean(dict((p[0], p[2]) for p in rr)[t] for rr in runs) / 1000 for t in its]
            my = [100 * E.mean(dict((p[0], p[1]) for p in rr)[t] for rr in runs) for t in its]
            ax.plot(mx, my, color=col, lw=1.4, label=f'{NAMES[m]} ({len(runs)})')
        ax.set_title(env.capitalize().replace('world', 'World'), loc='left')
        ax.set_xlabel('Real steps since the warm-up (thousands)'); ax.set_ylabel('Greedy success (%)')
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'curves.pdf')); print('wrote', os.path.join(out, 'curves.pdf'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(E.ROOT, 'figures'))
    ap.add_argument('--only', default='frontier,gate,ablation,curves')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from check_tables import Data
    D = Data()
    for name in a.only.split(','):
        {'frontier': lambda: frontier(a.out, D), 'gate': lambda: gate(a.out), 'ablation': lambda: ablation(a.out),
         'curves': lambda: curves(a.out)}[name]()


if __name__ == '__main__':
    main()
