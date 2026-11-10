#!/usr/bin/env python3
"""Recompute the result tables of the paper from results/ and compare every cell with the paper's tables
(results/reference_tables/*.tex): Table 2 (main), Tables 7-8 (baseline settings), Tables 10-13 (all methods and
iterations), Table 15 (runs), Table 16 (ScienceWorld task types), Table 18 (Sokoban) and Table 21 (gate ablation).
Table 17 (computation per iteration) is recomputed from cost_per_iteration.csv and real_steps.csv.
Marks: dagger = WM-PPO significantly higher, double dagger = the row significantly higher (per-game paired z-test,
p < 0.05, against WM-PPO at the same iteration; Table 21: against the first WM-PPO run). Bold face is not compared.
   python3 analysis/check_tables.py [--show]        exit code 1 if any cell differs"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaldata as E  # noqa: E402

REF = os.path.join(E.RESULTS, 'reference_tables')
UP, DN = '†', '‡'


def clean(cell):
    c = cell.strip()
    c = re.sub(r'\\textbf\{(.*)\}', r'\1', c)
    c = c.replace(r'\sigup{}', UP).replace(r'\sigdn{}', DN).replace('$^{b}$', '^b').replace('$^{s}$', '^s')
    c = re.sub(r'\\rlap\{(.*?)\}', r'\1', c)
    c = re.sub(r'\\,\{\\scriptsize\$\\pm\$([0-9.]+)\}', r'±\1', c)
    c = c.replace('$^{b}$', '^b').replace('$^{s}$', '^s').replace('$>$', '>').replace(r'\\', '').strip()
    return c


def rows(table):
    out = []
    for line in open(os.path.join(REF, f'{table}.tex')):
        line = line.strip()
        if not line.endswith(r'\\') or line.startswith(r'\multicolumn') or line.startswith(('Method', 'Baseline', 'Environment', 'Task', ' &')):
            continue
        r = [clean(c) for c in line[:-2].split('&')]
        if not any(r'\multicolumn' in c for c in r):      # column headers
            out.append(r)
    return out


def pct(x):
    return '--' if x is None else f'{100 * x:.1f}'


def mark(p):
    if not p or p['p'] >= 0.05:
        return ''
    return UP if p['diff'] > 0 else DN


class Data:
    def __init__(self):
        self.ck = {env: E.load_release(env) for env in ('alfworld', 'scienceworld', 'sokoban')}
        self.steps = E.real_steps()
        self.steps_it = E.exec_steps()

    def runs(self, env, method, it, which=None):
        names = sorted({r for r, i in self.ck[env] if i == it and re.fullmatch(rf'{re.escape(method)}_run\d+', r)},
                       key=lambda r: int(r.rsplit('run', 1)[1]))
        if which:
            names = [f'{method}_run{w}' for w in which]
        return [self.ck[env][(n, it)] for n in names], names

    def method(self, env, method, it, which=None):
        c, names = self.runs(env, method, it, which)
        if not c:
            return None
        v = E.method_values(c)
        st = [self.steps.get((env, n, it)) for n in names]
        v['steps'] = E.mean(st) if all(x is not None for x in st) else None
        v['names'] = names
        return v


def cell(v, key, ref=None, sd=False, mk=True):
    if v is None:
        return '--'
    s = pct(v[key])
    if mk and ref is not None:
        s += mark(E.paired(ref[f'_{"G" if key == "greedy" else "S"}'], v[f'_{"G" if key == "greedy" else "S"}']))
    if sd and v['n_runs'] >= 2:
        s += f'±{100 * E.stdev(v[key + "_runs"]):.1f}'
    return s


def kstep(v):
    return '--' if v is None or v['steps'] is None else f'{v["steps"] / 1000:.0f}k'


class Checker:
    def __init__(self, show):
        self.show = show; self.n = self.bad = 0

    def cmp(self, table, row, got, want):
        self.n += 1
        if got != want:
            self.bad += 1
            print(f'  MISMATCH {table} | {row}: computed {got!r}, paper {want!r}')
        elif self.show:
            print(f'  ok {table} | {row}: {got}')


MAIN = [('wmppo', 'wmppo'), ('wmppo_policy_only', 'wmppo_policy_only'), ('ppo160', 'ppo160'), ('ppo256', 'ppo256'),
        ('grpo', 'grpo'), ('gigpo', 'gigpo'), ('capo', 'capo'), ('sopo', 'sopo'), ('stppo', 'stppo'), ('qppo', 'qppo'),
        ('wide_tree_ppo', 'wide_tree_ppo'), ('fixed_wm', 'fixed_wm'), ('paw', 'paw'), ('majority_vote', 'majority_vote'),
        ('ocsd_self', 'ocsd_self'), ('ocsd_32b', 'ocsd_32b'), ('opd_32b', 'opd_32b')]


def check_main(D, C):
    ref = rows('main')
    refA, refS, refS18 = D.method('alfworld', 'wmppo', 36), D.method('scienceworld', 'wmppo', 60), D.method('scienceworld', 'wmppo', 18)
    for (ma, ms), r in zip(MAIN, ref):
        a = D.method('alfworld', ma, 36)
        s18 = ms == 'capo'
        s = D.method('scienceworld', ms, 18 if s18 else 60)
        rs = refS18 if s18 else refS
        first = ma == 'wmppo'
        got = [f"{a['n_runs']} / {s['n_runs']}",
               cell(a, 'greedy', None if first else refA, True), cell(a, 'pass1', None if first else refA, True),
               cell(s, 'greedy', None if first else rs, True) + ('^b' if s18 else ''),
               cell(s, 'pass1', None if first else rs, True) + ('^b' if s18 else '')]
        for j, (g, w) in enumerate(zip(got, r[1:])):
            C.cmp('Table 2', f'{r[0]} col {j + 2}', g, w)


FULL = {'full_alfworld_a': ('alfworld', [24, 36], [
            ('wmppo', None), ('wmppo_policy_only', None), ('ppo160', None), ('ppo256', None), ('grpo', None), ('grpo_g4', None),
            ('gigpo', None), ('hgpo', None), ('tree_grpo', None), ('capo', None), ('sopo', None), ('stppo', None),
            ('vapo_lite', None), ('qppo', None), ('wide_tree_ppo', None), ('fixed_wm', None), ('paw', None),
            ('majority_vote', None), ('ocsd_self', None), ('ocsd_32b', None), ('opd_32b', None), ('wmppo', ['2_rerun'])]),
        'full_alfworld_b': ('alfworld', [55], [
            ('wmppo', None), ('wmppo_policy_only', None), ('ppo160', None), ('ppo256', None), ('wide_tree_ppo', None),
            ('fixed_wm', None), ('paw', None), ('wmppo', ['2_rerun'])]),
        'full_scienceworld_a': ('scienceworld', [24, 36], None),
        'full_scienceworld_b': ('scienceworld', [48, 60], None)}
SW_FULL = [('wmppo', None), ('wmppo_policy_only', None), ('wmppo_terminal_value', None), ('ppo160', None), ('ppo256', None),
           ('grpo', None), ('grpo_g4', None), ('gigpo', None), ('gigpo_g4', None), ('sopo', None), ('sopo_clip0.2', None),
           ('stppo', None), ('qppo', None), ('wide_tree_ppo', None), ('fixed_wm', None), ('paw', None), ('paw_coef1.0', None),
           ('paw_coef0.25', None), ('majority_vote', None), ('ocsd_self', None), ('ocsd_32b', None), ('opd_32b', None),
           ('ppo256', ['1_rerun']), ('wmppo_policy_only', ['3_rerun'])]


def check_full(D, C):
    for table, (env, its, spec) in FULL.items():
        spec = spec or SW_FULL
        ref = rows(table)
        assert len(ref) == len(spec), (table, len(ref), len(spec))
        for (m, which), r in zip(spec, ref):
            got = []
            for it in its:
                v = D.method(env, m, it, which)
                base = D.method(env, 'wmppo', it)
                no_test = m == 'wmppo' and which is None or which is not None
                got += [kstep(v), cell(v, 'greedy', None if no_test else base), cell(v, 'pass1', None if no_test else base),
                        pct(v['pass3'])]
            for j, (g, w) in enumerate(zip(got, r[1:])):
                C.cmp(f'{table}', f'{r[0]} col {j + 2}', g, w)


SEEDS = [('alfworld', 36, ['wmppo', 'wmppo_policy_only', 'ppo160', 'ppo256', 'qppo', 'wide_tree_ppo', 'ocsd_self']),
         ('alfworld', 55, ['wmppo', 'wmppo_policy_only', 'ppo160', 'ppo256', 'wide_tree_ppo']),
         ('scienceworld', 60, ['wmppo', 'wmppo_policy_only', 'ppo160', 'ppo256', 'qppo', 'wide_tree_ppo'])]


def check_seeds(D, C):
    ref = iter(rows('seeds'))
    for env, it, ms in SEEDS:
        for m in ms:
            r = next(ref)
            for k in (1, 2, 3):
                st = D.ck[env][(f'{m}_run{k}', it)].stats()
                C.cmp('Table 15', f'{env} it{it} {m} run {k}', f'{pct(st["greedy"])} / {pct(st["pass1"])}', r[2 + k])


TUNING = {'baseline_tuning': ('alfworld', [('capo_clip0.001', 36), ('capo_clip0.1', 36), ('capo', 36), ('stppo_clip3e-4', 36),
                                            ('stppo', 36), ('grpo_g4', 36), ('grpo', 36), ('tree_grpo_8leaves', 36), ('tree_grpo', 36)]),
          'baseline_tuning_sw': ('scienceworld', [('capo_clip0.3', None), ('capo', 18), ('capo_clip0.03', None), ('sopo_clip0.2', 60),
                                                  ('sopo', 60), ('sopo_clip0.1', 60), ('stppo', 60), ('paw_coef1.0', 60), ('paw', 60),
                                                  ('paw_coef0.25', 60)])}


def check_tuning(D, C):
    for table, (env, spec) in TUNING.items():
        for (m, it), r in zip(spec, rows(table)):
            v = D.method(env, m, it) if it else None
            got = [pct(v['greedy']) if v else '--', pct(v['pass1']) if v else '--']
            for j, (g, w) in enumerate(zip(got, r[-2:])):
                C.cmp(f'Table {7 if env == "alfworld" else 8}', f'{m} col {j}', g, w)


SW_TYPES = {'use-thermometer': 'use-thermometer', 'melting-point (known)': 'measure-melting-point-known-substance',
            'power-component': 'power-component', 'test-conductivity': 'test-conductivity',
            'find-living-thing': 'find-living-thing', 'find-non-living-thing': 'find-non-living-thing', 'find-plant': 'find-plant',
            'find-animal': 'find-animal', 'chemistry-mix': 'chemistry-mix', 'paint-secondary': 'chemistry-mix-paint-secondary-color',
            'paint-tertiary': 'chemistry-mix-paint-tertiary-color', 'power-(non)renewable': 'power-component-renewable-vs-nonrenewable-energy',
            'conductivity (unknown)': 'test-conductivity-of-unknown-substances', 'lifespan-longest-lived': 'lifespan-longest-lived',
            'lifespan-shortest-lived': 'lifespan-shortest-lived', 'lifespan-longest-then-shortest': 'lifespan-longest-lived-then-shortest-lived',
            'identify-life-stages-1': 'identify-life-stages-1', 'identify-life-stages-2': 'identify-life-stages-2'}


def check_sw_tasks(D, C):
    ms = ['wmppo', 'wmppo_policy_only', 'ppo160', 'qppo', 'sopo']
    V = {m: D.method('scienceworld', m, 60) for m in ms}
    for r in rows('sw_tasks'):
        lab = r[0].replace(r'\quad', '').replace(r'\emph{', '').replace('}', '').strip()
        if lab.startswith('Procedural'):
            sel = lambda t: t not in E.FORCED_CHOICE  # noqa: E731
        elif lab.startswith('Forced-choice'):
            sel = lambda t: t in E.FORCED_CHOICE  # noqa: E731
        else:
            sel = (lambda name: (lambda t: t == name))(SW_TYPES[lab])
        for j, m in enumerate(ms):
            G = {k: v for k, v in V[m]['_G'].items() if sel(k.split('|')[0])}
            S = {k: v for k, v in V[m]['_S'].items() if sel(k.split('|')[0])}
            if j == 0:
                C.cmp('Table 16', f'{lab} games', str(len(G)), r[1])
            C.cmp('Table 16', f'{lab} {m}', f'{pct(E.mean(G.values()))}/{pct(E.mean(S.values()))}', r[2 + j])


def check_sokoban(D, C):
    for m, r in zip(['wmppo', 'mcppo', 'ppo160'], rows('sokoban')):
        got = []
        for it in (24, 36):
            v, base = D.method('sokoban', m, it), D.method('sokoban', 'wmppo', it)
            b = None if m == 'wmppo' else base
            got += [kstep(v), cell(v, 'greedy', b), cell(v, 'pass1', b)]
        for j, (g, w) in enumerate(zip(got, r[1:])):
            C.cmp('Table 18', f'{m} col {j + 2}', g, w)


def check_gate_ablation(D, C):
    for m, r in zip(['wmppo', 'gate_ablation_no_gate', 'gate_ablation_top_bin', 'ppo160'], rows('gate_ablation_sw')):
        got = []
        for it in (36, 48, 60):
            v, base = D.method('scienceworld', m, it, [1]), D.method('scienceworld', 'wmppo', it, [1])
            b = None if m == 'wmppo' else base
            got += [cell(v, 'greedy', b), cell(v, 'pass1', b)]
        for j, (g, w) in enumerate(zip(got, r[2:])):
            C.cmp('Table 21', f'{m} col {j + 3}', g, w)


COST = [('wmppo', 'WM-PPO'), ('ppo160', 'PPO-160'), ('ppo256', 'PPO-256'), ('wide_tree_ppo', 'Wide-Tree-PPO'), ('qppo', 'Q-PPO')]
LAST = {'alfworld': 36, 'scienceworld': 60}


def check_cost(D, C):
    """Table 17: means per iteration over iterations 13-36 (ALFWorld) / 13-60 (ScienceWorld). Real steps: all runs of the
    method (as in Table 2). Policy samples: K = 4 per chain step for WM-PPO and Q-PPO, from the runs with decoded counts,
    with their executed steps replaced by the real steps of the same runs; one per real step for the other methods.
    WM generations: the runs with decoded counts. Wall and GPU-h: measured, over the iterations of the three runs on
    A100-40GB nodes (wall_s x GPUs allocated in that iteration)."""
    import csv
    cost = list(csv.DictReader(open(os.path.join(E.RESULTS, 'cost_per_iteration.csv'))))
    ref = iter(rows('cost'))
    for env in ('alfworld', 'scienceworld'):
        for m, label in COST:
            r = next(ref)
            runs = [f'{m}_run{k}' for k in (1, 2, 3)]
            its = range(13, LAST[env] + 1)
            real = E.mean(D.steps_it[(env, n, it)] for n in runs for it in its if (env, n, it) in D.steps_it)
            dec = [x for x in cost if x['env'] == env and x['run'] in runs and 13 <= int(x['it']) <= LAST[env] and x['policy_samples']]
            dec_runs = sorted({x['run'] for x in dec})
            if m in ('wmppo', 'qppo'):
                real_dec = E.mean(D.steps_it[(env, n, it)] for n in dec_runs for it in its if (env, n, it) in D.steps_it)
                ps = E.mean(float(x['policy_samples']) for x in dec) + real_dec - E.mean(float(x['executed_steps']) for x in dec)
            else:
                ps = real
            wmg = '--' if m == 'qppo' else f"{E.mean(float(x['wm_generations']) for x in dec):,.0f}"
            hw = [x for x in cost if x['env'] == env and x['run'] in runs and 13 <= int(x['it']) <= LAST[env]
                  and x['gpu'].startswith('A100') and x['wall_s']]
            wall = E.mean(float(x['wall_s']) for x in hw) / 60
            gpuh = E.mean(float(x['wall_s']) * int(x['gpus']) for x in hw) / 3600
            got = [f'{real:,.0f}', f'{ps:,.0f}', wmg, f'{wall:.0f}', f'{gpuh:.1f}']
            for j, (g, w) in enumerate(zip(got, r[2:])):
                C.cmp('Table 17', f'{env} {label} col {j + 3}', g, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--show', action='store_true', help='print every compared cell')
    a = ap.parse_args()
    D, C = Data(), Checker(a.show)
    for name, fn in (('Table 2', check_main), ('Tables 10-13', check_full), ('Table 15', check_seeds),
                     ('Tables 7-8', check_tuning), ('Table 16', check_sw_tasks), ('Table 18', check_sokoban),
                     ('Table 21', check_gate_ablation), ('Table 17', check_cost)):
        n0, b0 = C.n, C.bad
        fn(D, C)
        print(f'{name}: {C.n - n0 - (C.bad - b0)}/{C.n - n0} cells match')
    print(f'total: {C.n - C.bad}/{C.n} cells match')
    sys.exit(1 if C.bad else 0)


if __name__ == '__main__':
    main()
