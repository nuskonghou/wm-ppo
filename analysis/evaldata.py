"""Official-evaluation data and the statistics of the paper (standard library only).

Evaluation protocol: per checkpoint, one greedy episode per evaluation game and three temperature-1 runs.
  greedy   mean success of the greedy episodes over all evaluation games
  pass@1   mean over games of the mean success of the three temperature-1 runs
  pass@3   mean over games of: at least one of the three runs succeeded
Method values are means over its runs (paper runs 1-3); the spread is the sample standard deviation over runs.
Significance: per-game paired z-test (two-sided, p < 0.05) on the per-game success averaged over each method's runs.

Data: results/eval/<env>.csv.gz (columns run, iteration, protocol, split, game, success, steps), or the per-checkpoint
files written by scripts/launch.py eval (final_eval/greedy_<name>.json, infer_<name>_random_f<r>.jsonl)."""
import collections
import csv
import glob
import gzip
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FORCED_CHOICE = {'power-component-renewable-vs-nonrenewable-energy', 'test-conductivity-of-unknown-substances',
                 'lifespan-longest-lived', 'lifespan-shortest-lived', 'lifespan-longest-lived-then-shortest-lived',
                 'identify-life-stages-1', 'identify-life-stages-2'}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


def stdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return None
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


class Checkpoint:
    """Per-game outcomes of one evaluated checkpoint."""

    def __init__(self, greedy, samples, split):
        self.greedy = greedy            # game -> 0/1
        self.samples = samples          # game -> [0/1, ...] (three temperature-1 runs)
        self.split = split              # game -> split name

    def per_game_pass1(self):
        return {k: mean(v) for k, v in self.samples.items()}

    def stats(self):
        keys = list(self.greedy)
        s = self.per_game_pass1()
        return {'greedy': mean(self.greedy[k] for k in keys),
                'pass1': mean(s[k] for k in keys if k in s),
                'pass3': mean(float(max(self.samples[k]) > 0) for k in keys if k in self.samples),
                'n_games': len(keys)}


def load_release(env):
    """{(run, iteration): Checkpoint} from results/eval/<env>.csv.gz."""
    g, s, sp = collections.defaultdict(dict), collections.defaultdict(lambda: collections.defaultdict(list)), collections.defaultdict(dict)
    with gzip.open(os.path.join(RESULTS, 'eval', f'{env}.csv.gz'), 'rt') as f:
        for r in csv.DictReader(f):
            key = (r['run'], int(r['iteration']))
            if r['protocol'] == 'greedy':
                g[key][r['game']] = float(r['success']); sp[key][r['game']] = r['split']
            else:
                s[key][r['game']].append(float(r['success']))
    return {k: Checkpoint(g[k], dict(s[k]), sp[k]) for k in g}


def load_files(env, final_eval_dir, name):
    """One checkpoint from the files of scripts/launch.py eval (name = the name given to the evaluation)."""
    gr = json.load(open(os.path.join(final_eval_dir, f'greedy_{name}.json')))
    greedy, split = {}, {}
    for sp in [k[:-9] for k in gr if k.endswith('_per_game')]:
        for k, v in gr[sp + '_per_game'].items():
            k = f'{sp}/{k}' if env == 'alfworld' else k
            greedy[k] = float(v > 0); split[k] = sp
    samples = collections.defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(final_eval_dir, f'infer_{name}_random_f[0-9].jsonl'))):
        for line in open(fn):
            x = json.loads(line)
            if 'error' in x:
                continue
            k = x['game']
            if env == 'alfworld':                     # absolute game path -> <split>/<task>/<trial>
                p = k.rstrip('/').split('/')
                k = '/'.join(p[p.index('json_2.1.1') + 1:p.index('json_2.1.1') + 4]) if 'json_2.1.1' in p else '/'.join(p[-4:-1])
            samples[k].append(float(x['reward'] > 0))
    return Checkpoint(greedy, dict(samples), split)


def method_values(ckpts):
    """Runs of one method at one iteration -> mean / per-run values and the run-averaged per-game outcomes."""
    st = [c.stats() for c in ckpts]
    G, S = collections.defaultdict(list), collections.defaultdict(list)
    for c in ckpts:
        for k, v in c.greedy.items():
            G[k].append(v)
        for k, v in c.per_game_pass1().items():
            S[k].append(v)
    return {'greedy': mean(x['greedy'] for x in st), 'pass1': mean(x['pass1'] for x in st),
            'pass3': mean(x['pass3'] for x in st), 'greedy_runs': [x['greedy'] for x in st],
            'pass1_runs': [x['pass1'] for x in st], 'n_runs': len(st),
            '_G': {k: mean(v) for k, v in G.items()}, '_S': {k: mean(v) for k, v in S.items()}}


def paired(A, B):
    """A, B: per-game success. Mean difference A - B and the two-sided p-value of the per-game paired z-test."""
    c = sorted(set(A) & set(B))
    d = [A[k] - B[k] for k in c]
    if len(d) < 2:
        return None
    mu = mean(d)
    se = (sum((v - mu) ** 2 for v in d) / (len(d) - 1) / len(d)) ** 0.5
    return {'diff': mu, 'p': math.erfc(abs(mu / se) / 2 ** 0.5) if se > 0 else 1.0, 'n': len(d)}


def exec_steps():
    """{(env, run, iteration): executed policy steps of that iteration} from results/real_steps.csv."""
    out = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, 'real_steps.csv'))):
        if r['exec_steps'] != '':
            out[(r['env'], r['run'], int(r['it']))] = float(r['exec_steps'])
    return out


def real_steps():
    """{(env, run, iteration): cumulative executed policy steps from iteration 13} from results/real_steps.csv."""
    out = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, 'real_steps.csv'))):
        if r['cum_exec_steps_from_it13'] != '':
            out[(r['env'], r['run'], int(r['it']))] = float(r['cum_exec_steps_from_it13'])
    return out
