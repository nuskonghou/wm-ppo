#!/usr/bin/env python3
"""Score an evaluation written by `scripts/wmppo eval ENV ADAPTER NAME` and compare it with the official evaluation of
the same checkpoint in results/eval (greedy, pass@1, pass@3, and the share of games whose greedy outcome agrees).
   python3 analysis/score_eval.py ENV NAME [--dir DIR] [--official RUN:ITERATION]
NAME of the form <config>_run<k>_it<NNN> (e.g. wmppo_run1_it036) is compared with run <config>_run<k>, iteration NNN by
default. DIR defaults to <ENV>/runs/final_eval."""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaldata as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('env', choices=['alfworld', 'scienceworld', 'sokoban'])
    ap.add_argument('name')
    ap.add_argument('--dir', default=None)
    ap.add_argument('--official', default=None, help='RUN:ITERATION in results/eval, e.g. wmppo_run1:36')
    a = ap.parse_args()
    d = a.dir or os.path.join(E.ROOT, a.env, 'runs', 'final_eval')
    new = E.load_files(a.env, d, a.name)
    st = new.stats()
    n_samples = min((len(v) for v in new.samples.values()), default=0)
    print(f'{a.name}: {st["n_games"]} games, greedy {100 * st["greedy"]:.1f}, pass@1 {100 * st["pass1"]:.1f}, '
          f'pass@3 {100 * st["pass3"]:.1f} ({n_samples} temperature-1 runs per game)')
    off = a.official
    if off is None:
        m = re.fullmatch(r'(.+_run\d+)_it(\d+)', a.name)
        off = f'{m.group(1)}:{int(m.group(2))}' if m else None
    if off is None:
        return
    run, it = off.rsplit(':', 1)
    ref = E.load_release(a.env).get((run, int(it)))
    if ref is None:
        sys.exit(f'no official evaluation of {run} at iteration {it} in results/eval/{a.env}.csv.gz')
    rs = ref.stats()
    common = sorted(set(ref.greedy) & set(new.greedy))
    agree = E.mean(float(ref.greedy[k] == new.greedy[k]) for k in common)
    print(f'official {run} iteration {it}: greedy {100 * rs["greedy"]:.1f}, pass@1 {100 * rs["pass1"]:.1f}, '
          f'pass@3 {100 * rs["pass3"]:.1f}')
    print(f'greedy outcome equal to the official one on {100 * agree:.1f}% of {len(common)} games')


if __name__ == '__main__':
    main()
