"""Paired per-game comparison of inference rules (outputs of ct_infer.py): mean difference in success, its standard
error, and an exact two-sided sign test on the games where the two rules disagree.
   python ct_paired.py A B [A B ...]   (A, B: <arm>_<rule><tag> of infer_*.jsonl in $CT_DIR, default runs/critic_test)"""
import json
import math
import os
import sys

D = os.environ.get('CT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'critic_test'))


def load(name):
    return {r['game']: r['reward'] for r in map(json.loads, open(f'{D}/infer_{name}.jsonl')) if 'error' not in r}


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    A, B = load(a), load(b)
    g = sorted(set(A) & set(B))
    d = [A[x] - B[x] for x in g]
    m = sum(d) / len(d); se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    w = sum(x > 0 for x in d); l = sum(x < 0 for x in d)
    print(f'{a:34s} - {b:28s} SR {sum(A[x] for x in g) / len(g):.3f} vs {sum(B[x] for x in g) / len(g):.3f}  '
          f'diff {100 * m:+.1f} pts (SE {100 * se:.1f})  wins {w} losses {l}  sign p={sign_p(w, l):.3f}')
