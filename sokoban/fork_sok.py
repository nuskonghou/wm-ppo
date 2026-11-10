"""Start a Sokoban run from the warm-up's iteration-12 policy and critic with the warm-up's task cursor (64 levels per
warm-up iteration), so every arm sees the same levels from iteration 13 on. The warm-up run directory is
paths.SOKOBAN_WARMUP (environment variable WMPPO_SOKOBAN_WARMUP).
   python fork_sok.py <out dir> <arm name>"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmppo import paths  # noqa: E402

SRC = paths.SOKOBAN_WARMUP
out, arm = sys.argv[1], sys.argv[2]
w = json.load(open(f'{SRC}/state.json'))         # the warm-up's state: iteration, task cursor, tokens, episodes
assert w['it'] == 12 and w['cursor'] == w['episodes'] == 64 * 12, w
os.makedirs(out, exist_ok=True)
st = {'it': 12, 'adapter': f'{SRC}/it012/adapter', 'lora': f'sok_{arm}_it012', 'critic': f'{SRC}/it012/critic',
      'cursor': w['cursor'], 'tokens': w['tokens'], 'episodes': w['episodes'], 'forked_from': 'warm-up it012'}
json.dump(st, open(os.path.join(out, 'state.json'), 'w'))
print(json.dumps(st))
