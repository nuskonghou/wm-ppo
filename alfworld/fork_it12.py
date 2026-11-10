"""Start a new arm from the PPO arm's iteration-12 policy and critic, with the PPO arm's task cursor, so it sees the
same tasks as PPO iterations 13-24.   python fork_it12.py <out dir> <arm name>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
out, arm = sys.argv[1], sys.argv[2]
m = [json.loads(l) for l in open(f'{HERE}/runs/ppo/metrics.jsonl')]
r12 = [x for x in m if x['it'] == 12][-1]
os.makedirs(out, exist_ok=True)
st = {'it': 12, 'adapter': f'{HERE}/runs/ppo/it012/adapter', 'lora': f'alf_{arm}_it012', 'critic': f'{HERE}/runs/ppo/it012/critic',
      'cursor': 128 + 11 * 64, 'tokens': r12['tokens'], 'episodes': r12['episodes'], 'forked_from': 'runs/ppo it012'}
assert st['episodes'] == st['cursor'], st
json.dump(st, open(os.path.join(out, 'state.json'), 'w'))
print(json.dumps(st))
