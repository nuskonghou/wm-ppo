"""Accuracy of a world model on held-out transitions of the evaluation games (Section 2.2 and Appendix C).

Transitions come from wm_collect.py on valid_seen + valid_unseen. Each one falls in one class:
  nothing  the true next observation is "Nothing happens."
  reveal   the true next observation shows contents ('you see ...') that no earlier observation of the episode showed
  known    everything else: the next observation follows from what the episode has already shown
Metrics per class: exact match of the greedy prediction. For reveal: object F1 per transition on instances ('mug 3') and on
types ('mug'), and whether the task's target object type is among the revealed contents (true vs predicted; with
--samples M the predicted probability is the fraction of M temperature-1 samples that contain it, scored by AUC).
  python wm_eval.py --url http://HOST:PORT --lora <name|none> --name <tag> [--n 1500 --samples 4]"""
import argparse
import collections
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rollout  # noqa: E402
from ct_infer import mask_new_contents  # noqa: E402
from wm_prompt import SYSTEM, wm_user  # noqa: E402
from wmppo import paths  # noqa: E402

SEE_PART = re.compile(r"you see ([^.]*)\.", re.I)
OBJ = re.compile(r"\ban? ([a-z]+) (\d+)")


def objects(obs):
    return {f"{m.group(1)} {m.group(2)}" for part in SEE_PART.findall(obs) for m in OBJ.finditer(part)}


def f1(p, t):
    if not p and not t:
        return 1.0
    tp = len(p & t)
    return 0.0 if tp == 0 else 2 * tp / (len(p) + len(t))


def klass(r):
    if r['next_obs'].startswith('Nothing happens'):
        return 'nothing'
    prior = [r['room']] + [o for o, _ in r['hist']] + [r['obs']]
    return 'reveal' if mask_new_contents(r['next_obs'], prior) != re.sub(r"\s+", " ", r['next_obs']).strip() else 'known'


def target_type(task, types):
    for w in re.findall(r"[a-z]+", task.lower()):
        if w in types:
            return w
    return None


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]; neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--lora', required=True, help="adapter name loaded on the server, or 'none' for the base model")
    ap.add_argument('--name', required=True)
    ap.add_argument('--data', nargs='+', default=[f'{HERE}/runs/wm/valid_seen.jsonl', f'{HERE}/runs/wm/valid_unseen.jsonl'])
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--samples', type=int, default=4, help='temperature-1 samples per reveal transition (target belief)')
    ap.add_argument('--threads', type=int, default=64)
    ap.add_argument('--model', default=paths.MODEL)
    ap.add_argument('--out', default=f'{HERE}/runs/wm/eval')
    a = ap.parse_args()
    tok = rollout._tok(a.model)
    rows = [json.loads(l) for f in a.data for l in open(f)]
    random.Random(0).shuffle(rows)
    rows = rows[:a.n]
    types = {o.split()[0] for r in rows for o in objects(r['next_obs']) | objects(r['obs'])}
    lora = None if a.lora == 'none' else a.lora

    def gen(r, temperature):
        ids = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": wm_user(r)}],
                                      tokenize=True, add_generation_prompt=True, return_dict=False)
        ids = ids if isinstance(ids, list) else ids['input_ids']
        text = rollout.generate({'url': a.url, 'lora': lora, 'temperature': temperature, 'max_new_tokens': 600}, ids)[0]
        mo = re.search(r"<obs>(.*?)</obs>", text, re.S)
        return (mo.group(1) if mo else text.replace('<|im_end|>', '')).strip()

    def one(r):
        k = klass(r)
        o_hat = gen(r, 0.0)
        rec = {'class': k, 'action': r['action'], 'true': r['next_obs'], 'pred': o_hat, 'exact': o_hat == r['next_obs'].strip()}
        if k == 'reveal':
            to, po = objects(r['next_obs']), objects(o_hat)
            rec.update(inst_f1=f1(po, to), type_f1=f1({x.split()[0] for x in po}, {x.split()[0] for x in to}))
            tt = target_type(r['task'], types)
            if tt:
                samp = [gen(r, 1.0) for _ in range(a.samples)] if a.samples else []
                rec.update(target=tt, target_true=any(x.split()[0] == tt for x in to),
                           target_greedy=any(x.split()[0] == tt for x in po),
                           target_p=sum(any(x.split()[0] == tt for x in objects(s)) for s in samp) / max(1, len(samp)))
        return rec

    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(one, rows))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f'{a.name}.jsonl'), 'w') as f:
        for x in res:
            f.write(json.dumps(x) + '\n')
    by = collections.defaultdict(list)
    for x in res:
        by[x['class']].append(x)
    summ = {'name': a.name, 'n': len(res), 'exact_all': sum(x['exact'] for x in res) / len(res)}
    for k, xs in by.items():
        summ[f'{k}_n'] = len(xs); summ[f'{k}_exact'] = sum(x['exact'] for x in xs) / len(xs)
    rv = by['reveal']
    if rv:
        summ['reveal_inst_f1'] = sum(x['inst_f1'] for x in rv) / len(rv)
        summ['reveal_type_f1'] = sum(x['type_f1'] for x in rv) / len(rv)
        tg = [x for x in rv if 'target' in x]
        if tg:
            summ['target_n'] = len(tg); summ['target_base_rate'] = sum(x['target_true'] for x in tg) / len(tg)
            summ['target_greedy_acc'] = sum(x['target_true'] == x['target_greedy'] for x in tg) / len(tg)
            summ['target_auc'] = auc([x['target_p'] for x in tg], [x['target_true'] for x in tg])
    json.dump(summ, open(os.path.join(a.out, f'{a.name}.summary.json'), 'w'), indent=1)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)


if __name__ == '__main__':
    main()
