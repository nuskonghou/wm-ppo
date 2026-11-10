"""World-model training: a LoRA adapter on the same base model as the policy, trained with the next-observation
(cross-entropy) loss: given the task, the history, the current observation, the admissible actions and the action,
predict the next observation. Used for the initial world model (transitions executed during the warm-up), for the
online updates of WM-PPO (each iteration's executed transitions, continuing from the current adapter) and for the
world model of the diagnostics. Records are in the format written by wm_from_log.py and rollout.py.

torchrun --nproc_per_node G train_wm.py --data a.jsonl [b.jsonl ...] --out <adapter dir> [--epochs 1 --lr 1e-4]"""
import argparse
import json
import os
import random
import time

import torch
import torch.distributed as dist

from wm_prompt import SYSTEM, wm_target, wm_user  # noqa: E402,F401


def encode(tok, r, max_len):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": wm_user(r)}]
    p = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
    p = p if isinstance(p, list) else p['input_ids']
    t = tok(wm_target(r) + tok.eos_token, add_special_tokens=False)['input_ids']
    p = p[-(max_len - len(t)):]
    return p, t


def packs(enc, pack_tokens):
    """Group (prompt, target) pairs into packs of at most pack_tokens tokens (one sequence per pack; flash-attention
    keeps the records apart through position_ids that restart at 0)."""
    out, cur, n = [], [], 0
    for e in enc:
        m = len(e[0]) + len(e[1])
        if cur and n + m > pack_tokens:
            out.append(cur); cur, n = [], 0
        cur.append(e); n += m
    return out + ([cur] if cur else [])


def pack_nll(model, pack):
    """Per-token NLL of the targets of one pack. Pack items are (prompt, target) or (prompt, target, extra)."""
    ids, pos, idx, tgt = [], [], [], []
    for e in pack:
        p, t = e[0], e[1]
        off = len(ids); ids += p + t; pos += list(range(len(p) + len(t)))
        idx += list(range(off + len(p) - 1, off + len(p) + len(t) - 1)); tgt += t
    inner = model.base_model.model
    h = inner.model(input_ids=torch.tensor([ids], device='cuda'), position_ids=torch.tensor([pos], device='cuda')).last_hidden_state[0]
    nll = None
    if tgt:
        logits = inner.lm_head(h[torch.tensor(idx, device='cuda')]).float()
        nll = -torch.log_softmax(logits, -1).gather(-1, torch.tensor(tgt, device='cuda')[:, None])[:, 0]
    return nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter-in', default='none', help="initial LoRA (e.g. the current world model), else a fresh one")
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--mb', type=int, default=32, help='records per optimizer step (global)')
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--max-len', type=int, default=6144)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pack-tokens', type=int, default=16384)
    a = ap.parse_args()
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    if a.adapter_in != 'none':
        model = PeftModel.from_pretrained(model, a.adapter_in, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
                                                 target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    rows = [json.loads(l) for f in a.data for l in open(f)]
    random.Random(a.seed).shuffle(rows)
    per = max(1, a.mb // world) * world
    n_steps = int(len(rows) * a.epochs) // per
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.1, 1 - s / max(n_steps, 1)))
    model.train(); t0 = time.time(); tot = ntok = 0.0
    for s in range(n_steps):
        chunk = [rows[(s * per + i) % len(rows)] for i in range(per)][rank::world]
        enc = [encode(tok, r, a.max_len) for r in chunk]
        n_t = max(1, sum(len(e[1]) for e in enc))
        opt.zero_grad(set_to_none=True)
        for pk in packs(enc, a.pack_tokens):
            nll = pack_nll(model, pk)
            loss = 0.0
            if nll is not None:
                loss = loss + nll.sum() / n_t
                tot += nll.sum().item(); ntok += len(nll)
            loss.backward()
        for p in params:
            if p.grad is None:                       # e.g. no target token on this rank: keep the collective in step
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if rank == 0 and (s % 50 == 0 or s == n_steps - 1):
            print(f'step {s + 1}/{n_steps} token nll {tot / max(ntok, 1):.4f} '
                  f'{time.time() - t0:.0f}s', flush=True)
            tot = ntok = 0.0
    if rank == 0:
        model.save_pretrained(a.out)
        json.dump({'records': len(rows), 'steps': n_steps, 'adapter_in': a.adapter_in,
                   'wall_s': round(time.time() - t0)},
                  open(os.path.join(a.out, 'train_wm_stats.json'), 'w'))
        print('saved', a.out, flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
