"""One PPO-clip epoch of policy LoRA training on turn samples with given advantages (the GRPO-family baselines and the
offline update of the diagnostics in Section 2).

torchrun --nproc_per_node G train_lora.py --data samples.jsonl --base <model dir> --adapter-in <dir|none>
         --adapter-out <dir>
Sample = {"prompt_ids", "resp_ids", "old_lp", "adv"}. Loss of a minibatch = token-level mean over all its
response tokens of -min(r A, clip(r, 1-eps_low, 1+eps_high) A).
--pack-tokens N > 0 packs several samples into one sequence (flash-attention varlen through position ids that restart
at 0; the same loss, several times faster on short samples). 0 runs one sample per forward pass.
"""
import argparse, json, os, random, time
import torch
import torch.distributed as dist

sys_path_here = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402
sys.path.insert(0, sys_path_here)
from train_wm import packs, pack_nll  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter-in", default="none")
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--mb", type=int, default=64, help="samples per optimizer step (global)")
    ap.add_argument("--eps-low", type=float, default=0.2)
    ap.add_argument("--eps-high", type=float, default=0.28)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default="")
    ap.add_argument("--pack-tokens", type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.adapter_in != "none" and os.path.isdir(args.adapter_in):
        model = PeftModel.from_pretrained(model, args.adapter_in, is_trainable=True)
    else:
        cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)

    rows = [json.loads(l) for l in open(args.data)]
    rows = [r for r in rows if r["resp_ids"] and abs(r["adv"]) > 0]
    random.Random(args.seed).shuffle(rows)
    per_step = max(1, args.mb // world) * world
    n_steps = max(1, len(rows) // per_step)
    rows = rows[: n_steps * per_step] if len(rows) >= per_step else rows
    stats = {"n": 0, "loss": 0.0, "ratio": 0.0, "clipfrac": 0.0, "kl": 0.0, "tokens": 0, "skipped_long": 0}
    t0 = time.time()
    model.train()
    for s in range(n_steps):
        chunk = rows[s * per_step:(s + 1) * per_step][rank::world]
        ntok_local = sum(len(r["resp_ids"]) for r in chunk) or 1
        opt.zero_grad(set_to_none=True)
        items = []
        for r in chunk:
            p_ids, resp = r["prompt_ids"], r["resp_ids"]
            if len(p_ids) + len(resp) > args.max_len:
                p_ids = p_ids[-(args.max_len - len(resp)):]
                stats["skipped_long"] += 1
            items.append((p_ids, resp, r))
        groups = packs(items, args.pack_tokens) if args.pack_tokens > 0 else [[it] for it in items]
        for g in groups:
            if args.pack_tokens > 0:
                logp_all = -pack_nll(model, [(p_ids, resp) for p_ids, resp, _ in g])
            else:
                p_ids, resp, _ = g[0]
                out = model(input_ids=torch.tensor([p_ids + resp], device="cuda"), logits_to_keep=len(resp) + 1)
                logp_all = torch.log_softmax(out.logits[0, :-1].float(), -1).gather(
                    -1, torch.tensor(resp, device="cuda")[:, None])[:, 0]
            old = torch.tensor([x for _, resp, r in g for x in r["old_lp"][: len(resp)]], device="cuda", dtype=torch.float32)
            A = torch.tensor([float(r["adv"]) for _, resp, r in g for _ in resp], device="cuda", dtype=torch.float32)
            ratio = torch.exp(logp_all - old)
            l1 = ratio * A
            l2 = torch.clamp(ratio, 1 - args.eps_low, 1 + args.eps_high) * A
            loss = -torch.min(l1, l2).sum() / ntok_local
            loss.backward()
            with torch.no_grad():
                off = 0
                for _, resp, _ in g:                        # per-sample statistics, as in the unpacked path
                    rr, oo, ll = ratio[off:off + len(resp)], old[off:off + len(resp)], logp_all[off:off + len(resp)]
                    stats["ratio"] += rr.mean().item()
                    stats["clipfrac"] += ((rr < 1 - args.eps_low) | (rr > 1 + args.eps_high)).float().mean().item()
                    stats["kl"] += (oo - ll).mean().item()
                    stats["tokens"] += len(resp); stats["n"] += 1
                    off += len(resp)
                stats["loss"] += loss.item()
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if rank == 0 and (s % 10 == 0 or s == n_steps - 1):
            n = max(stats["n"], 1)
            print(f"step {s + 1}/{n_steps} ratio {stats['ratio'] / n:.4f} clipfrac {stats['clipfrac'] / n:.4f} "
                  f"kl {stats['kl'] / n:+.5f} {time.time() - t0:.0f}s", flush=True)
    agg = torch.tensor([stats[k] for k in ("n", "loss", "ratio", "clipfrac", "kl", "tokens", "skipped_long")],
                       device="cuda", dtype=torch.float64)
    dist.all_reduce(agg)
    if rank == 0:
        model.save_pretrained(args.adapter_out)
        n = max(agg[0].item(), 1)
        st = {"samples": int(agg[0].item()), "ratio": agg[2].item() / n, "clipfrac": agg[3].item() / n,
              "kl": agg[4].item() / n, "tokens": int(agg[5].item()), "truncated": int(agg[6].item()),
              "steps": n_steps, "wall_s": round(time.time() - t0, 1)}
        print("TRAIN_STATS " + json.dumps(st), flush=True)
        if args.stats_out:
            json.dump(st, open(args.stats_out, "w"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
