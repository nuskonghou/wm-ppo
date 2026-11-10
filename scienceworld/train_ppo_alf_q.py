"""Q-PPO (structural control): the WM-PPO pipeline of train_ppo_alf.py with a learned Q(h, y) as the scorer of the
candidate actions instead of the critic's value of the predicted successor. Q = a third LoRA adapter ("qnet") + a linear
head reading the last token of the policy prompt h followed by the response y; initialised from the warm-up critic and
trained every iteration on all executed edges (squared error to the edge return G = mean final reward below the edge)
plus the sibling-centred term on branch groups (weight --sib-weight), lr = critic lr. The "child" entries of the
decision points and calibration pairs hold h + y token ids (alf_final.py, arm qppo); gate, bins, auxiliary rows,
critic and GAE are unchanged."""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_wm import packs, pack_nll  # noqa: E402


def load(path):
    d = json.load(open(path))
    for i, e in enumerate(d["edges"]):
        d["nodes"][e["node"]].setdefault("out", []).append(i)
    for n in d["nodes"]:
        n.setdefault("out", [])
    return d["nodes"], d["edges"], d["groups"], d.get("distill", []), d.get("calib", [])


def calib_weights(pairs, min_n=20, prior_n=40.0, g_ref=0.05):
    """Reliability gate. Per margin bin b: the gain of following the preference, g_b = mean of (G_c - G_b) *
    sign(preference) over the calibration pairs (an unbiased estimate of E[A(a_hat) - A(a')] in that bin), shrunk
    towards 0 as g_b * n_b / (n_b + prior_n); weight w_b = clip(shrunk gain / g_ref, 0, 1), and 0 for bins with fewer
    than min_n pairs. Bins whose comparisons do not pay off in real outcomes get weight ~0."""
    import statistics as sts
    w, g, se, n, lcb = [], [], [], [], []
    for b in range(3):
        xs = [p["gain"] for p in pairs if p["bin"] == b]
        n.append(len(xs)); g.append(sts.mean(xs) if xs else 0.0)
        se.append(sts.pstdev(xs) / len(xs) ** 0.5 if len(xs) > 1 and sts.pstdev(xs) > 0 else float("inf"))
        lcb.append(g[b] - 1.64 * se[b] if se[b] < float("inf") else 0.0)
        post = g[b] * n[b] / (n[b] + prior_n)
        w.append(min(1.0, max(0.0, post / g_ref)) if n[b] >= min_n else 0.0)
    return w, g, se, n, lcb


def calib_weights_lcb(pairs, min_n=20):
    """The gate rule of iterations 13-14 (--calib-rule lcb): w_b = (g_b - 1.64 se_b)_+ / max_b (.)_+, bins with fewer
    than min_n pairs count as 0, and all weights are 0 if no bound is positive."""
    import statistics as sts
    g, se, n = [], [], []
    for b in range(3):
        xs = [p["gain"] for p in pairs if p["bin"] == b]
        n.append(len(xs))
        g.append(sts.mean(xs) if xs else 0.0)
        se.append(sts.pstdev(xs) / max(1, len(xs)) ** 0.5 if len(xs) > 1 else 1.0)
    lcb = [(g[b] - 1.64 * se[b]) if n[b] >= min_n else 0.0 for b in range(3)]
    top = max(max(lcb), 0.0)
    w = [max(lcb[b], 0.0) / top if top > 0 else 0.0 for b in range(3)]
    return w, g, se, n, lcb


def gae(nodes, edges, V, lam):
    """Advantage per edge, children before parents (child node index > parent within a tree is not guaranteed,
    so recurse with memo)."""
    A = [None] * len(edges)

    def node_adv(ni):
        outs = [ei for ei in nodes[ni].get("out", []) if edge_adv(ei) is not None]
        return sum(A[ei] for ei in outs) / len(outs) if outs else None

    def edge_adv(ei):
        if A[ei] is not None:
            return A[ei]
        e = edges[ei]
        if e["term"]:
            A[ei] = e["R"] - V[e["node"]]
        else:
            nxt = node_adv(e["child"])
            delta = V[e["child"]] - V[e["node"]]
            A[ei] = delta + (lam * nxt if nxt is not None else 0.0)
        return A[ei]
    sys.setrecursionlimit(10000)
    for ei in range(len(edges)):
        edge_adv(ei)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--policy-in", default="none")
    ap.add_argument("--policy-out", required=True)
    ap.add_argument("--critic-in", default="none")
    ap.add_argument("--critic-out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--critic-lr", type=float, default=5e-5)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--sib-weight", type=float, default=0.0)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--eps-low", type=float, default=0.2)
    ap.add_argument("--eps-high", type=float, default=0.28)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default="")
    ap.add_argument("--distill-nu", type=float, default=0.0)
    ap.add_argument("--distill-label", choices=["la", "sc"], default="la",
                    help="la: preference from the predicted successors (gated); sc: majority sampled action with weight --distill-const-w")
    ap.add_argument("--distill-const-w", type=float, default=0.0)
    ap.add_argument("--calib-history", default="", help="comma-separated calib_scored.json of previous iterations")
    ap.add_argument("--calib-out", default="")
    ap.add_argument("--calib-min", type=int, default=30, help="minimum calibration pairs per margin bin")
    ap.add_argument("--calib-prior-n", type=float, default=40.0, help="shrinkage strength n_0 (pseudo-pairs) for the bin gain")
    ap.add_argument("--calib-g-ref", type=float, default=0.05, help="calibrated gain per decision that earns weight 1")
    ap.add_argument("--calib-rule", choices=["posterior", "lcb"], default="posterior")
    ap.add_argument("--pack-tokens", type=int, default=0, help="> 0: pack policy rows (flash-attn varlen)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    if args.policy_in != "none" and os.path.isdir(args.policy_in):
        model = PeftModel.from_pretrained(model, args.policy_in, is_trainable=True)
    else:
        model = get_peft_model(model, cfg)
    cdir = os.path.join(args.critic_in, "critic")
    if args.critic_in != "none" and os.path.isdir(cdir):
        model.load_adapter(cdir, adapter_name="critic", is_trainable=True)
    else:
        model.add_adapter("critic", cfg)
    hidden = model.config.hidden_size
    vhead = torch.nn.Linear(hidden, 1).cuda().float()
    torch.nn.init.zeros_(vhead.weight); torch.nn.init.zeros_(vhead.bias)
    hp = os.path.join(args.critic_in, "value_head.pt")
    if args.critic_in != "none" and os.path.exists(hp):
        vhead.load_state_dict(torch.load(hp, map_location="cuda"))
    qdir = os.path.join(args.critic_in, "qnet")                   # Q-PPO: Q network (initialised from the critic)
    if args.critic_in != "none" and os.path.isdir(qdir):
        model.load_adapter(qdir, adapter_name="qnet", is_trainable=True)
    elif args.critic_in != "none" and os.path.isdir(cdir):
        model.load_adapter(cdir, adapter_name="qnet", is_trainable=True)
    else:
        model.add_adapter("qnet", cfg)
    qhead = torch.nn.Linear(hidden, 1).cuda().float()
    torch.nn.init.zeros_(qhead.weight); torch.nn.init.zeros_(qhead.bias)
    qhp = os.path.join(args.critic_in, "q_head.pt")
    if args.critic_in != "none" and os.path.exists(qhp):
        qhead.load_state_dict(torch.load(qhp, map_location="cuda"))
    elif args.critic_in != "none" and os.path.exists(hp):
        qhead.load_state_dict(torch.load(hp, map_location="cuda"))
    pol_params = [p for n, p in model.named_parameters() if "lora_" in n and ".default." in n]
    cri_params = [p for n, p in model.named_parameters() if "lora_" in n and ".critic." in n] + list(vhead.parameters())
    q_params = [p for n, p in model.named_parameters() if "lora_" in n and ".qnet." in n] + list(qhead.parameters())
    for p in pol_params + cri_params + q_params:
        p.data = p.data.float()
    opt_p = torch.optim.AdamW(pol_params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    opt_c = torch.optim.AdamW(cri_params, lr=args.critic_lr, betas=(0.9, 0.99), weight_decay=0.0)
    opt_q = torch.optim.AdamW(q_params, lr=args.critic_lr, betas=(0.9, 0.99), weight_decay=0.0)
    inner = model.base_model.model.model  # decoder stack with LoRA layers injected

    def prompt_ids(nd):
        return nd["prompt_ids"]

    def value(nd, grad):
        ids = prompt_ids(nd)[-args.max_len:]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            h = inner(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]
            return vhead(h.float())[0]

    def qvalue(ids, grad):                     # Q-PPO: requires the "qnet" adapter to be active
        ids = ids[-args.max_len:]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            h = inner(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]
            return qhead(h.float())[0]

    nodes, edges, groups, distill, calib = load(args.trees)
    t0 = time.time()
    stats = {"nodes": len(nodes), "edges": len(edges), "sib_groups": len(groups)}

    # ---- 1. values (critic adapter, no grad), sharded then gathered
    model.set_adapter("critic")
    model.eval()
    V = torch.zeros(len(nodes), device="cuda")
    for i in range(rank, len(nodes), world):
        V[i] = value(nodes[i], False).detach()
    dist.all_reduce(V)
    V = V.tolist()
    # prequential critic diagnostics
    tg = [(V[i], nd["G"]) for i, nd in enumerate(nodes) if nd["G"] is not None]
    if len(tg) > 2:
        import statistics as sts
        ys = [g for _, g in tg]; res = [g - v for v, g in tg]
        vy = sts.pvariance(ys)
        stats["critic_explained_var_before"] = round(1 - sts.pvariance(res) / vy, 4) if vy > 0 else None
    hit = tot = 0
    for g in groups:
        nt = [ei for ei in g if not edges[ei]["term"]]
        for i, a in enumerate(nt):
            for b in nt[i + 1:]:
                da = edges[a]["G"] - edges[b]["G"]
                dv = V[edges[a]["child"]] - V[edges[b]["child"]]
                if da != 0 and dv != 0:
                    tot += 1; hit += (da > 0) == (dv > 0)
    stats["critic_sib_acc_before"] = [round(hit / tot, 3) if tot else None, tot]

    # ---- 1b. WM-PPO: critic values of the predicted successors, gate, auxiliary rows
    drows = []
    if distill and args.distill_label == "sc":     # majority-vote labels: constant weight, no world model or critic
        for d in distill:
            acts = [x["action"] for x in d["samples"]]
            cnt = {}
            for a in acts:
                cnt[a] = cnt.get(a, 0) + 1
            a_hat = max(dict.fromkeys(acts), key=lambda a: cnt[a])      # ties: first sampled
            share = cnt[a_hat] / len(acts)
            for x in d["samples"]:
                a_k = args.distill_nu * args.distill_const_w * (float(x["action"] == a_hat) - share)
                if x["resp_ids"] and a_k != 0:
                    drows.append({"prompt_ids": d["prompt_ids"], "resp": x["resp_ids"], "old": x["lps"], "adv": a_k})
        stats.update({"distill_states": len(distill), "distill_rows": len(drows), "distill_label": "sc",
                      "distill_const_w": args.distill_const_w})
    elif distill or calib:
        import statistics as sts
        kid_ids = []
        for d in distill:
            d["_k"] = {}
            for a, ids in d["child"].items():
                d["_k"][a] = len(kid_ids); kid_ids.append(ids)
        for c in calib:
            c["_c"] = len(kid_ids); kid_ids.append(c["child_c"]); c["_b"] = len(kid_ids); kid_ids.append(c["child_b"])
        KV = torch.zeros(len(kid_ids), device="cuda")
        model.set_adapter("qnet"); model.eval()                   # Q-PPO: candidates scored by Q(h, y)
        for i in range(rank, len(kid_ids), world):
            KV[i] = qvalue(kid_ids[i], False).detach()
        dist.all_reduce(KV)
        model.set_adapter("critic")
        KV = KV.tolist()
        margins = []
        for d in distill:
            q = {a: KV[k] for a, k in d["_k"].items()}
            order_q = sorted(q.values(), reverse=True)
            d["_ahat"] = max(q, key=q.get); d["_margin"] = order_q[0] - order_q[1]
            margins.append(d["_margin"])
        qs = sorted(margins) if margins else sorted(abs(KV[c["_c"]] - KV[c["_b"]]) for c in calib)
        thr = [qs[int(0.5 * (len(qs) - 1))], qs[int(0.8 * (len(qs) - 1))]] if qs else [0.0, 0.0]
        def bin_of(m):
            return 0 if m <= thr[0] else (1 if m <= thr[1] else 2)
        cur_pairs = []
        for c in calib:
            dh = KV[c["_c"]] - KV[c["_b"]]; dg = c["G_c"] - c["G_b"]
            cur_pairs.append({"bin": bin_of(abs(dh)), "dhat": dh, "dG": dg, "gain": dg * ((dh > 0) - (dh < 0))})
        hist = []
        for f in [x for x in args.calib_history.split(",") if x]:
            try:
                hist += json.load(open(f))["pairs"]
            except Exception as e:  # noqa: BLE001
                print("calib history unreadable", f, e, flush=True)
        if args.calib_rule == "lcb":
            w, g, se, n, lcb = calib_weights_lcb(hist, args.calib_min)
        else:
            w, g, se, n, lcb = calib_weights(hist, args.calib_min, args.calib_prior_n, args.calib_g_ref)
        n_w = [0, 0, 0]
        for d in distill:
            b = bin_of(d["_margin"]); n_w[b] += 1
            if args.distill_nu <= 0 or w[b] <= 0:
                continue
            acts = [x["action"] for x in d["samples"]]
            share = sum(a == d["_ahat"] for a in acts) / len(acts)
            for x in d["samples"]:
                a_k = args.distill_nu * w[b] * (float(x["action"] == d["_ahat"]) - share)
                if x["resp_ids"] and a_k != 0:
                    drows.append({"prompt_ids": d["prompt_ids"], "resp": x["resp_ids"], "old": x["lps"], "adv": a_k})
        acc = [p for p in cur_pairs if p["dG"] != 0 and p["dhat"] != 0]
        stats.update({"distill_states": len(distill), "distill_rows": len(drows), "distill_states_per_bin": n_w,
                      "margin_thresholds": [round(t, 5) for t in thr], "calib_hist_n": n,
                      "calib_hist_gain": [round(x, 4) for x in g], "calib_hist_lcb": [round(x, 4) for x in lcb], "calib_hist_se": [round(x, 4) if x < 1e9 else None for x in se],
                      "distill_weight_per_bin": [round(x, 3) for x in w], "calib_rule": args.calib_rule, "calib_pairs_now": len(cur_pairs),
                      "calib_now_sign_acc": [round(sum(p["gain"] > 0 for p in acc) / len(acc), 3) if acc else None, len(acc)],
                      "calib_now_gain_per_bin": [round(sts.mean([p["gain"] for p in cur_pairs if p["bin"] == b]), 4)
                                                 if any(p["bin"] == b for p in cur_pairs) else None for b in range(3)]})
        if rank == 0 and args.calib_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.calib_out)), exist_ok=True)
            json.dump({"thresholds": thr, "pairs": cur_pairs}, open(args.calib_out, "w"))

    # ---- 2. GAE on the tree, whitened
    A = gae(nodes, edges, V, args.lam)
    valid = [i for i, e in enumerate(edges) if A[i] is not None and e["resp"]]
    import statistics as sts
    mu = sts.mean([A[i] for i in valid]); sd = sts.pstdev([A[i] for i in valid]) or 1.0
    adv = {i: (A[i] - mu) / (sd + 1e-6) for i in valid}
    stats.update({"adv_raw_mean": round(mu, 4), "adv_raw_std": round(sd, 4), "policy_samples": len(valid)})

    # ---- 3. policy PPO-clip update
    model.set_adapter("default")
    model.train()
    rng = random.Random(args.seed)
    rows = [{"prompt_ids": prompt_ids(nodes[edges[ei]["node"]]), "resp": edges[ei]["resp"], "old": edges[ei]["old"],
             "adv": adv[ei]} for ei in valid] + drows
    order = list(range(len(rows))); rng.shuffle(order)
    per = max(1, args.mb // world) * world
    nsteps = len(order) // per
    ps = {"n": 0, "loss": 0.0, "ratio": 0.0, "clipfrac": 0.0, "kl": 0.0, "skipped_long": 0}
    for s in range(nsteps):
        chunk = order[s * per:(s + 1) * per][rank::world]
        ntok = sum(len(rows[ri]["resp"]) for ri in chunk) or 1
        opt_p.zero_grad(set_to_none=True)
        items = []
        for ri in chunk:
            r = rows[ri]; p_ids, resp = r["prompt_ids"], r["resp"]
            if len(p_ids) + len(resp) > args.max_len:
                p_ids = p_ids[-(args.max_len - len(resp)):]
                ps["skipped_long"] += 1
            items.append((p_ids, resp, r))
        for grp in (packs(items, args.pack_tokens) if args.pack_tokens > 0 else [[it] for it in items]):
            if args.pack_tokens > 0:
                logp = -pack_nll(model, [(p_ids, resp) for p_ids, resp, _ in grp])
            else:
                p_ids, resp, _ = grp[0]
                out = model(input_ids=torch.tensor([p_ids + resp], device="cuda"), logits_to_keep=len(resp) + 1)
                logp = torch.log_softmax(out.logits[0, :-1].float(), -1).gather(
                    -1, torch.tensor(resp, device="cuda")[:, None])[:, 0]
            old = torch.tensor([x for _, resp, r in grp for x in r["old"][: len(resp)]], device="cuda", dtype=torch.float32)
            a = torch.tensor([float(r["adv"]) for _, resp, r in grp for _ in resp], device="cuda", dtype=torch.float32)
            ratio = torch.exp(logp - old)
            loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.eps_low, 1 + args.eps_high) * a).sum() / ntok
            loss.backward()
            with torch.no_grad():
                off = 0
                for _, resp, _ in grp:
                    rr, oo, ll = ratio[off:off + len(resp)], old[off:off + len(resp)], logp[off:off + len(resp)]
                    ps["n"] += 1; ps["ratio"] += rr.mean().item(); ps["kl"] += (oo - ll).mean().item()
                    ps["clipfrac"] += ((rr < 1 - args.eps_low) | (rr > 1 + args.eps_high)).float().mean().item()
                    off += len(resp)
                ps["loss"] += loss.item()
        for p in pol_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(pol_params, 1.0)
        opt_p.step()
        if rank == 0 and (s % 10 == 0 or s == nsteps - 1):
            n = max(ps["n"], 1)
            print(f"policy step {s + 1}/{nsteps} loss {ps['loss'] / n:+.4f} ratio {ps['ratio'] / n:.4f} "
                  f"kl {ps['kl'] / n:+.5f} {time.time() - t0:.0f}s", flush=True)

    # ---- 4. critic update: units = sibling groups (their non-terminal children together) + remaining nodes
    model.set_adapter("critic")
    model.train()
    in_group = set()
    units = []
    for g in groups:
        kids = [edges[ei]["child"] for ei in g if not edges[ei]["term"]]
        if len(kids) >= 1:
            units.append(("group", g))
            in_group.update(kids)
    for i, nd in enumerate(nodes):
        if nd["G"] is not None and i not in in_group:
            units.append(("node", i))
    rng.shuffle(units)
    nsteps_c = len(units) // per
    cs = {"n": 0, "mse": 0.0, "sib": 0.0}
    for s in range(nsteps_c):
        chunk = units[s * per:(s + 1) * per][rank::world]
        opt_c.zero_grad(set_to_none=True)
        for kind, u in chunk:
            if kind == "node":
                v = value(nodes[u], True)
                loss = (v - nodes[u]["G"]) ** 2
                cs["mse"] += loss.item(); cs["n"] += 1
            else:
                vals, Gs, mse = [], [], 0.0
                for ei in u:
                    e = edges[ei]
                    if e["term"]:
                        vals.append(torch.tensor(e["R"], device="cuda")); Gs.append(e["G"])
                    else:
                        v = value(nodes[e["child"]], True)
                        mse = mse + (v - nodes[e["child"]]["G"]) ** 2
                        vals.append(v); Gs.append(e["G"])
                        cs["n"] += 1
                vt = torch.stack(vals); gt = torch.tensor(Gs, device="cuda")
                sib = (((vt - vt.mean()) - (gt - gt.mean())) ** 2).sum()
                loss = mse + args.sib_weight * sib
                cs["mse"] += float(mse); cs["sib"] += sib.item()
            (loss / len(chunk)).backward()
        for p in cri_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(cri_params, 1.0)
        opt_c.step()
        if rank == 0 and (s % 10 == 0 or s == nsteps_c - 1):
            print(f"critic step {s + 1}/{nsteps_c} mse {cs['mse'] / max(cs['n'], 1):.4f} sib {cs['sib']:.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)

    # ---- 5. Q update (Q-PPO): every executed edge; units = sibling groups (sibling-centred term) + remaining edges
    model.set_adapter("qnet")
    model.train()
    qin, qunits = set(), []
    for g in groups:
        ex = [ei for ei in g if edges[ei]["resp"]]
        if len(ex) >= 2:
            qunits.append(("group", ex)); qin.update(ex)
    qunits += [("edge", ei) for ei, e in enumerate(edges) if e["resp"] and e.get("G") is not None and ei not in qin]
    rng.shuffle(qunits)
    nsteps_q = len(qunits) // per
    qs_ = {"n": 0, "mse": 0.0, "sib": 0.0}
    for s in range(nsteps_q):
        chunk = qunits[s * per:(s + 1) * per][rank::world]
        opt_q.zero_grad(set_to_none=True)
        for kind_, u in chunk:
            eis = [u] if kind_ == "edge" else u
            qv = torch.stack([qvalue(prompt_ids(nodes[edges[ei]["node"]]) + edges[ei]["resp"], True) for ei in eis])
            gt = torch.tensor([edges[ei]["G"] for ei in eis], device="cuda", dtype=torch.float32)
            mse = ((qv - gt) ** 2).sum()
            loss = mse
            if kind_ == "group":
                sib = (((qv - qv.mean()) - (gt - gt.mean())) ** 2).sum()
                loss = loss + args.sib_weight * sib; qs_["sib"] += sib.item()
            qs_["mse"] += mse.item(); qs_["n"] += len(eis)
            (loss / len(chunk)).backward()
        for p in q_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(q_params, 1.0)
        opt_q.step()
        if rank == 0 and (s % 10 == 0 or s == nsteps_q - 1):
            print(f"q step {s + 1}/{nsteps_q} mse {qs_['mse'] / max(qs_['n'], 1):.4f} sib {qs_['sib']:.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    qagg = torch.tensor([qs_["n"], qs_["mse"], qs_["sib"]], device="cuda", dtype=torch.float64)
    dist.all_reduce(qagg)
    stats.update({"q_edges": int(qagg[0].item()), "q_mse": qagg[1].item() / max(qagg[0].item(), 1),
                  "q_sib_loss": qagg[2].item(), "q_steps": nsteps_q})

    agg = torch.tensor([ps[k] for k in ("n", "loss", "ratio", "clipfrac", "kl", "skipped_long")] +
                       [cs["n"], cs["mse"], cs["sib"]], device="cuda", dtype=torch.float64)
    dist.all_reduce(agg)
    if rank == 0:
        model.save_pretrained(args.policy_out, selected_adapters=["default"])
        os.makedirs(args.critic_out, exist_ok=True)
        model.save_pretrained(args.critic_out, selected_adapters=["critic"])
        torch.save(vhead.state_dict(), os.path.join(args.critic_out, "value_head.pt"))
        model.save_pretrained(args.critic_out, selected_adapters=["qnet"])
        torch.save(qhead.state_dict(), os.path.join(args.critic_out, "q_head.pt"))
        n = max(agg[0].item(), 1)
        stats.update({"samples": int(agg[0].item()), "loss": agg[1].item() / n, "ratio": agg[2].item() / n,
                      "clipfrac": agg[3].item() / n, "kl": agg[4].item() / n, "skipped_long": int(agg[5].item()),
                      "critic_states": int(agg[6].item()), "critic_mse": agg[7].item() / max(agg[6].item(), 1),
                      "critic_sib_loss": agg[8].item(), "steps": nsteps, "critic_steps": nsteps_c,
                      "wall_s": round(time.time() - t0, 1)})
        print("TRAIN_STATS " + json.dumps(stats), flush=True)
        if args.stats_out:
            json.dump(stats, open(args.stats_out, "w"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
