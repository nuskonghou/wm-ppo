"""PaW baseline on PPO-160: train_ppo_alf.py plus PaW's auxiliary next-observation loss on the policy (Policy and World
Modeling co-training; settings of the paper and its released code: compute_world_model_loss and rollout_loop.py).
The PPO-clip policy update, the critic update and everything else are those of train_ppo_alf.py; the PaW part:
  - after each executed response (which ends with <|im_end|>) a prediction turn is appended in the chat template: a user
    turn with one of PaW's 20 prediction prompts (drawn at random) + PaW's instruction to output the observation
    directly, and an assistant turn holding the true next observation; only the answer tokens (observation +
    <|im_end|> + newline) carry the loss, and the appended turn never enters the policy loss;
  - transitions: deduplicated (prompt + response + observation), then per step index t the top 75% by action
    uncertainty, the mean negative log-probability of the response tokens under the sampler;
  - loss: clipped MAE, mean over answer tokens of (1 - p) 1[p <= rho] (rho = 0.2), times the reward-adaptive coefficient
    [(R_max - mean episode return of the batch) / R_max]_+, times --paw-coef; added to the policy loss of the same
    minibatch. Edges carry 'next_obs' and 't' (written by alf_final.py for arm ppo160paw).
The prompts are read from a clone of the PaW repository (paw_constants)."""
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


# ---- PaW's prediction prompts are read from a clone of the PaW repository (github.com/ColinLu50/Policy-WorldModel-Agent,
# commit 4feb964, agent_system/multi_turn_rollout/rollout_loop.py): $PAW_REPO or <repo>/third_party/Policy-WorldModel-Agent
def paw_constants():
    """(DIRECT_OBS_INSTRUCTION, PREDICTION_PROMPTS) of the PaW repository."""
    import ast
    root = os.environ.get('PAW_REPO') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      'third_party', 'Policy-WorldModel-Agent')
    path = os.path.join(root, 'agent_system', 'multi_turn_rollout', 'rollout_loop.py')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: clone github.com/ColinLu50/Policy-WorldModel-Agent (commit 4feb964) '
                                'into third_party/Policy-WorldModel-Agent or set PAW_REPO')
    vals = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ('DIRECT_OBS_INSTRUCTION', 'PREDICTION_PROMPTS'):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    return vals['DIRECT_OBS_INSTRUCTION'], list(vals['PREDICTION_PROMPTS'])


DIRECT_OBS_INSTRUCTION, PREDICTION_PROMPTS = paw_constants()


def paw_turn_ids(tok, ask_text, obs_text):
    """PaW's _build_llm_predict_turn_ids: (ids of the appended prediction turn, answer start, answer length)."""
    ask = f"{ask_text}\n{DIRECT_OBS_INSTRUCTION}"
    U, A = "__paw_dummy_user__", "__paw_dummy_assistant__"
    pre = [{"role": "user", "content": U}, {"role": "assistant", "content": A}]
    askm = pre + [{"role": "user", "content": ask}]
    full = askm + [{"role": "assistant", "content": obs_text}]
    prefix_text = tok.apply_chat_template(pre, add_generation_prompt=False, tokenize=False)
    before = tok.apply_chat_template(askm, add_generation_prompt=True, tokenize=False)
    full_text = tok.apply_chat_template(full, add_generation_prompt=False, tokenize=False)
    auto = "<think>\n\n</think>\n\n" + A                       # Qwen3 auto-injected empty think block (no-op otherwise)
    if auto in prefix_text:
        prefix_text = prefix_text.replace(auto, A, 1)
    b = prefix_text.rstrip()
    if not (full_text.startswith(b) and full_text.startswith(before)):
        raise ValueError("PaW prediction turn: chat template does not split at the prefix")
    ids = tok.encode(full_text[len(b):], add_special_tokens=False)
    return ids, len(tok.encode(before[len(b):], add_special_tokens=False)), len(tok.encode(full_text[len(before):], add_special_tokens=False))


def paw_select(tok, cand, ratio, rng, eos):
    """cand: [(row index, edge)]; returns {row index: (appended ids, answer start, answer length)} for the kept rows
    (dedup by prompt + response + observation, then per step t the top `ratio` by mean response NLL), and counts."""
    import hashlib
    by_t, seen, n_dup, n_noeos = {}, set(), 0, 0
    for k, e in cand:
        if eos not in e["resp"]:
            n_noeos += 1; continue                      # truncated response: PaW skips the append
        h = hashlib.sha1(repr((e["_p"], e["resp"], e["next_obs"])).encode()).digest()
        if h in seen:
            n_dup += 1; continue
        seen.add(h)
        n = len(e["old"]) or 1
        by_t.setdefault(e.get("t", 0), []).append((-sum(e["old"]) / n, k, e))
    out = {}
    for t, xs in by_t.items():
        xs.sort(key=lambda x: -x[0])
        for _, k, e in xs[:min(len(xs), max(1, int(len(xs) * ratio)))]:
            out[k] = paw_turn_ids(tok, rng.choice(PREDICTION_PROMPTS), e["next_obs"])
    return out, {"paw_eligible": sum(len(x) for x in by_t.values()), "paw_dedup": n_dup, "paw_no_eos": n_noeos}


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
    ap.add_argument("--sc-strict", action="store_true", help="majority label only with a unique plurality (ties skipped)")
    ap.add_argument("--calib-history", default="", help="comma-separated calib_scored.json of previous iterations")
    ap.add_argument("--calib-out", default="")
    ap.add_argument("--calib-min", type=int, default=30, help="minimum calibration pairs per margin bin")
    ap.add_argument("--calib-prior-n", type=float, default=40.0, help="shrinkage strength n_0 (pseudo-pairs) for the bin gain")
    ap.add_argument("--calib-g-ref", type=float, default=0.05, help="calibrated gain per decision that earns weight 1")
    ap.add_argument("--calib-rule", choices=["posterior", "lcb"], default="posterior")
    ap.add_argument("--cons-weight", type=float, default=0.0,
                    help="value channel: weight beta of the critic loss towards the predicted successors' values at the "
                         "gated decision points (times the bin's gate weight)")
    ap.add_argument("--cons-mode", choices=["full", "terminal"], default="full",
                    help="full: target = sum_a share(a) Q_hat(a) (predicted successor valued by the pre-update critic, endings 1/0); "
                         "terminal: only predicted endings move the target, V_old(h) + sum_{a ending} share(a)(end(a) - V_old(h))")
    ap.add_argument("--pack-tokens", type=int, default=0, help="> 0: pack policy rows (flash-attn varlen)")
    ap.add_argument("--paw-coef", type=float, default=1.0, help="PaW loss_coefficient")
    ap.add_argument("--paw-select", type=float, default=0.75, help="PaW selection_ratio (top share by action uncertainty)")
    ap.add_argument("--paw-rho", type=float, default=0.2, help="PaW cmae_threshold")
    ap.add_argument("--paw-rmax", type=float, default=1.0, help="maximum episode return (0/1 rewards here)")
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
    pol_params = [p for n, p in model.named_parameters() if "lora_" in n and ".default." in n]
    cri_params = [p for n, p in model.named_parameters() if "lora_" in n and ".critic." in n] + list(vhead.parameters())
    for p in pol_params + cri_params:
        p.data = p.data.float()
    opt_p = torch.optim.AdamW(pol_params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    opt_c = torch.optim.AdamW(cri_params, lr=args.critic_lr, betas=(0.9, 0.99), weight_decay=0.0)
    inner = model.base_model.model.model  # decoder stack with LoRA layers injected

    def prompt_ids(nd):
        return nd["prompt_ids"]

    def value(nd, grad):
        ids = prompt_ids(nd)[-args.max_len:]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            h = inner(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]
            return vhead(h.float())[0]

    nodes, edges, groups, distill, calib = load(args.trees)
    t0 = time.time()
    cons = []                                   # value channel: (node index, target, weight)
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
            if args.sc_strict and sum(c == cnt[a_hat] for c in cnt.values()) > 1:
                continue                                                # no unique majority: no label
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
        for i in range(rank, len(kid_ids), world):
            KV[i] = value({"prompt_ids": kid_ids[i]}, False).detach()
        dist.all_reduce(KV)
        KV = KV.tolist()
        n_term = 0                    # ScienceWorld: a predicted ending replaces the critic's value (1 / 0)
        for d in distill:
            for a, k in d["_k"].items():
                t = d.get("term", {}).get(a)
                if t is not None:
                    KV[k] = float(t); n_term += 1
        for c in calib:
            for key, k in (("term_c", c["_c"]), ("term_b", c["_b"])):
                if c.get(key) is not None:
                    KV[k] = float(c[key]); n_term += 1
        stats["wm_terminal_overrides"] = n_term
        margins = []
        for d in distill:
            q = {a: KV[k] for a, k in d["_k"].items()}
            order_q = sorted(q.values(), reverse=True)
            d["_ahat"] = max(q, key=q.get); d["_margin"] = order_q[0] - order_q[1]
            if d["_margin"] > 0:          # exact ties (e.g. two predicted failures, both 0) carry no preference
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
            if d["_margin"] <= 0:
                continue
            b = bin_of(d["_margin"]); n_w[b] += 1
            if args.distill_nu <= 0 or w[b] <= 0:
                continue
            acts = [x["action"] for x in d["samples"]]
            share = sum(a == d["_ahat"] for a in acts) / len(acts)
            for x in d["samples"]:
                a_k = args.distill_nu * w[b] * (float(x["action"] == d["_ahat"]) - share)
                if x["resp_ids"] and a_k != 0:
                    drows.append({"prompt_ids": d["prompt_ids"], "resp": x["resp_ids"], "old": x["lps"], "adv": a_k})
        if args.cons_weight > 0:                # value channel: targets for V at the gated decision points
            nidx = {tuple(nd["prompt_ids"]): i for i, nd in enumerate(nodes)}
            dg = []
            for d in distill:
                ni = nidx.get(tuple(d["prompt_ids"]))
                if ni is None or d["_margin"] <= 0 or w[bin_of(d["_margin"])] <= 0:
                    continue
                acts = [x["action"] for x in d["samples"]]
                share = {a: acts.count(a) / len(acts) for a in set(acts)}
                if args.cons_mode == "full":
                    tgt = sum(sh * KV[d["_k"][a]] for a, sh in share.items())
                else:
                    ends = {a: t for a, t in d.get("term", {}).items() if t is not None and a in share}
                    if not ends:
                        continue
                    tgt = V[ni] + sum(share[a] * (float(t) - V[ni]) for a, t in ends.items())
                cons.append((ni, float(tgt), args.cons_weight * w[bin_of(d["_margin"])]))
                if nodes[ni]["G"] is not None:
                    dg.append(((tgt - nodes[ni]["G"]) ** 2, (V[ni] - nodes[ni]["G"]) ** 2, abs(tgt - V[ni])))
            stats.update({"cons_states": len(cons), "cons_mode": args.cons_mode,
                          "cons_target_mse_vs_G": round(sum(x[0] for x in dg) / len(dg), 4) if dg else None,
                          "cons_Vold_mse_vs_G": round(sum(x[1] for x in dg) / len(dg), 4) if dg else None,
                          "cons_mean_abs_shift": round(sum(x[2] for x in dg) / len(dg), 4) if dg else None})
        acc = [p for p in cur_pairs if p["dG"] != 0 and p["dhat"] != 0]
        stats.update({"distill_states": len(distill), "distill_rows": len(drows), "distill_states_per_bin": n_w,
                      "distill_ties": sum(1 for d in distill if d["_margin"] <= 0),
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
    # ---- PaW: prediction turns for the selected executed transitions, reward-adaptive coefficient
    kids = {e["child"] for e in edges if e["child"] is not None}
    roots = [nd["G"] for i, nd in enumerate(nodes) if i not in kids and nd["G"] is not None]
    ep_mean = sts.mean(roots) if roots else 0.0
    paw_lam = max(0.0, (args.paw_rmax - ep_mean) / args.paw_rmax)
    for ei in valid:
        edges[ei]["_p"] = prompt_ids(nodes[edges[ei]["node"]])
    sel, pst = paw_select(tok, [(k, edges[ei]) for k, ei in enumerate(valid) if edges[ei].get("next_obs") is not None],
                          args.paw_select, random.Random(args.seed + 17), tok.convert_tokens_to_ids("<|im_end|>"))
    for k, x in sel.items():
        rows[k]["paw"] = x
    stats.update(pst, paw_rows=len(sel), paw_episode_mean_return=round(ep_mean, 4), paw_lambda=round(paw_lam, 4))
    order = list(range(len(rows))); rng.shuffle(order)
    per = max(1, args.mb // world) * world
    nsteps = len(order) // per
    ps = {"n": 0, "loss": 0.0, "ratio": 0.0, "clipfrac": 0.0, "kl": 0.0, "skipped_long": 0,
          "wm_loss": 0.0, "wm_tok": 0, "wm_p": 0.0, "wm_active": 0.0}
    for s in range(nsteps):
        chunk = order[s * per:(s + 1) * per][rank::world]
        ntok = sum(len(rows[ri]["resp"]) for ri in chunk) or 1
        opt_p.zero_grad(set_to_none=True)
        items = []
        n_wm = sum(rows[ri]["paw"][2] for ri in chunk if "paw" in rows[ri])   # PaW token-mean over this chunk's answer tokens
        for ri in chunk:
            r = rows[ri]; p_ids = r["prompt_ids"]
            full = r["resp"] + r["paw"][0] if "paw" in r else r["resp"]       # response, then PaW's prediction turn
            if len(p_ids) + len(full) > args.max_len:
                p_ids = p_ids[-(args.max_len - len(full)):]
                ps["skipped_long"] += 1
            items.append((p_ids, full, r))
        for grp in (packs(items, args.pack_tokens) if args.pack_tokens > 0 else [[it] for it in items]):
            if args.pack_tokens > 0:
                logp_all = -pack_nll(model, [(p_ids, full) for p_ids, full, _ in grp])
            else:
                p_ids, full, _ = grp[0]
                out = model(input_ids=torch.tensor([p_ids + full], device="cuda"), logits_to_keep=len(full) + 1)
                logp_all = torch.log_softmax(out.logits[0, :-1].float(), -1).gather(
                    -1, torch.tensor(full, device="cuda")[:, None])[:, 0]
            pol_idx, wm_idx, off = [], [], 0
            for _, full, r in grp:
                n = len(r["resp"]); pol_idx += range(off, off + n)
                if "paw" in r:
                    s0 = off + n + r["paw"][1]; wm_idx += range(s0, s0 + r["paw"][2])
                off += len(full)
            logp = logp_all[torch.tensor(pol_idx, device="cuda")]
            old = torch.tensor([x for _, _, r in grp for x in r["old"][: len(r["resp"])]], device="cuda", dtype=torch.float32)
            a = torch.tensor([float(r["adv"]) for _, _, r in grp for _ in r["resp"]], device="cuda", dtype=torch.float32)
            ratio = torch.exp(logp - old)
            loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.eps_low, 1 + args.eps_high) * a).sum() / ntok
            if wm_idx:                                   # PaW clipped MAE: (1 - p) 1[p <= rho], token-mean, x lambda x coef
                pw = logp_all[torch.tensor(wm_idx, device="cuda")].exp()
                act = (pw.detach() <= args.paw_rho).float()
                lw = ((1.0 - pw) * act).sum() / max(n_wm, 1)
                loss = loss + args.paw_coef * paw_lam * lw
                ps["wm_loss"] += lw.item(); ps["wm_tok"] += len(wm_idx); ps["wm_p"] += pw.detach().sum().item()
                ps["wm_active"] += act.sum().item()
            loss.backward()
            with torch.no_grad():
                off = 0
                for _, _, r in grp:
                    n = len(r["resp"])
                    rr, oo, ll = ratio[off:off + n], old[off:off + n], logp[off:off + n]
                    ps["n"] += 1; ps["ratio"] += rr.mean().item(); ps["kl"] += (oo - ll).mean().item()
                    ps["clipfrac"] += ((rr < 1 - args.eps_low) | (rr > 1 + args.eps_high)).float().mean().item()
                    off += n
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
    units += [("cons", c) for c in cons]          # value channel
    rng.shuffle(units)
    nsteps_c = len(units) // per
    cs = {"n": 0, "mse": 0.0, "sib": 0.0}
    for s in range(nsteps_c):
        chunk = units[s * per:(s + 1) * per][rank::world]
        opt_c.zero_grad(set_to_none=True)
        for kind, u in chunk:
            if kind == "cons":                    # value channel: beta * w_b * (V(h_t) - sg[target])^2
                ni, tgt, wt = u
                v = value(nodes[ni], True)
                loss = wt * (v - tgt) ** 2
                cs["cons"] = cs.get("cons", 0.0) + loss.item()
            elif kind == "node":
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
            print(f"critic step {s + 1}/{nsteps_c} mse {cs['mse'] / max(cs['n'], 1):.4f} sib {cs['sib']:.3f} cons {cs.get('cons', 0.0):.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)

    agg = torch.tensor([ps[k] for k in ("n", "loss", "ratio", "clipfrac", "kl", "skipped_long")] +
                       [cs["n"], cs["mse"], cs["sib"]] + [ps[k] for k in ("wm_loss", "wm_tok", "wm_p", "wm_active")],
                       device="cuda", dtype=torch.float64)
    dist.all_reduce(agg)
    if rank == 0:
        model.save_pretrained(args.policy_out, selected_adapters=["default"])
        os.makedirs(args.critic_out, exist_ok=True)
        model.save_pretrained(args.critic_out, selected_adapters=["critic"])
        torch.save(vhead.state_dict(), os.path.join(args.critic_out, "value_head.pt"))
        n = max(agg[0].item(), 1)
        stats.update({"samples": int(agg[0].item()), "loss": agg[1].item() / n, "ratio": agg[2].item() / n,
                      "clipfrac": agg[3].item() / n, "kl": agg[4].item() / n, "skipped_long": int(agg[5].item()),
                      "critic_states": int(agg[6].item()), "critic_mse": agg[7].item() / max(agg[6].item(), 1),
                      "critic_sib_loss": agg[8].item(), "steps": nsteps, "critic_steps": nsteps_c,
                      "wall_s": round(time.time() - t0, 1)})
        wt = max(agg[10].item(), 1)                     # PaW diagnostics (released code's wm_obs_mean_prob / cmae clip ratio)
        stats.update({"paw_wm_tokens": int(agg[10].item()), "paw_wm_mean_prob": round(agg[11].item() / wt, 4),
                      "paw_cmae_active_ratio": round(agg[12].item() / wt, 4), "paw_wm_loss_sum": round(agg[9].item(), 4)})
        print("TRAIN_STATS " + json.dumps(stats), flush=True)
        if args.stats_out:
            json.dump(stats, open(args.stats_out, "w"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
