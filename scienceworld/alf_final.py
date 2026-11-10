"""ScienceWorld training loop (orchestrator) of WM-PPO and the baselines; the same loop as alfworld/alf_final.py on the
ScienceWorld environment of env.py. All methods train LoRA adapters on Qwen3-4B-Instruct-2507 (non-thinking mode); the
policy is served by sglang with dynamic LoRA loading, the environment workers run here (rollout.Pool), and every update
runs in the trainer container (train_ppo_alf*.py, train_lora.py).

One iteration: rollouts (chains, branches, candidate samples, world-model predictions) -> tree.json / samples.jsonl ->
one policy (and critic) update -> the new adapter is loaded on the policy server -> for WM-PPO the world model is
fine-tuned on the iteration's executed transitions, concurrently with the policy update -> greedy evaluation every
--eval-every iterations. Arms (tasks per iteration at --scale 1, chains per task, branches per task):
  ppo           128 x 1 chain; PPO with a learned critic, GAE (the shared warm-up, iterations 1-12)
  ppo160        PPO-160: 320 x 1 (160 at --scale 0.5);   ppo256: 512 x 1 (PPO-256)
  treeppo_wide  Wide-Tree-PPO: 128 x (1 chain + 3 branches); critic + tree GAE + sibling term
  wmppo         WM-PPO: the trees of treeppo_wide + K = 4 samples per chain step (the first is executed); the world
                model predicts the successor of every distinct sample (a predicted ending counts as success / failure),
                the critic values the predicted successors, and train_ppo_alf.py adds the gated auxiliary rows (policy
                channel) and, with --critic-cons-weight, the value-channel targets; calibration pairs come from the
                branches. --wm-online fine-tunes the world model every iteration (without it: Fixed WM)
  qppo          Q-PPO: wmppo's pipeline with a learned Q(h, y) as the scorer (train_ppo_alf_q.py)
  grpo160 / gigpo160   20 x 8 episodes, GRPO-family advantages (grpo_family.py); grpo160g4 / gigpo160g4: 40 x 4
  capo160v2, sopo160, sopo160v2, steppo160v2   PPO-160's sampling with CAPO / SO-PPO / ST-PPO updates
  ppo160mv, ppo160opd, ppo160ocsd, ppo160paw   PPO-160 + majority-vote labels / standard OPD / OCSD / PaW
A branch replays the chain's prefix (ScienceWorld is deterministic) and resamples the first step until its action
differs from the chain's (up to --k-alt tries).
"""
import argparse, json, os, random, subprocess, sys, time
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from env import game_files, task_type  # noqa: E402
from rollout import Pool  # noqa: E402
from wmppo import paths  # noqa: E402

IMAGE = paths.TRAINER_IMAGE
GRPO_FAMILY = {'grpo160': 'grpo', 'gigpo160': 'gigpo',   # 20 tasks x 8
               'grpo160g4': 'grpo',   # GRPO with 40 tasks x 4 (the same 160 episodes)
               'gigpo160g4': 'gigpo'}   # GiGPO with 40 tasks x 4
ARMS = {'ppo': (128, 1, 0),
        **{a: (40, 8, 0) for a in GRPO_FAMILY}, 'grpo160g4': (80, 4, 0), 'gigpo160g4': (80, 4, 0),
        'treeppo_wide': (128, 1, 3), 'wmppo': (128, 1, 3), 'ppo256': (512, 1, 0),
        'qppo': (128, 1, 3),   # Q-PPO: WM-PPO with a learned Q(h, y) scorer instead of world model + V (train_ppo_alf_q.py)
        'sopo160': (320, 1, 0),   # SO-PPO with the published clip, train_ppo_alf_ppox.py
        'sopo160v2': (320, 1, 0),   # SO-PPO with the turn-ratio clip of --ppox-eps
        'ppo160': (320, 1, 0),
        'ppo160paw': (320, 1, 0),   # PPO-160 + PaW's next-observation loss on the policy (train_ppo_alf_paw.py)
        'capo160v2': (320, 1, 0), 'steppo160v2': (320, 1, 0),   # CAPO / ST-PPO with the clip of --capo-eps / --ppox-eps
        'ppo160mv': (320, 1, 0),   # PPO-160 + majority-vote labels (K samples per step, fixed weight 1, no gate)
        'ppo160opd': (320, 1, 0),   # PPO-160 + standard OPD (train_ppo_alf_opd.py)
        'ppo160ocsd': (320, 1, 0)}   # PPO-160 + OCSD: future-evidence teacher, advantage modulation (train_ppo_alf_ocsd.py)


def sg_post(url, path, payload, timeout=600):
    r = requests.post(url + path, json=payload, timeout=timeout); r.raise_for_status()
    try: return r.json()
    except Exception: return r.text


def load_adapter(url, name, path, old=None):
    try:
        sg_post(url, '/load_lora_adapter', {'lora_name': name, 'lora_path': path})
    except requests.HTTPError:          # e.g. the same name is already loaded (after a restart): reload it
        try: sg_post(url, '/unload_lora_adapter', {'lora_name': name})
        except Exception: pass
        sg_post(url, '/load_lora_adapter', {'lora_name': name, 'lora_path': path})
    if old:
        try: sg_post(url, '/unload_lora_adapter', {'lora_name': old})
        except Exception as e: print('unload failed', e, flush=True)


def job(args, gf, prefix=(), tag=None, lora=None, temperature=None, avoid=None, K=1, wm=False, opd=False, ocsd=False, paw=False):
    j = {'gamefile': gf, 'prefix': list(prefix), 'url': args.sglang, 'lora': lora, 'avoid_first': avoid, 'k_alt': args.k_alt,
         'temperature': args.temperature if temperature is None else temperature,
         'max_new_tokens': args.max_new_tokens, 'max_steps': args.max_steps, 'model': args.model, 'tag': tag, 'K': K}
    if opd:                                         # ppo160opd: the teacher scores the executed response
        j['opd'] = True
        if args.opd_url:                            # stronger teacher (base weights, same tokenizer)
            j['opd_url'] = args.opd_url; j['opd_lora'] = None
            j['opd_prefix_ids'] = [151667, 271, 151668, 271]   # Qwen3-32B non-thinking mode: '<think>\n\n</think>\n\n'
    if ocsd:                                        # ppo160ocsd: OCSD teacher scores (rollout.ocsd_scores)
        j['ocsd'] = True; j['ocsd_env'] = 'scienceworld'; j['ocsd_top'] = args.ocsd_top; j['ocsd_k'] = args.ocsd_k
        if args.ocsd_url:                           # stronger teacher (Qwen3-32B base weights, same tokenizer)
            j['ocsd_url'] = args.ocsd_url; j['ocsd_prefix_ids'] = [151667, 271, 151668, 271]
    if paw:                                         # ppo160paw: record each executed action's next observation
        j['paw'] = True
    if wm and getattr(args, 'wm_online', False):
        j['record_wm'] = True
    if wm and args.wm_url:
        urls = args.wm_url.split(',')
        j['wm_url'] = urls[hash((gf, len(prefix))) % len(urls)]; j['wm_lora'] = args.wm_lora
    return j


def evaluate(args, pool, lora, splits=('valid_seen', 'valid_unseen')):
    res = {}
    for sp in splits:
        eps = pool.run([job(args, gf, tag=sp, lora=lora, temperature=0.0) for gf in game_files(sp)])
        ok = [e for e in eps if 'error' not in e]
        res[sp] = float(np.mean([e['reward'] for e in ok])) if ok else float('nan')
        res[sp + '_errors'] = len(eps) - len(ok)
        res[sp + '_per_game'] = {e['gamefile']: e['reward'] for e in ok}      # ScienceWorld: '<task>|<variation>'
        res[sp + '_failed'] = float(np.mean([e['final_obs'].endswith('[task failed]') for e in ok])) if ok else float('nan')
        res[sp + '_truncated'] = float(np.mean([e['reward'] == 0 and not e['final_obs'].endswith('[task failed]') for e in ok])) if ok else float('nan')
        res[sp + '_steps'] = float(np.mean([e['steps'] for e in ok])) if ok else float('nan')
        by = {}
        for e in ok: by.setdefault(task_type(e['gamefile']), []).append(e['reward'])
        res[sp + '_by_type'] = {k: round(float(np.mean(v)), 3) for k, v in by.items()}
    return res


def build_tree(task_eps, nodes, edges, groups):
    """Append one task's episodes (chains first, then branches) to the global node/edge lists.
    Node = action prefix; edge = (prefix, action). Token ids of an edge come from the episode that generated it."""
    nkey, ekey = {}, {}
    pending = []
    for e in task_eps:
        acts, R = e['actions'], float(e['reward'])
        gen = {tu['t']: tu for tu in e['turns']}
        for t in range(len(acts)):
            key = tuple(acts[:t])
            if key not in nkey:
                tu = gen.get(t)
                if tu is None:
                    break                      # prefix state never generated by an episode we kept
                nkey[key] = len(nodes); nodes.append({'prompt_ids': tu['prompt_ids'], 'rs': []})
            nodes[nkey[key]]['rs'].append(R)
            ek = (key, acts[t])
            if ek not in ekey:
                tu = gen.get(t)
                if tu is None:
                    break
                term = t == len(acts) - 1
                ekey[ek] = len(edges)
                edges.append({'node': nkey[key], 'child': None, 'term': term, 'R': R if term else 0.0, 'rs': [],
                              'resp': tu['resp_ids'], 'old': tu['lps'], '_ck': tuple(acts[:t + 1]),
                              **({'opd': tu['opd_lps']} if tu.get('opd_lps') else {}),       # ppo160opd teacher log-probs
                              **({'ocsd': tu['ocsd']} if tu.get('ocsd') else {}),
                              **({'next_obs': tu['next_obs'], 't': t} if 'next_obs' in tu else {})})   # ppo160paw target
                if not term:
                    pending.append(ekey[ek])
            edges[ekey[ek]]['rs'].append(R)
    for ei in pending:
        edges[ei]['child'] = nkey.get(edges[ei]['_ck'])
        if edges[ei]['child'] is None:         # child never generated (should not happen): treat as terminal
            edges[ei]['term'] = True; edges[ei]['R'] = float(np.mean(edges[ei]['rs']))
    outs = {}
    for ei in ekey.values():
        outs.setdefault(edges[ei]['node'], []).append(ei)
    for ni, lst in outs.items():
        if len(lst) >= 2:
            groups.append(lst)


def train(args, it_dir, st, kind, data):
    out = os.path.join(it_dir, 'adapter'); stats = os.path.join(it_dir, 'train_stats.json')
    if kind == 'ppo':
        script = (f"{HERE}/train_ppo_alf.py --trees {data} --base {args.model} --policy-in {st['adapter'] or 'none'} "
                  f"--policy-out {out} --critic-in {st.get('critic') or 'none'} --critic-out {os.path.join(it_dir, 'critic')} "
                  f"--lr {args.lr} --critic-lr {args.critic_lr} --lam {args.lam} --sib-weight {args.sib_weight} --mb {args.mb} "
                  f"--stats-out {stats}")
        if args.arm in ('ppo', 'treeppo_wide', 'wmppo', 'ppo256', 'ppo160', 'qppo', 'sopo160', 'ppo160mv', 'ppo160opd', 'ppo160ocsd', 'sopo160v2', 'capo160v2', 'steppo160v2', 'ppo160paw'):
            script += " --pack-tokens 16384"           # packed policy loop (also the warm-up)
        if args.arm == 'capo160v2':                    # CAPO action-level ratio + discounted GAE (gamma 0.99)
            script = script.replace(f"{HERE}/train_ppo_alf.py", f"{HERE}/train_ppo_alf_capo.py", 1)
            script += f" --gamma 0.99 --capo-eps {args.capo_eps} --capo-alpha 0.5"
        if args.arm == 'qppo':
            script = script.replace(f"{HERE}/train_ppo_alf.py", f"{HERE}/train_ppo_alf_q.py", 1)
        if args.arm == 'ppo160opd':                    # per-token teacher - sampler log-prob added to the advantage
            script = script.replace(f"{HERE}/train_ppo_alf.py", f"{HERE}/train_ppo_alf_opd.py", 1) + f" --opd-coef {args.opd_coef} --opd-clip {args.opd_clip}"
        if args.arm == 'ppo160paw':                    # PaW: clipped-MAE next-observation loss, published settings
            script = script.replace(f"{HERE}/train_ppo_alf.py", f"{HERE}/train_ppo_alf_paw.py", 1) + f" --paw-coef {args.paw_coef}"
        if args.arm == 'ppo160ocsd':                   # OCSD: A_t = A (1 + beta sign(A) tanh((full - ablated) / 2))
            script = script.replace(f"{HERE}/train_ppo_alf.py", f"{HERE}/train_ppo_alf_ocsd.py", 1) + f" --ocsd-beta {args.ocsd_beta}"
        if args.arm == 'ppo160mv':                     # standard majority vote: fixed weight 1, unique plurality only
            script += " --distill-nu 1 --distill-label sc --distill-const-w 1.0 --sc-strict"
        PPOX = {'sopo160': ' --ratio turn_geo --ctn-delta 1 --eps-low 0.2 --eps-high 0.2 --recompute-old',      # SO-PPO (published clip)
                'sopo160v2': f' --ratio turn_geo --ctn-delta 1 --eps-low {args.ppox_eps} --eps-high {args.ppox_eps} --recompute-old',   # SO-PPO, --ppox-eps
                'steppo160v2': f' --ratio turn_geo --eps-low {args.ppox_eps} --eps-high {args.ppox_eps} --recompute-old'}   # ST-PPO, --ppox-eps
        if args.arm in PPOX:
            script = script.replace(f"{HERE}/train_ppo_alf.py", f"{HERE}/train_ppo_alf_ppox.py", 1) + PPOX[args.arm]
        if args.arm in ('wmppo', 'qppo'):
            it = int(os.path.basename(it_dir)[2:])
            hist = [os.path.join(args.out, f'it{j:03d}', 'calib_scored.json') for j in range(it - args.calib_window, it)]
            hist = [h for h in hist if os.path.exists(h)]
            if it <= args.calib_lcb_until:            # the gate rule of iterations 13-14
                script += " --calib-rule lcb"
            if args.gate_fixed:                       # gate ablation
                script += f" --gate-fixed {args.gate_fixed}"
            if args.critic_cons_weight > 0:           # value channel
                script += f" --cons-weight {args.critic_cons_weight} --cons-mode {args.critic_cons_mode}"
            script += (f" --distill-nu {args.distill_nu} --calib-min {args.calib_min} --calib-out {os.path.join(it_dir, 'calib_scored.json')}"
                       + (f" --calib-history {','.join(hist)}" if hist else ""))
    else:
        script = (f"{HERE}/train_lora.py --data {data} --base {args.model} --adapter-in {st['adapter'] or 'none'} "
                  f"--adapter-out {out} --lr {args.lr} --mb {args.mb} --stats-out {stats}")
        if args.arm in GRPO_FAMILY:                    # the same packed policy loop as the PPO arms
            script += " --pack-tokens 16384"
    name = f"sw_train_{args.arm}" + ("_" + os.path.basename(args.out.rstrip("/")) if args.arm in ("ppo160opd", "ppo160ocsd") or args.seed != 0 else "")   # runs of one arm may share a machine
    cmd = (f"docker rm -f {name} >/dev/null 2>&1; docker run --rm --name {name} --gpus '\"device={args.train_gpus}\"' "
           f"--net=host --ipc=host --shm-size=32g {paths.docker_mounts()} -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 "
           f"{IMAGE} bash -c '{args.trainer_venv}/bin/python -m torch.distributed.run --nproc_per_node "
           f"{len(args.train_gpus.split(','))} --master_port {args.master_port} {script}'")
    with open(os.path.join(it_dir, 'train.log'), 'w') as log:
        rc = subprocess.call(['bash', '-c', cmd],
                             stdout=log, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(stats):
        raise RuntimeError(f'training failed rc={rc}; see {it_dir}/train.log')
    return out, json.load(open(stats))


def wm_train_start(args, it_dir, st):
    """Online world model: fine-tune the current world model on this iteration's executed transitions, concurrently with
    the policy update (train_wm.py: 1 epoch, --wm-lr, 32 records per step, continuing from the current adapter)."""
    out = os.path.join(it_dir, 'wm')
    name = f"sw_wmtrain_{os.path.basename(args.out.rstrip('/'))}"
    script = (f"{HERE}/train_wm.py --data {os.path.join(it_dir, 'wm_trans.jsonl')} --out {out} --adapter-in {st['wm_adapter']} "
              f"--lr {args.wm_lr} --epochs 1 --base {args.model}")
    cmd = (f"docker rm -f {name} >/dev/null 2>&1; docker run --rm --name {name} --gpus '\"device={args.wm_train_gpus}\"' "
           f"--net=host --ipc=host --shm-size=32g {paths.docker_mounts()} -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 "
           f"{IMAGE} bash -c '{args.trainer_venv}/bin/python -m torch.distributed.run --nproc_per_node "
           f"{len(args.wm_train_gpus.split(','))} --master_port {args.wm_master_port} {script}'")
    log = open(os.path.join(it_dir, 'wm_train.log'), 'w')
    return subprocess.Popen(['bash', '-c', cmd],
                            stdout=log, stderr=subprocess.STDOUT), out


def wm_train_finish(args, it, proc_out, st):
    proc, out = proc_out
    rc = proc.wait()
    if rc != 0 or not os.path.exists(os.path.join(out, 'adapter_model.safetensors')):
        raise RuntimeError(f'online WM training failed rc={rc}; see {out}/../wm_train.log')
    name = f"wm_{os.path.basename(args.out.rstrip('/'))}_it{it:03d}"
    old = st.get('wm_lora') if st.get('wm_lora') != args.wm_lora_init else None
    for u in args.wm_url.split(','):
        load_adapter(u, name, out, old=old)
    st['wm_adapter'], st['wm_lora'] = out, name
    args.wm_lora = name
    return json.load(open(os.path.join(out, 'train_wm_stats.json')))


def run(args):
    n_tasks, n_chain, n_branch = ARMS[args.arm]
    n_tasks = max(1, int(round(n_tasks * args.scale)))    # --scale 0.5: 64 tasks for the tree arms
    kind = 'grpo' if args.arm in GRPO_FAMILY else 'ppo'
    os.makedirs(args.out, exist_ok=True)
    stf = os.path.join(args.out, 'state.json')
    st = json.load(open(stf)) if os.path.exists(stf) else {'it': 0, 'adapter': None, 'lora': None, 'critic': None,
                                                            'cursor': 0, 'tokens': 0, 'episodes': 0}
    train_files = game_files('train')
    order = list(range(len(train_files))); random.Random(args.seed).shuffle(order)
    pool = Pool(args.workers)
    if st['lora']:
        load_adapter(args.sglang, st['lora'], st['adapter'])
    if getattr(args, 'wm_online', False):
        args.wm_lora_init = args.wm_lora
        if st.get('wm_lora'):
            for u in args.wm_url.split(','):
                load_adapter(u, st['wm_lora'], st['wm_adapter'])
            args.wm_lora = st['wm_lora']
        else:
            st['wm_adapter'], st['wm_lora'] = args.wm_adapter_init, args.wm_lora
    metf = open(os.path.join(args.out, 'metrics.jsonl'), 'a')
    while st['it'] < args.iters:
        t0 = time.time(); it = st['it'] + 1
        rng = random.Random(args.seed * 100003 + it)
        it_dir = os.path.join(args.out, f'it{it:03d}'); os.makedirs(it_dir, exist_ok=True)
        tasks = [train_files[order[(st['cursor'] + k) % len(order)]] for k in range(n_tasks)]
        st['cursor'] += n_tasks
        saved = os.path.join(it_dir, 'samples.jsonl' if kind == 'grpo' else 'tree.json')
        if os.path.exists(os.path.join(it_dir, 'rollout.json')) and os.path.exists(saved):
            # resume: this iteration's rollout (and possibly training) finished before a restart
            ro = json.load(open(os.path.join(it_dir, 'rollout.json')))
            if os.path.exists(os.path.join(it_dir, 'train_stats.json')) and os.path.isdir(os.path.join(it_dir, 'adapter')):
                adapter, tstats = os.path.join(it_dir, 'adapter'), json.load(open(os.path.join(it_dir, 'train_stats.json')))
            else:
                adapter, tstats = train(args, it_dir, st, kind, saved)
            name = f'sw_{args.arm}_it{it:03d}'
            load_adapter(args.sglang, name, adapter, old=st['lora'])
            st['adapter'], st['lora'] = adapter, name
            if kind == 'ppo':
                st['critic'] = os.path.join(it_dir, 'critic')
            if getattr(args, 'wm_online', False):
                if not os.path.exists(os.path.join(it_dir, 'wm', 'adapter_model.safetensors')):
                    wm_train_finish(args, it, wm_train_start(args, it_dir, st), st)
                else:
                    wm_train_finish(args, it, (subprocess.Popen(['true']), os.path.join(it_dir, 'wm')), st)
            st['tokens'] += ro['tokens']; st['episodes'] += ro['episodes']; st['it'] = it
            rec = dict(ro['rec'], train=tstats, resumed=True)
            if args.eval_every > 0 and it % args.eval_every == 0:
                ev = evaluate(args, pool, st['lora'])
                json.dump(ev, open(os.path.join(args.out, f'eval_{it:03d}.json'), 'w'))
                rec['eval'] = {k: ev[k] for k in ('valid_seen', 'valid_unseen')}
            metf.write(json.dumps(rec) + '\n'); metf.flush(); json.dump(st, open(stf, 'w'))
            print(f"[{time.strftime('%H:%M:%S')}] {args.arm} it {it} resumed", flush=True)
            continue
        wm_arm = args.arm in ('wmppo',)
        k_arm = args.arm in ('wmppo', 'qppo', 'ppo160mv')
        cjobs = [job(args, gf, tag=(i, j), lora=st['lora'], K=args.K if k_arm else 1, wm=wm_arm, opd=args.arm == 'ppo160opd', ocsd=args.arm == 'ppo160ocsd', paw=args.arm == 'ppo160paw')
                 for i, gf in enumerate(tasks) for j in range(n_chain)]
        for attempt in range(3):
            res = pool.run(cjobs)
            chains = [c for c in res if 'error' not in c]
            if len(chains) >= len(cjobs) // 2:
                break
            errs = [c for c in res if 'error' in c]
            print(f"[{time.strftime('%H:%M:%S')}] {args.arm} it {it}: {len(errs)}/{len(res)} episodes failed "
                  f"(first: {errs[0]['error'][:200]}); rebuilding the pool and retrying", flush=True)
            pool.close(); time.sleep(30); pool = Pool(args.workers)
        if len(chains) < len(cjobs) // 2:
            raise RuntimeError('rollouts keep failing')
        by_task = {}
        for c in chains: by_task.setdefault(c['tag'][0], []).append(c)
        branches = []
        if n_branch:
            bj = []
            for i, eps in by_task.items():
                cand = [(j, t) for j, e in enumerate(eps) for t in range(len(e['actions']))]
                for b in range(n_branch):
                    if not cand: break
                    j, t = cand[rng.randrange(len(cand))]
                    bj.append(job(args, eps[j]['gamefile'], prefix=eps[j]['actions'][:t], tag=(i, 'b', b, eps[j]['actions'][t], j),
                                  lora=st['lora'], avoid=eps[j]['actions'][t], wm=wm_arm))
            branches = [r for r in pool.run(bj) if 'error' not in r]
            for r in branches: by_task.setdefault(r['tag'][0], []).append(r)
        t_roll = time.time() - t0
        wm_job = None
        if getattr(args, 'wm_online', False):
            with open(os.path.join(it_dir, 'wm_trans.jsonl'), 'w') as f:
                for e in chains + branches:
                    for x in e.get('wm_trans', []):
                        f.write(json.dumps(x) + '\n')
            wm_job = wm_train_start(args, it_dir, st)
        # ---- training data
        if kind == 'grpo':
            data = os.path.join(it_dir, 'samples.jsonl'); n_train = 0; mixed = 0
            with open(data, 'w') as f:
                for i, eps in by_task.items():
                    if args.arm in GRPO_FAMILY:                # per-turn advantages ported from verl-agent
                        import grpo_family
                        At = grpo_family.advantages(GRPO_FAMILY[args.arm], eps)
                        mixed += int(any(abs(x) > 0 for row in At for x in row))
                        for e, row in zip(eps, At):
                            for tu, a in zip(e['turns'], row):
                                if a == 0: continue
                                f.write(json.dumps({'prompt_ids': tu['prompt_ids'], 'resp_ids': tu['resp_ids'],
                                                    'old_lp': tu['lps'], 'adv': float(a)}) + '\n'); n_train += 1
                        continue
        else:
            nodes, edges, groups = [], [], []
            mixed = 0
            for i, eps in by_task.items():
                build_tree(eps, nodes, edges, groups)
                rs = [e['reward'] for e in eps]; mixed += int(0 < sum(rs) < len(rs))
            for n in nodes: n['G'] = float(np.mean(n.pop('rs')))
            for e in edges: e['G'] = float(np.mean(e.pop('rs'))); e.pop('_ck', None)
            distill, calib = [], []
            if wm_arm:
                for c in chains:                       # decision points: chain steps with >= 2 distinct samples
                    for tu in c['turns']:
                        if tu.get('wm_child'):
                            distill.append({'prompt_ids': tu['prompt_ids'], 'samples': tu['samples'], 'child': tu['wm_child'],
                                            'term': tu.get('wm_term', {}),   # ScienceWorld: WM-predicted endings
                                            'R': float(c['reward'])})    # return of the executed episode
                for b in branches:                     # calibration pairs: chain action vs branch action at the branch point
                    cb = b['turns'][0].get('calib') if b['turns'] else None
                    if cb:
                        i, j = b['tag'][0], b['tag'][4]
                        calib.append({'child_c': cb['child_c'], 'child_b': cb['child_b'],
                                      'term_c': cb.get('term_c'), 'term_b': cb.get('term_b'),
                                      'G_c': float(by_task[i][j]['reward']), 'G_b': float(b['reward'])})
            if args.arm == 'qppo':                     # Q-PPO: candidates and calibration pairs as policy prompt + response
                for c in chains:
                    for tu in c['turns']:
                        acts = {}
                        for x in tu.get('samples', []):
                            acts.setdefault(x['action'], x['resp_ids'])
                        if len(acts) > 1:
                            distill.append({'prompt_ids': tu['prompt_ids'], 'samples': tu['samples'], 'R': float(c['reward']),
                                            'child': {a: tu['prompt_ids'] + r for a, r in acts.items()}})
                for b in branches:
                    if not b['turns']:
                        continue
                    i, j = b['tag'][0], b['tag'][4]; c = by_task[i][j]; tb = len(b['prefix'])
                    tu_c = next((tu for tu in c['turns'] if tu['t'] == tb), None)
                    if tu_c is not None and b['turns'][0]['action'] != tu_c['action']:
                        calib.append({'child_c': tu_c['prompt_ids'] + tu_c['resp_ids'], 'child_b': tu_c['prompt_ids'] + b['turns'][0]['resp_ids'],
                                      'G_c': float(c['reward']), 'G_b': float(b['reward'])})
            if args.arm == 'ppo160mv':
                for c in chains:
                    for tu in c['turns']:
                        if len({x['action'] for x in tu.get('samples', [])}) > 1:
                            distill.append({'prompt_ids': tu['prompt_ids'], 'samples': tu['samples']})
            data = os.path.join(it_dir, 'tree.json')
            json.dump({'nodes': nodes, 'edges': edges, 'groups': groups, 'distill': distill, 'calib': calib},
                      open(data, 'w'))
            n_train = len(edges)
        eps_all = chains + branches
        rec0 = {'it': it, 'n_distill_states': len(distill) if kind != 'grpo' else 0,
                'n_calib_pairs': len(calib) if kind != 'grpo' else 0, 'chain_success': float(np.mean([c['reward'] for c in chains])) if chains else None,
                'branch_success': float(np.mean([b['reward'] for b in branches])) if branches else None,
                'n_chains': len(chains), 'n_branches': len(branches), 'mixed_tasks': mixed, 'tasks': len(by_task),
                'n_train': n_train, 'wall_rollout_s': round(t_roll, 1),
                **({'opd_turns': sum(1 for c in chains for tu in c['turns'] if tu.get('opd_lps')),
                    'opd_errors': sum(1 for c in chains for tu in c['turns'] if tu.get('opd_err'))} if args.arm == 'ppo160opd' else {}),
                **({'ocsd_turns': sum(1 for c in chains for tu in c['turns'] if tu.get('ocsd')),
                    'ocsd_errors': sum(1 for c in chains for tu in c['turns'] if tu.get('ocsd_err'))} if args.arm == 'ppo160ocsd' else {})}
        json.dump({'tokens': sum(e['gen_tokens'] for e in eps_all), 'episodes': len(eps_all), 'rec': rec0},
                  open(os.path.join(it_dir, 'rollout.json'), 'w'))
        adapter, tstats = train(args, it_dir, st, kind, data)
        name = f'sw_{args.arm}_it{it:03d}'
        load_adapter(args.sglang, name, adapter, old=st['lora'])
        wm_stats = wm_train_finish(args, it, wm_job, st) if wm_job else None
        st['adapter'], st['lora'] = adapter, name
        if kind == 'ppo':
            st['critic'] = os.path.join(it_dir, 'critic')
        eps_all = chains + branches
        st['tokens'] += sum(e['gen_tokens'] for e in eps_all); st['episodes'] += len(eps_all); st['it'] = it
        rec = {'it': it, 'tokens': st['tokens'], 'episodes': st['episodes'],
               'chain_success': float(np.mean([c['reward'] for c in chains])) if chains else None,
               'branch_success': float(np.mean([b['reward'] for b in branches])) if branches else None,
               'n_chains': len(chains), 'n_branches': len(branches), 'mixed_tasks': mixed, 'tasks': len(by_task),
               'branch_changed_action': sum(1 for b in branches if len(b['actions']) > len(b['prefix'])
                                            and b['actions'][len(b['prefix'])] != b['tag'][3]),
               'n_train': n_train, 'train': tstats, 'wall_rollout_s': round(t_roll, 1), 'wall_s': round(time.time() - t0, 1),
               **({'wm_online': wm_stats, 'wm_lora': st['wm_lora']} if wm_stats else {})}
        if args.eval_every > 0 and it % args.eval_every == 0:
            ev = evaluate(args, pool, st['lora'])
            json.dump(ev, open(os.path.join(args.out, f'eval_{it:03d}.json'), 'w'))
            rec['eval'] = {k: ev[k] for k in ('valid_seen', 'valid_unseen')}
        metf.write(json.dumps(rec) + '\n'); metf.flush()
        json.dump(st, open(stf, 'w'))
        print(f"[{time.strftime('%H:%M:%S')}] {args.arm} it {it} chain_succ {rec['chain_success'] or 0:.3f} "
              f"mixed {mixed}/{len(by_task)} train {n_train} kl {tstats.get('kl', 0):+.4f} "
              f"{('eval ' + json.dumps(rec['eval'])) if 'eval' in rec else ''} {rec['wall_s']}s", flush=True)
    pool.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(ARMS), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--max-steps', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--critic-lr', type=float, default=5e-5)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--sib-weight', type=float, default=0.0)
    ap.add_argument('--K', type=int, default=4, help='wmppo / qppo / ppo160mv: samples per chain step (the first is executed)')
    ap.add_argument('--k-alt', type=int, default=4, help='branch: resample its first turn up to this many times for a different action')
    ap.add_argument('--wm-url', default='', help='wmppo: WM sglang server(s), comma-separated')
    ap.add_argument('--wm-lora', default='wm_obs')
    ap.add_argument('--wm-online', action='store_true', help='fine-tune the world model every iteration on its executed transitions')
    ap.add_argument('--wm-adapter-init', default='', help='adapter dir of the initial world model (the one loaded as --wm-lora)')
    ap.add_argument('--wm-train-gpus', default='5,6')
    ap.add_argument('--wm-lr', type=float, default=5e-5)
    ap.add_argument('--wm-master-port', type=int, default=29690)
    ap.add_argument('--distill-nu', type=float, default=1.0)
    ap.add_argument('--opd-coef', type=float, default=1.0, help='ppo160opd: weight of the per-token log-ratio (standard OPD reward)')
    ap.add_argument('--opd-url', default='', help="ppo160opd: sglang URL of the teacher")
    ap.add_argument('--opd-clip', type=float, default=5.0, help='ppo160opd: |teacher - sampler log-prob| clip per token')
    ap.add_argument('--ocsd-url', default='', help="ppo160ocsd: sglang URL of a stronger teacher ('' = the policy itself, theta_old)")
    ap.add_argument('--ocsd-beta', type=float, default=0.5, help='ppo160ocsd: modulation strength (OCSD default 0.5)')
    ap.add_argument('--ocsd-top', type=float, default=0.2, help='ppo160ocsd: top-NLL step ratio per trajectory (OCSD default 0.2)')
    ap.add_argument('--ocsd-k', type=int, default=2, help='ppo160ocsd: future horizon K (OCSD default 2)')
    ap.add_argument('--critic-cons-weight', type=float, default=0.0, help='value channel: critic loss weight beta (train_ppo_alf.py --cons-weight)')
    ap.add_argument('--critic-cons-mode', default='full', choices=['full', 'terminal'])
    ap.add_argument('--ppox-eps', type=float, default=0.2, help='sopo160v2 / steppo160v2: turn-ratio clip')
    ap.add_argument('--capo-eps', type=float, default=0.001, help='capo160v2: action-ratio clip (published value 0.001)')
    ap.add_argument('--paw-coef', type=float, default=1.0, help='ppo160paw: PaW loss coefficient (published value 1.0)')
    ap.add_argument('--calib-min', type=int, default=30, help='wmppo: minimum calibration pairs per margin bin')
    ap.add_argument('--calib-lcb-until', type=int, default=0, help='wmppo: use the lower-confidence-bound gate rule up to this iteration')
    ap.add_argument('--gate-fixed', default='', help='gate ablation: fixed weights of the three margin bins, e.g. 1,1,1 (no gate) or 0,0,1 (top bin only)')
    ap.add_argument('--calib-window', type=int, default=3, help='wmppo: calibrate on the branch pairs of this many previous iterations')
    ap.add_argument('--mb', type=int, default=64)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--scale', type=float, default=1.0, help='multiply the tasks per iteration (budget)')
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default=paths.MODEL)
    ap.add_argument('--sglang', required=True)
    ap.add_argument('--train-gpus', required=True)
    ap.add_argument('--master-port', type=int, required=True)
    ap.add_argument('--trainer-venv', default=paths.TRAINER_VENV)
    ap.add_argument('--eval-only', action='store_true', help='evaluate --eval-lora (or the base model) and exit')
    ap.add_argument('--eval-lora', default=None)
    ap.add_argument('--eval-adapter', default=None, help='adapter dir to load and evaluate (with --eval-only)')
    ap.add_argument('--eval-name', default='eval.json', help='output file name in --out (with --eval-only)')
    a = ap.parse_args()
    if a.eval_only:
        os.makedirs(a.out, exist_ok=True)
        if a.eval_adapter:                     # re-evaluate a saved adapter under a temporary name
            a.eval_lora = 'reeval_' + os.path.basename(os.path.dirname(a.eval_adapter.rstrip('/'))) + '_' + str(os.getpid())
            load_adapter(a.sglang, a.eval_lora, a.eval_adapter)
        p = Pool(a.workers); ev = evaluate(a, p, a.eval_lora); p.close()
        if a.eval_adapter:
            try: sg_post(a.sglang, '/unload_lora_adapter', {'lora_name': a.eval_lora})
            except Exception: pass
        json.dump(ev, open(os.path.join(a.out, a.eval_name), 'w'))
        print(json.dumps({k: ev[k] for k in ('valid_seen', 'valid_unseen')}), flush=True)
    else:
        run(a)
