"""Text-Sokoban training loop (orchestrator) for the comparison with ProAct's MC-PPO: the loop of
scienceworld/alf_final.py on the Sokoban environment of env.py (6x6 grid, one box, at most 15 steps, 0/1 reward), with
several policy servers per run (episodes routed by a crc32 hash), 1,024 response tokens and these arms (tasks per
iteration at --scale 1, chains per task, branches per task):
  ppo      128 x 1 chain; PPO with a learned critic, GAE (the shared warm-up, iterations 1-12)
  ppo160   PPO-160: 320 x 1 (160 at --scale 0.5)
  wmppo    WM-PPO: 128 x (1 chain + 3 branches) + K = 4 samples per chain step, as in scienceworld/alf_final.py
  mcppo    ProAct MC-PPO (released configuration mc_ppo_sokoban.yaml) on PPO-160's episodes: the state value is the
           Monte-Carlo critic V^MC(s) = mean discounted return of M = 1000 uniformly random rollouts of T = 15 steps
           from the cloned state (mc_value_weight 1, so the learned critic does not enter the advantage), turn-level
           TD(0) advantages with gamma 0.95 (train_ppo_alf_mc.py); the 0/1 task reward of every arm.
"""
import argparse, json, os, random, subprocess, sys, time, zlib
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from env import game_files, task_type, EVAL_SPLITS  # noqa: E402
from rollout import Pool  # noqa: E402
from wmppo import paths  # noqa: E402

IMAGE = paths.TRAINER_IMAGE
ARMS = {'ppo': (128, 1, 0), 'wmppo': (128, 1, 3), 'ppo160': (320, 1, 0),
        'mcppo': (320, 1, 0)}   # ProAct MC-PPO on PPO-160's episodes (train_ppo_alf_mc.py)


def sg_post(url, path, payload, timeout=600):
    r = requests.post(url + path, json=payload, timeout=timeout); r.raise_for_status()
    try: return r.json()
    except Exception: return r.text


def load_adapter(urls, name, path, old=None):
    """urls: one sglang base URL or a comma-separated list (several policy servers per arm)."""
    for url in urls.split(','):
        _load_adapter(url, name, path, old)


def _load_adapter(url, name, path, old=None):
    try:
        sg_post(url, '/load_lora_adapter', {'lora_name': name, 'lora_path': path})
    except requests.HTTPError:          # e.g. the same name is already loaded (after a restart): reload it
        try: sg_post(url, '/unload_lora_adapter', {'lora_name': name})
        except Exception: pass
        sg_post(url, '/load_lora_adapter', {'lora_name': name, 'lora_path': path})
    if old:
        try: sg_post(url, '/unload_lora_adapter', {'lora_name': old})
        except Exception as e: print('unload failed', e, flush=True)


def job(args, gf, prefix=(), tag=None, lora=None, temperature=None, avoid=None, K=1, wm=False, mc=False):
    pol = args.sglang.split(',')                # several policy servers: each episode goes to one of them
    j = {'gamefile': gf, 'prefix': list(prefix), 'url': pol[zlib.crc32(f'{gf}|{len(prefix)}|{tag}'.encode()) % len(pol)], 'lora': lora, 'avoid_first': avoid, 'k_alt': args.k_alt,
         'temperature': args.temperature if temperature is None else temperature,
         'max_new_tokens': args.max_new_tokens, 'max_steps': args.max_steps, 'model': args.model, 'tag': tag, 'K': K}
    if mc:                                          # mcppo: ProAct's Monte-Carlo critic value of every visited state
        j['mc'] = {'M': args.mc_m, 'T': args.mc_t, 'gamma': args.mc_gamma}
    if wm and getattr(args, 'wm_online', False):
        j['record_wm'] = True
    if wm and args.wm_url:
        urls = args.wm_url.split(',')
        j['wm_url'] = urls[hash((gf, len(prefix))) % len(urls)]; j['wm_lora'] = args.wm_lora
    return j


def evaluate(args, pool, lora, splits=EVAL_SPLITS):
    res = {}
    for sp in splits:
        eps = pool.run([job(args, gf, tag=sp, lora=lora, temperature=0.0) for gf in game_files(sp)])
        ok = [e for e in eps if 'error' not in e]
        res[sp] = float(np.mean([e['reward'] for e in ok])) if ok else float('nan')
        res[sp + '_errors'] = len(eps) - len(ok)
        res[sp + '_per_game'] = {e['gamefile']: e['reward'] for e in ok}      # '<split>/<seed>/level'
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
                nkey[key] = len(nodes); nodes.append({'prompt_ids': tu['prompt_ids'], 'rs': [], **({'mc': tu['mc']} if 'mc' in tu else {})})
            nodes[nkey[key]]['rs'].append(R)
            ek = (key, acts[t])
            if ek not in ekey:
                tu = gen.get(t)
                if tu is None:
                    break
                term = t == len(acts) - 1
                ekey[ek] = len(edges)
                edges.append({'node': nkey[key], 'child': None, 'term': term, 'R': R if term else 0.0, 'rs': [],
                              'resp': tu['resp_ids'], 'old': tu['lps'], '_ck': tuple(acts[:t + 1])})
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
        if args.arm in ('ppo', 'wmppo', 'ppo160', 'mcppo'):
            script += " --pack-tokens 8192"            # Sokoban rows are long (1,024 response tokens): smaller packs
        if args.arm == 'mcppo':                        # ProAct MC-PPO: TD(0) advantages on the Monte-Carlo critic values
            script = script.replace(f"{HERE}/train_ppo_alf.py", f"{HERE}/train_ppo_alf_mc.py", 1) + f" --mc-gamma {args.mc_gamma}"
        if args.arm in ('wmppo',):
            it = int(os.path.basename(it_dir)[2:])
            hist = [os.path.join(args.out, f'it{j:03d}', 'calib_scored.json') for j in range(it - args.calib_window, it)]
            hist = [h for h in hist if os.path.exists(h)]
            if it <= args.calib_lcb_until:            # the gate rule of iterations 13-14
                script += " --calib-rule lcb"
            if args.critic_cons_weight > 0:           # value channel
                script += f" --cons-weight {args.critic_cons_weight} --cons-mode {args.critic_cons_mode}"
            script += (f" --distill-nu {args.distill_nu} --calib-min {args.calib_min} --calib-out {os.path.join(it_dir, 'calib_scored.json')}"
                       + (f" --calib-history {','.join(hist)}" if hist else ""))
    name = f"sok_train_{args.arm}" + ("_" + os.path.basename(args.out.rstrip("/")) if args.seed != 0 else "")   # runs of one arm may share a machine
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
    name = f"sok_wmtrain_{os.path.basename(args.out.rstrip('/'))}"
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
    n_tasks = max(1, int(round(n_tasks * args.scale)))    # --scale 0.5: 64 tasks for the tree arm
    kind = 'ppo'
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
        saved = os.path.join(it_dir, 'tree.json')
        if os.path.exists(os.path.join(it_dir, 'rollout.json')) and os.path.exists(saved):
            # resume: this iteration's rollout (and possibly training) finished before a restart
            ro = json.load(open(os.path.join(it_dir, 'rollout.json')))
            if os.path.exists(os.path.join(it_dir, 'train_stats.json')) and os.path.isdir(os.path.join(it_dir, 'adapter')):
                adapter, tstats = os.path.join(it_dir, 'adapter'), json.load(open(os.path.join(it_dir, 'train_stats.json')))
            else:
                adapter, tstats = train(args, it_dir, st, kind, saved)
            name = f'sok_{args.arm}_it{it:03d}'
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
                rec['eval'] = {k: ev[k] for k in EVAL_SPLITS}
            metf.write(json.dumps(rec) + '\n'); metf.flush(); json.dump(st, open(stf, 'w'))
            print(f"[{time.strftime('%H:%M:%S')}] {args.arm} it {it} resumed", flush=True)
            continue
        wm_arm = args.arm in ('wmppo',)
        k_arm = args.arm in ('wmppo',)
        cjobs = [job(args, gf, tag=(i, j), lora=st['lora'], K=args.K if k_arm else 1, wm=wm_arm, mc=args.arm == 'mcppo')
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
        if kind == 'ppo':
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
                                            'term': tu.get('wm_term', {}),   # predicted endings (a solved grid)
                                            'R': float(c['reward'])})    # return of the executed episode
                for b in branches:                     # calibration pairs: chain action vs branch action at the branch point
                    cb = b['turns'][0].get('calib') if b['turns'] else None
                    if cb:
                        i, j = b['tag'][0], b['tag'][4]
                        calib.append({'child_c': cb['child_c'], 'child_b': cb['child_b'],
                                      'term_c': cb.get('term_c'), 'term_b': cb.get('term_b'),
                                      'G_c': float(by_task[i][j]['reward']), 'G_b': float(b['reward'])})
            data = os.path.join(it_dir, 'tree.json')
            json.dump({'nodes': nodes, 'edges': edges, 'groups': groups, 'distill': distill, 'calib': calib},
                      open(data, 'w'))
            n_train = len(edges)
        eps_all = chains + branches
        rec0 = {'it': it, 'n_distill_states': len(distill),
                'n_calib_pairs': len(calib), 'chain_success': float(np.mean([c['reward'] for c in chains])) if chains else None,
                'branch_success': float(np.mean([b['reward'] for b in branches])) if branches else None,
                'n_chains': len(chains), 'n_branches': len(branches), 'mixed_tasks': mixed, 'tasks': len(by_task),
                'n_train': n_train, 'wall_rollout_s': round(t_roll, 1)}
        json.dump({'tokens': sum(e['gen_tokens'] for e in eps_all), 'episodes': len(eps_all), 'rec': rec0},
                  open(os.path.join(it_dir, 'rollout.json'), 'w'))
        adapter, tstats = train(args, it_dir, st, kind, data)
        name = f'sok_{args.arm}_it{it:03d}'
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
            rec['eval'] = {k: ev[k] for k in EVAL_SPLITS}
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
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--max-steps', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--critic-lr', type=float, default=5e-5)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--sib-weight', type=float, default=0.0)
    ap.add_argument('--K', type=int, default=4, help='wmppo: samples per chain step (the first is executed)')
    ap.add_argument('--k-alt', type=int, default=4, help='branch: resample its first turn up to this many times for a different action')
    ap.add_argument('--wm-url', default='', help='wmppo: WM sglang server(s), comma-separated')
    ap.add_argument('--wm-lora', default='wm_obs')
    ap.add_argument('--wm-online', action='store_true', help='fine-tune the world model every iteration on its executed transitions')
    ap.add_argument('--wm-adapter-init', default='', help='adapter dir of the initial world model (the one loaded as --wm-lora)')
    ap.add_argument('--wm-train-gpus', default='5,6')
    ap.add_argument('--wm-lr', type=float, default=5e-5)
    ap.add_argument('--wm-master-port', type=int, default=29690)
    ap.add_argument('--distill-nu', type=float, default=1.0)
    ap.add_argument('--critic-cons-weight', type=float, default=0.0, help='value channel: critic loss weight beta (train_ppo_alf.py --cons-weight)')
    ap.add_argument('--critic-cons-mode', default='full', choices=['full', 'terminal'])
    ap.add_argument('--mc-m', type=int, default=1000, help='mcppo: random rollouts per state (ProAct rollout_time 1000)')
    ap.add_argument('--mc-t', type=int, default=15, help='mcppo: steps per rollout (ProAct max_steps = the episode limit)')
    ap.add_argument('--mc-gamma', type=float, default=0.95, help='mcppo: discount of rollouts and TD advantages (ProAct turn_gamma)')
    ap.add_argument('--calib-min', type=int, default=30, help='wmppo: minimum calibration pairs per margin bin')
    ap.add_argument('--calib-lcb-until', type=int, default=0, help='wmppo: use the lower-confidence-bound gate rule up to this iteration')
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
            try: [sg_post(u, '/unload_lora_adapter', {'lora_name': a.eval_lora}) for u in a.sglang.split(',')]
            except Exception: pass
        json.dump(ev, open(os.path.join(a.out, a.eval_name), 'w'))
        print(json.dumps({k: ev[k] for k in EVAL_SPLITS}), flush=True)
    else:
        run(a)
