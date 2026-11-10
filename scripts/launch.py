#!/usr/bin/env python3
"""Launcher of the release: inference servers, training runs, the warm-up, the initial world model and the evaluation
protocol, on one machine with NVIDIA GPUs. It runs inside the environment image (docker/Dockerfile.env) with the Docker
socket mounted, through scripts/wmppo (see the README); every inference server and every trainer runs in its own
container of the trainer image (docker/Dockerfile.trainer).

  launch.py train  ENV METHOD [--run K]      train run K of configs/ENV/METHOD.yaml (forks from the warm-up;
                                             --iters N stops earlier, --evaluate evaluates the reported iterations)
  launch.py eval   ENV ADAPTER_DIR NAME      evaluation protocol: greedy + 3 temperature-1 runs on the evaluation set
  launch.py warmup ENV                       retrain the 12-iteration PPO warm-up (configs/ENV/warmup.yaml)
  launch.py init-wm ENV                      initial world model from the warm-up's executed transitions
  launch.py serve  NAME GPU PORT [--lora N=DIR ...]   one sglang server of the trainer image (base model + dynamic
                                             LoRA), optionally loading adapters under the names N
  launch.py run    ENV SCRIPT [ARGS ...]     run one environment-side script of ENV/ (e.g. ct_infer.py, wm_collect.py)
  launch.py gpu    ENV GPUS SCRIPT [ARGS ...]   run one GPU script of ENV/ (train_wm.py, train_lora.py, ct_score.py,
                                             ct_value_server.py) in the trainer image; --base is added if missing
  launch.py stop   [PREFIX]                  remove the containers started by this launcher (name prefix, default wmppo_)
  launch.py data   alfworld                  download the ALFWorld games into ALFWORLD_DATA (alfworld-download)

GPUs are taken in order from WMPPO_GPUS (default 0,1,2,3,4,5,6,7): policy servers, world-model servers, the teacher
server, the trainer, the world-model trainer. Environment variables: see wmppo/paths.py and README.md."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wmppo import paths  # noqa: E402

ENVS = {'alfworld': {'venv': '/opt/venv-alfworld', 'fork': 'fork_it12.py'},
        'scienceworld': {'venv': '/opt/venv-scienceworld', 'fork': 'fork_it12.py'},
        'sokoban': {'venv': '/opt/venv-alfworld', 'fork': 'fork_sok.py'}}
TEACHER_MODEL = paths.TEACHER_MODEL
PREFIX = 'wmppo_'


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


def load_config(env, method):
    import yaml
    p = os.path.join(ROOT, 'configs', env, f'{method}.yaml')
    if not os.path.exists(p):
        sys.exit(f'no config {p}; available: ' + ', '.join(sorted(f[:-5] for f in os.listdir(os.path.dirname(p)))))
    return yaml.safe_load(open(p))


def cli_args(args):
    out = []
    for k, v in args.items():
        if v is True:
            out.append(f'--{k}')
        elif v is not False and v is not None:
            out += [f'--{k}', str(v)]
    return out


def python_of(env):
    venv = os.environ.get('WMPPO_ENV_VENV', ENVS[env]['venv'])
    p = os.path.join(venv, 'bin', 'python')
    return p if os.path.exists(p) else sys.executable


def env_vars(env, tag):
    e = dict(os.environ, PYTHONUNBUFFERED='1', ALFWORLD_DATA=paths.ALFWORLD_DATA)
    tmp = os.path.join(os.environ.get('WMPPO_TMP', '/tmp'), f'wmppo_{tag}')   # ALFWorld's planner writes temporary files
    os.makedirs(tmp, exist_ok=True)
    e['TMPDIR'] = tmp
    return e


def gpus():
    return [g for g in os.environ.get('WMPPO_GPUS', '0,1,2,3,4,5,6,7').split(',') if g != '']


def http_ok(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def serve(name, gpu, port, tp=1, model=None, lora=True, context=8192, extra=''):
    """Start one sglang server (dp 1: dynamic LoRA loading requires dp_size == 1) and wait until it is healthy."""
    model = model or paths.MODEL
    subprocess.call(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = (f"docker run -d --name {name} --gpus '\"device={gpu}\"' --net=host --ipc=host --shm-size=32g "
           f"{paths.docker_mounts()} -e HF_HUB_OFFLINE=1 {paths.TRAINER_IMAGE} "
           f"python3 -m sglang.launch_server --model-path {model} --host 0.0.0.0 --port {port} --dp-size 1 --tp-size {tp} "
           f"--context-length {context} --mem-fraction-static 0.85 --disable-custom-all-reduce "
           + ("--trust-remote-code --enable-lora --max-lora-rank 64 --lora-target-modules all --max-loras-per-batch 4 "
              if lora else '') + extra)
    subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)
    for _ in range(180):
        if http_ok(f'http://127.0.0.1:{port}/health'):
            log(f'{name} up on :{port} (GPU {gpu})')
            return f'http://127.0.0.1:{port}'
        time.sleep(10)
    subprocess.call(['docker', 'logs', '--tail', '40', name])
    sys.exit(f'{name} did not come up')


def stop(names):
    for n in names:
        subprocess.call(['docker', 'rm', '-f', n], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_lora(url, name, path):
    post(url + '/load_lora_adapter', {'lora_name': name, 'lora_path': path})


def trainer_segments(cfg, k, last):
    """[(last iteration, trainer GPUs), ...] of paper run k up to iteration last: the layout's trainer GPUs, or those of
    the reported run (run_trainer_gpus), which set the data-parallel split of each policy / critic minibatch."""
    v = (cfg.get('run_trainer_gpus') or {}).get(k, cfg.get('layout', {}).get('trainer_gpus', 2))
    out = []
    for it, n in ([(cfg['iters'], v)] if isinstance(v, int) else [tuple(x) for x in v]):
        out.append((min(it, last), n))
        if it >= last:
            break
    return out


def run_dir(env, method, k):
    return os.path.join(ROOT, env, 'runs', f'{method}_run{k}')


def start_servers(env, cfg, tag, port_base):
    """Policy, world-model and teacher servers of a config; returns (urls, gpu cursor, container names)."""
    g = gpus(); i = 0; names = []
    lay = cfg.get('layout', {})
    extra = cfg.get('sglang_extra', '')
    pol = []
    for j in range(lay.get('policy_servers', 1)):
        n = f'{PREFIX}{tag}_pol{j}'; names.append(n)
        pol.append(serve(n, g[i], port_base + 12 + j, extra=extra)); i += 1
    wm = []
    for j in range(lay.get('wm_servers', 0)):
        n = f'{PREFIX}{tag}_wm{j}'; names.append(n)
        wm.append(serve(n, g[i], port_base + 25 + 2 * j)); i += 1
    teacher = None
    if cfg.get('teacher'):
        n = f'{PREFIX}{tag}_teacher'; names.append(n)
        k = lay.get('teacher_server_gpus', 2)
        teacher = serve(n, ','.join(g[i:i + k]), port_base + 99, tp=k, model=TEACHER_MODEL, lora=False, context=16384)
        i += k
    return {'policy': pol, 'wm': wm, 'teacher': teacher}, i, names


def cmd_train(a):
    cfg = load_config(a.env, a.method)
    if a.method == 'warmup':
        sys.exit('use "launch.py warmup ENV" for the warm-up')
    k = a.run
    if k not in cfg['runs']:
        sys.exit(f'{a.method} has paper runs {sorted(cfg["runs"])}')
    seed = cfg['runs'][k]
    here = os.path.join(ROOT, a.env)
    out = os.path.realpath(a.out) if a.out else run_dir(a.env, a.method, k)
    tag = f'{a.env[:3]}_{a.method}_run{k}'.replace('.', '')
    urls, i, names = start_servers(a.env, cfg, tag, a.port_base)
    g = gpus(); lay = cfg.get('layout', {})
    segs = trainer_segments(cfg, k, a.iters or cfg['iters'])
    tr = g[i:i + max(n for _, n in segs)]; i += len(tr)
    wtr = g[i:i + lay.get('wm_trainer_gpus', 0)]; i += len(wtr)
    extra = []
    if cfg.get('world_model', 'none') != 'none':
        init = os.path.realpath(a.init_wm) if a.init_wm else os.path.join(here, 'runs', 'wm', 'warmup_log')
        for u in urls['wm']:
            load_lora(u, 'wm_init', init)
        extra += ['--wm-url', ','.join(urls['wm']), '--wm-lora', 'wm_init']
        if cfg['world_model'] == 'online':
            extra += ['--wm-adapter-init', init, '--wm-train-gpus', ','.join(wtr), '--wm-master-port', str(a.port_base + 2 + 1)]
    if cfg.get('teacher'):
        extra += [f'--{cfg["teacher"]}', urls['teacher']]
    py = python_of(a.env)
    if not os.path.exists(os.path.join(out, 'state.json')):
        subprocess.check_call([py, os.path.join(here, ENVS[a.env]['fork']), out, cfg['arm']])
    rc = 0
    try:
        for last, n in segs:           # one segment unless the reported run changed its trainer GPUs (it resumes)
            cmd = [py, '-u', os.path.join(here, 'alf_final.py'), '--arm', cfg['arm'], '--out', out, '--iters', str(last),
                   '--seed', str(seed), '--workers', str(cfg.get('workers', 48)), '--sglang', ','.join(urls['policy']),
                   '--train-gpus', ','.join(tr[:n]), '--master-port', str(a.port_base + 2)] + cli_args(cfg['args']) + extra
            log(' '.join(cmd))
            rc = subprocess.call(cmd, env=env_vars(a.env, tag), cwd=here)
            if rc != 0:
                break
    finally:
        if not a.keep_servers:
            stop(names)
    if rc == 0 and a.evaluate:
        for it in cfg.get('evaluate', []):
            ad = os.path.join(out, f'it{it:03d}', 'adapter')
            if os.path.isdir(ad):
                evaluate(a.env, ad, f'{a.method}_run{k}_it{it:03d}', a.eval_gpu or g[0], a.port_base + 41, a.n_sample_runs)
    sys.exit(rc)


def evaluate(env, adapter, name, gpu, port, n_runs=3, out=None):
    """Evaluation protocol: one dedicated TP1 server with only this adapter loaded; greedy evaluation of every game,
    then n_runs temperature-1 runs (ct_infer.py --rule random). Writes final_eval/greedy_<name>.json and
    final_eval/infer_<name>_random_f<r>.jsonl."""
    here = os.path.join(ROOT, env)
    out = os.path.realpath(out) if out else os.path.join(here, 'runs', 'final_eval')
    os.makedirs(out, exist_ok=True)
    srv = f'{PREFIX}{env[:3]}_eval_{gpu}'
    url = serve(srv, gpu, port)
    py = python_of(env); e = env_vars(env, f'eval_{name}')
    try:
        if not os.path.exists(os.path.join(out, f'greedy_{name}.json')):
            subprocess.check_call([py, '-u', 'alf_final.py', '--arm', 'ppo', '--eval-only', '--eval-adapter', adapter, '--out', out,
                                   '--eval-name', f'greedy_{name}.json', '--sglang', url, '--train-gpus', str(gpu),
                                   '--master-port', str(port + 1000), '--workers', '24'], env=e, cwd=here)
        lora = f'eval_{name}'
        load_lora(url, lora, adapter)
        for r in range(1, n_runs + 1):
            subprocess.check_call([py, '-u', 'ct_infer.py', '--arm', name, '--rule', 'random', '--lora', lora, '--sglang', url,
                                   '--tag', f'_f{r}', '--out', out, '--workers', '24'], env=e, cwd=here)
        post(url + '/unload_lora_adapter', {'lora_name': lora}, timeout=60)
    finally:
        stop([srv])
    log(f'evaluation of {name} written to {out}')


def cmd_eval(a):
    evaluate(a.env, os.path.realpath(a.adapter), a.name, a.gpu or gpus()[0], a.port, a.n_sample_runs, a.out)


def cmd_warmup(a):
    cfg = load_config(a.env, 'warmup')
    here = os.path.join(ROOT, a.env)
    out = os.path.realpath(a.out) if a.out else os.path.join(here, 'runs', 'ppo')
    tag = f'{a.env[:3]}_warmup'
    urls, i, names = start_servers(a.env, cfg, tag, a.port_base)
    g = gpus(); tr = g[i:i + cfg['layout'].get('trainer_gpus', 2)]
    py = python_of(a.env)
    try:
        for ph in cfg['phases']:
            args = dict(cfg['args'] or {}, **{k: v for k, v in ph.items() if k != 'iters'})
            cmd = [py, '-u', 'alf_final.py', '--arm', 'ppo', '--out', out, '--iters', str(ph['iters']), '--seed', '0',
                   '--eval-every', '3', '--workers', str(cfg.get('workers', 48)), '--sglang', ','.join(urls['policy']),
                   '--train-gpus', ','.join(tr), '--master-port', str(a.port_base + 2)] + cli_args(args)
            log(' '.join(cmd))
            subprocess.check_call(cmd, env=env_vars(a.env, tag), cwd=here)
    finally:
        stop(names)


def cmd_init_wm(a):
    """The initial world model: wm_from_log.py replays the warm-up's executed actions (iterations 1-12), then train_wm.py
    (1 epoch, lr 1e-4, 32 transitions per step) on 4 GPUs of the trainer image."""
    here = os.path.join(ROOT, a.env)
    warm = os.path.realpath(a.warmup) if a.warmup else (paths.SOKOBAN_WARMUP if a.env == 'sokoban' else os.path.join(here, 'runs', 'ppo'))
    data = os.path.join(here, 'runs', 'wm_data', 'warmup_log.jsonl'); os.makedirs(os.path.dirname(data), exist_ok=True)
    out = os.path.realpath(a.out) if a.out else os.path.join(here, 'runs', 'wm', 'warmup_log')
    e = env_vars(a.env, f'{a.env[:3]}_initwm')
    if a.env == 'alfworld':
        e['ALF_SHARED_DOWNWARD'] = '0'
    subprocess.check_call([python_of(a.env), '-u', 'wm_from_log.py', '--run', warm, '--its', '1-12', '--out', data,
                           '--workers', '40'], env=e, cwd=here)
    g = ','.join(gpus()[:4]); n_gpu = len(g.split(','))
    name = f'{PREFIX}{a.env[:3]}_wmtrain_init'
    cmd = (f"docker rm -f {name} >/dev/null 2>&1; docker run --rm --name {name} --gpus '\"device={g}\"' --net=host --ipc=host "
           f"--shm-size=32g {paths.docker_mounts()} -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 {paths.TRAINER_IMAGE} bash -c "
           f"'cd {here} && {paths.TRAINER_VENV}/bin/python -m torch.distributed.run --nproc_per_node {n_gpu} "
           f"--master_port {a.port_base + 9} train_wm.py --data {data} --out {out} --base {paths.MODEL}'")
    subprocess.check_call(cmd, shell=True)
    log('initial world model written to', out)


def cmd_serve(a):
    url = serve(a.name, a.gpu, a.port, a.tp, a.model, not a.no_lora)
    for x in a.lora:
        n, d = x.split('=', 1)
        load_lora(url, n, os.path.realpath(d))
        log(f'loaded LoRA {n} from {d}')


def cmd_run(a):
    here = os.path.join(ROOT, a.env)
    sys.exit(subprocess.call([python_of(a.env), '-u', a.script] + a.args, env=env_vars(a.env, f'{a.env[:3]}_run'), cwd=here))


def cmd_gpu(a):
    """One trainer-image script on the GPUs a.gpus (torchrun with one process per GPU for the trainers)."""
    here = os.path.join(ROOT, a.env)
    args = a.args + ([] if '--base' in a.args else ['--base', paths.MODEL])
    n = len(a.gpus.split(','))
    py = f'{paths.TRAINER_VENV}/bin/python'
    run = (f'{py} -u {a.script}' if a.script in ('ct_value_server.py', 'ct_score.py') else
           f'{py} -m torch.distributed.run --nproc_per_node {n} --master_port {a.master_port} {a.script}')
    name = a.name or f'{PREFIX}{a.env[:3]}_{a.script[:-3]}'
    subprocess.call(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = (f"docker run {'-d' if a.detach else '--rm'} --name {name} --gpus '\"device={a.gpus}\"' --net=host --ipc=host "
           f"--shm-size=32g {paths.docker_mounts()} -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 {paths.TRAINER_IMAGE} bash -c "
           f"'cd {here} && {run} " + ' '.join(args) + "'")
    log(cmd)
    sys.exit(subprocess.call(cmd, shell=True))


def cmd_data(a):
    """ALFWorld: alfworld-download (the json_2.1.1 games, their PDDL and TextWorld files). ScienceWorld ships its simulator
    in the pip package and the Sokoban levels are in sokoban/data/levels.json."""
    if a.env != 'alfworld':
        sys.exit('only ALFWorld needs a download')
    os.makedirs(paths.ALFWORLD_DATA, exist_ok=True)
    subprocess.check_call([os.path.join(os.path.dirname(python_of('alfworld')), 'alfworld-download'), '--data-dir', paths.ALFWORLD_DATA])
    n = sum(1 for _ in __import__('glob').glob(os.path.join(paths.ALFWORLD_DATA, 'json_2.1.1', '*', '*', '*', 'game.tw-pddl')))
    log(f'{n} games in {paths.ALFWORLD_DATA}')


def cmd_stop(a):
    names = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'], capture_output=True, text=True).stdout.split()
    stop([n for n in names if n.startswith(a.prefix)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); t.add_argument('env', choices=list(ENVS)); t.add_argument('method')
    t.add_argument('--run', type=int, default=1, help='paper run k (configs map it to --seed)')
    t.add_argument('--out', default=None); t.add_argument('--init-wm', default=None, help='initial world model (adapter dir)')
    t.add_argument('--port-base', type=int, default=30000); t.add_argument('--keep-servers', action='store_true')
    t.add_argument('--iters', type=int, default=None, help='stop after this iteration (default: the config, i.e. the paper)')
    t.add_argument('--evaluate', action='store_true', help='run the evaluation protocol for the reported iterations afterwards')
    t.add_argument('--eval-gpu', default=None); t.add_argument('--n-sample-runs', type=int, default=3)
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser('eval'); e.add_argument('env', choices=list(ENVS)); e.add_argument('adapter'); e.add_argument('name')
    e.add_argument('--gpu', default=None); e.add_argument('--port', type=int, default=30141)
    e.add_argument('--n-sample-runs', type=int, default=3); e.add_argument('--out', default=None)
    e.set_defaults(fn=cmd_eval)
    w = sub.add_parser('warmup'); w.add_argument('env', choices=list(ENVS)); w.add_argument('--out', default=None)
    w.add_argument('--port-base', type=int, default=30000); w.set_defaults(fn=cmd_warmup)
    i = sub.add_parser('init-wm'); i.add_argument('env', choices=list(ENVS)); i.add_argument('--warmup', default=None)
    i.add_argument('--out', default=None); i.add_argument('--port-base', type=int, default=30000); i.set_defaults(fn=cmd_init_wm)
    s = sub.add_parser('serve'); s.add_argument('name'); s.add_argument('gpu'); s.add_argument('port', type=int)
    s.add_argument('--tp', type=int, default=1); s.add_argument('--model', default=None); s.add_argument('--no-lora', action='store_true')
    s.add_argument('--lora', action='append', default=[], help='NAME=ADAPTER_DIR to load after start (repeatable)')
    s.set_defaults(fn=cmd_serve)
    r = sub.add_parser('run'); r.add_argument('env', choices=list(ENVS)); r.add_argument('script')
    r.add_argument('args', nargs=argparse.REMAINDER); r.set_defaults(fn=cmd_run)
    q = sub.add_parser('gpu'); q.add_argument('env', choices=list(ENVS)); q.add_argument('gpus'); q.add_argument('script')
    q.add_argument('--detach', action='store_true', help='keep running in the background (servers)')
    q.add_argument('--name', default=None); q.add_argument('--master-port', type=int, default=29600)
    q.add_argument('args', nargs=argparse.REMAINDER); q.set_defaults(fn=cmd_gpu)
    x = sub.add_parser('stop'); x.add_argument('prefix', nargs='?', default=PREFIX); x.set_defaults(fn=cmd_stop)
    d = sub.add_parser('data'); d.add_argument('env', choices=['alfworld']); d.set_defaults(fn=cmd_data)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
