"""Stand-ins for the heavy dependencies (simulators, PyTorch, transformers, peft), so that the modules can be imported and
their pure-Python parts tested on a CPU machine without the environment or trainer images."""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def install():
    class Dummy:
        def __init__(self, *a, **k):
            pass
    _mod('textworld', EnvInfos=Dummy, start=lambda *a, **k: None)
    _mod('alfworld'); _mod('alfworld.agents'); _mod('alfworld.agents.environment')
    _mod('alfworld.agents.environment.alfred_tw_env', AlfredDemangler=Dummy, AlfredInfos=Dummy)
    _mod('fast_downward', load_lib=lambda *a, **k: None)
    _mod('scienceworld', ScienceWorldEnv=Dummy)
    torch = _mod('torch', Tensor=Dummy, no_grad=lambda: None, enable_grad=lambda: None)
    torch.distributed = _mod('torch.distributed')
    torch.nn = _mod('torch.nn', Linear=Dummy)
    _mod('transformers', AutoTokenizer=Dummy, AutoModelForCausalLM=Dummy)
    _mod('peft', LoraConfig=Dummy, PeftModel=Dummy, get_peft_model=lambda *a, **k: None)


def load(env, name):
    """Import <env>/<name>.py as a fresh module (each environment directory has its own copies of the shared names)."""
    import importlib.util
    d = os.path.join(ROOT, env)
    for m in ('env', 'rollout', 'prompts', 'wm_prompt', 'train_wm', 'grpo_family', 'sokoban_gen', 'ct_infer'):
        sys.modules.pop(m, None)
    sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(f'{env}_{name}', os.path.join(d, f'{name}.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(d)
