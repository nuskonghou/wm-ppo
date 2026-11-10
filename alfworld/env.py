"""Single-game ALFWorld (TextWorld, tw-pddl) environment with deterministic prefix replay. The data directory is
paths.ALFWORLD_DATA (environment variable ALFWORLD_DATA): the output of alfworld-download, which holds json_2.1.1/."""
import glob, os, re, sys
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmppo import paths  # noqa: E402  (before alfworld, whose import sets ALFWORLD_DATA to its own default)
import textworld  # noqa: E402
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos  # noqa: E402

# fast_downward.load_lib() copies libdownward.so (~35 MB) into a fresh temp dir for every environment and keeps the
# deleted copy mapped, so a long-lived process leaks one copy per game until it exits and fills the temp directory.
# Caching is only safe when a process has ONE environment alive at a time (training rollouts); ct_infer.py keeps
# sandbox envs alive next to the main one, so it sets ALF_SHARED_DOWNWARD=0 and relies on a large TMPDIR instead.
if os.environ.get('ALF_SHARED_DOWNWARD', '1') == '1':
    import fast_downward
    if not getattr(fast_downward.load_lib, '_alf_cached', False):
        _cached = functools.lru_cache(maxsize=1)(fast_downward.load_lib)
        _cached._alf_cached = True
        fast_downward.load_lib = _cached

DATA = paths.ALFWORLD_DATA
TASK_TYPES = ['pick_and_place_simple', 'look_at_obj_in_light', 'pick_clean_then_place_in_recep',
              'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_two_obj_and_place']


def game_files(split):
    """split in {'train', 'valid_seen', 'valid_unseen'}; only games with a solvable tw-pddl file."""
    files = sorted(glob.glob(os.path.join(DATA, 'json_2.1.1', split, '*', '*', 'game.tw-pddl')))
    return files


def task_type(gamefile):
    name = gamefile.split('/')[-3]
    for t in TASK_TYPES:
        if name.startswith(t):
            return t
    return 'other'


class AlfEnv:
    def __init__(self, gamefile, max_steps=40):
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=['gamefile'])
        self.gamefile = gamefile
        self.env = textworld.start(gamefile, request_infos=infos,
                                   wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        self.max_steps = max_steps

    def reset(self):
        st = self.env.reset()
        self.t = 0
        obs = st['feedback']
        obs = obs.split('\n\n', 1)[-1] if '-= Welcome to TextWorld' in obs else obs
        return obs.strip(), list(st['admissible_commands'])

    def step(self, action):
        st, score, done = self.env.step(action)
        self.t += 1
        won = bool(st['won'])
        done = done or won or self.t >= self.max_steps
        return st['feedback'].strip(), list(st['admissible_commands']), float(won), done

    def replay(self, actions):
        """Reset and replay a prefix of actions (deterministic env). Returns the state after the prefix."""
        obs, adm = self.reset()
        r, done = 0.0, False
        for a in actions:
            obs, adm, r, done = self.step(a)
            if done:
                break
        return obs, adm, r, done
