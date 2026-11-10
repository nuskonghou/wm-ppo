"""ScienceWorld (scienceworld 1.2.3, simplification 'easy') behind the AlfEnv interface of alfworld/env.py, so the
same pipeline (alf_final.py, rollout.py, ct_infer.py) runs unchanged. Protocol:
  - tasks SW-18 (the 18 tasks whose shortest solutions fit in 40 steps); a "gamefile" is '<task>|<variation>'
  - train = the official train variations in a task-uniform list (each task cycles its variations to 120 entries);
    valid_seen = dev-140, valid_unseen = test-144 (7 small tasks: all variations; 11 large tasks: 9 evenly spaced)
  - reward 1 iff the simulator reports completion with score 100 within max_steps agent actions; the -100 failure,
    other endings and truncation give 0
  - every agent action counts as a step (invalid, look, inventory); 'reset task' is disabled; ambiguity prompts are
    answered '0' automatically (same agent step)
  - the observation after an action = the action feedback, the current room (look, object lines sorted: 1.2.3 lists
    objects in a non-deterministic order), the inventory and '[score: S]'; when the simulator ends the task,
    '[task completed]' (score 100) or '[task failed]' (e.g. -100 after focusing on a wrong object); none on truncation
  - adm (shown in the prompt) = the objects the agent can refer to (get_possible_objects, sorted); the action templates
    are fixed text in prompts.py
One JVM per environment (heap capped by JAVA_TOOL_OPTIONS), killed by close()."""
import os
import re

os.environ.setdefault('JAVA_TOOL_OPTIONS', '-Xmx1g -Xss256k -XX:+UseSerialGC')
SIMPLIFICATION = 'easy'
SW18 = ['use-thermometer', 'measure-melting-point-known-substance', 'power-component',
        'power-component-renewable-vs-nonrenewable-energy', 'test-conductivity', 'test-conductivity-of-unknown-substances',
        'find-living-thing', 'find-non-living-thing', 'find-plant', 'find-animal', 'chemistry-mix',
        'chemistry-mix-paint-secondary-color', 'chemistry-mix-paint-tertiary-color', 'lifespan-longest-lived',
        'lifespan-shortest-lived', 'lifespan-longest-lived-then-shortest-lived', 'identify-life-stages-1',
        'identify-life-stages-2']
FORCED_CHOICE = {'power-component-renewable-vs-nonrenewable-energy', 'test-conductivity-of-unknown-substances',
                 'lifespan-longest-lived', 'lifespan-shortest-lived', 'lifespan-longest-lived-then-shortest-lived',
                 'identify-life-stages-1', 'identify-life-stages-2'}
SMALL = {'power-component', 'power-component-renewable-vs-nonrenewable-energy', 'chemistry-mix',
         'chemistry-mix-paint-secondary-color', 'chemistry-mix-paint-tertiary-color', 'identify-life-stages-1',
         'identify-life-stages-2'}
TRAIN_PER_TASK = 120
_SPLITS = None


def _splits():
    global _SPLITS
    if _SPLITS is None:
        from scienceworld import ScienceWorldEnv
        e = ScienceWorldEnv("", envStepLimit=10)
        out = {'train': {}, 'dev': {}, 'test': {}}
        for t in SW18:
            e.load(t, 0, SIMPLIFICATION)
            out['train'][t], out['dev'][t], out['test'][t] = e.get_variations_train(), e.get_variations_dev(), e.get_variations_test()
        e.close()
        _SPLITS = out
    return _SPLITS


def _spaced(vs, k=9):
    if len(vs) <= k:
        return list(vs)
    return [vs[round(i * (len(vs) - 1) / (k - 1))] for i in range(k)]


def game_files(split):
    s = _splits()
    if split == 'train':
        out = []
        for i in range(TRAIN_PER_TASK):
            for t in SW18:
                v = s['train'][t]
                out.append(f'{t}|{v[i % len(v)]}')
        return out
    part = {'valid_seen': 'dev', 'valid_unseen': 'test'}[split]
    return [f'{t}|{v}' for t in SW18 for v in (s[part][t] if t in SMALL else _spaced(s[part][t]))]


def task_type(gamefile):
    return gamefile.split('|')[0]


def sort_lines(text):
    """Sort the tab-indented object lines inside each block of a look / inventory text."""
    out, block = [], []
    for line in text.split('\n'):
        if line.startswith('\t'):
            block.append(line)
        else:
            out += sorted(block); block = []
            out.append(line)
    out += sorted(block)
    return '\n'.join(out).strip()


class AlfEnv:
    def __init__(self, gamefile, max_steps=40):
        from scienceworld import ScienceWorldEnv
        self.gamefile = gamefile
        self.task, var = gamefile.split('|')
        self.var = int(var)
        self.max_steps = max_steps
        self.env = ScienceWorldEnv("", envStepLimit=1000)
        self.env.load(self.task, self.var, SIMPLIFICATION)

    def _view(self, feedback, info, mark=''):
        fb = sort_lines(feedback.strip())
        if fb == sort_lines(info['look']):          # 'look around' (the room is printed below anyway)
            feedback = 'You look around.'
        elif fb == sort_lines(info['inv']):         # 'inventory'
            feedback = 'You check your inventory.'
        v = (f"{feedback.strip()}\n[Current room]\n{sort_lines(info['look'])}\n[Inventory]\n{sort_lines(info['inv'])}\n"
             f"[score: {info['score']}]")
        return v + (f' [{mark}]' if mark else '')

    def _adm(self):
        return sorted(set(self.env.get_possible_objects()))

    def reset(self):
        obs, info = self.env.reset()
        self.t, self.done = 0, False
        desc = re.sub(r'^(Task Description:)?\s*(Your task is to)?\s*', '', info['taskDesc'].strip())
        return f"Your task is to: {desc}\n{self._view('You are starting the task.', info)}", self._adm()

    def step(self, action):
        assert not self.done, 'step() after done'
        a = re.sub(r'\s+', ' ', action).strip()
        self.t += 1
        if a.lower().startswith('reset'):           # disabled: sent as an unparseable command (no state change)
            a = 'disabled action'
        obs, _, done_env, info = self.env.step(a)
        k = 0
        while obs.strip().lower().startswith('ambiguous request') and k < 3:
            obs, _, done_env, info = self.env.step('0'); k += 1
        sc = info['score']
        won = bool(done_env and sc >= 100)
        failed = sc < 0 or (done_env and not won)
        done = won or failed or self.t >= self.max_steps     # truncation: no marker (not predictable from the state)
        self.done = done
        mark = 'task completed' if won else ('task failed' if failed else '')
        return self._view(obs, info, mark), self._adm(), float(won), done

    def replay(self, actions):
        obs, adm = self.reset()
        r, done = 0.0, False
        for a in actions:
            obs, adm, r, done = self.step(a)
            if done:
                break
        return obs, adm, r, done

    def close(self):
        """ScienceWorldEnv.close() only asks the JVM to exit (it can linger): kill it."""
        p = getattr(getattr(self.env, '_gateway', None), 'java_process', None)
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
        if p is not None and p.poll() is None:
            try:
                p.kill(); p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
