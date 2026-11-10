"""Policy prompt for ScienceWorld (the interface of alfworld/prompts.py: SYSTEM, split_task, messages, parse_action).
Same layout as the ALFWorld (verl-agent) prompt: task, the most recent 5 steps, the current observation, then the
admissible space. ScienceWorld's full list of valid action/object combinations is too long for the prompt (median ~5.5k
tokens), so the prompt gives the action templates and the objects the agent can refer to (env.py adm)."""
import re

SYSTEM = "You are an expert agent operating in the ScienceWorld environment, a text-based simulator of elementary science experiments."
TEMPLATES = ['look around', 'look at OBJ', 'look in OBJ', 'inventory', 'task', 'teleport to LOC', 'go to LOC',
             'open OBJ', 'close OBJ', 'activate OBJ', 'deactivate OBJ', 'pick up OBJ', 'put down OBJ', 'move OBJ to OBJ',
             'pour OBJ in OBJ', 'dunk OBJ in OBJ', 'mix OBJ', 'use OBJ on OBJ', 'connect OBJ to OBJ', 'disconnect OBJ',
             'eat OBJ', 'flush OBJ', 'read OBJ', 'focus on OBJ', 'wait', 'wait1']
RULES = ("Rules: 'focus on OBJ' declares which object the task is about and cannot be undone; focusing on a wrong object "
         "ends the task as a failure, so explore first and focus only on the object the task asks about. OBJ must be an "
         "object you can refer to now (listed below); LOC is a room name (teleport moves you there directly). "
         "Answering questions such as 'which one ...' is done by focusing on the answer object.")
HIST_LEN = 5


def split_task(first_obs):
    """env.reset() returns 'Your task is to: <desc>\\n<initial observation>'."""
    head, _, rest = first_obs.partition('\n')
    return head.replace('Your task is to:', '').strip(), rest


def user_message(task, room, history, obs, adm):
    adm_s = ', '.join(adm)
    t = len(history)
    if t == 0:
        cur = f"Your current observation is:\n{room}\n"
    else:
        recent = history[-HIST_LEN:]
        start = t - len(recent)
        hist = '\n'.join(f"[Observation {start + i + 1}: '{o}', Action {start + i + 1}: '{a}']" for i, (o, a) in enumerate(recent))
        cur = (f"Prior to this step, you have already taken {t} step(s). Below are the most recent {len(recent)} "
               f"observations and the corresponding actions you took:\n{hist}\n"
               f"You are now at step {t + 1} and your current observation is:\n{obs}\n")
    return (f"Your task is to: {task}\n{RULES}\n{cur}"
            f"Action templates: [{', '.join(TEMPLATES)}].\nObjects you can refer to: [{adm_s}].\n\n"
            "Now it's your turn to take an action. First reason step by step (at most two sentences) inside "
            "<think> </think>, then give exactly one action inside <action> </action>.")


def messages(task, room, history, obs, adm):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(task, room, history, obs, adm)}]


def parse_action(text, adm):
    """Return (action, valid). The last <action>...</action>, whitespace-normalised and lower-cased; an output without
    an action tag is the invalid action 'none' (a step the simulator rejects), so long text never enters the history.
    Validity of a well-formed action is decided by the simulator (its feedback is 'No known action matches that input.')."""
    m = re.findall(r"<action>(.*?)</action>", text, flags=re.S)
    if not m:
        return 'none', False
    a = re.sub(r"\s+", " ", m[-1].strip().lower()).strip(" .'\"`")
    return (a, True) if a else ('none', False)
