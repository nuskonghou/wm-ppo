"""Prompt construction and action parsing for ALFWorld with a non-thinking chat model."""
import re

SYSTEM = "You are an expert agent operating in the ALFRED embodied household environment."

TEMPLATE_FIRST = (
    "Your task is to: {task}\n"
    "Your current observation is: {obs}\n"
    "Your admissible actions of the current situation are: [{adm}].\n\n"
    "Now it's your turn to take an action. First reason briefly (at most two sentences) about the current "
    "situation inside <think> </think> tags. Then choose one admissible action for the current step and "
    "present it inside <action> </action> tags."
)

TEMPLATE = (
    "Your task is to: {task}\n"
    "Prior to this step, you have already taken {t} step(s). Below are the most recent {k} observations "
    "and the corresponding actions you took: {hist}\n"
    "You are now at step {t1} and your current observation is: {obs}\n"
    "Your admissible actions of the current situation are: [{adm}].\n\n"
    "Now it's your turn to take an action. First reason briefly (at most two sentences) about the current "
    "situation inside <think> </think> tags. Then choose one admissible action for the current step and "
    "present it inside <action> </action> tags."
)

HIST_LEN = 5


def split_task(first_obs):
    """The first ALFWorld observation contains the room description and 'Your task is to: ...'."""
    m = re.search(r"Your task is to:\s*(.*)", first_obs)
    task = m.group(1).strip().rstrip('.') + '.' if m else ''
    room = first_obs[:m.start()].strip() if m else first_obs
    return task, room


def user_message(task, room, history, obs, adm, hist_len=HIST_LEN):
    """history: list of (observation seen at step i, action taken at step i); obs: current observation."""
    adm_s = ", ".join(f"'{a}'" for a in adm if a not in ('help',))
    t = len(history)
    if t == 0:
        return TEMPLATE_FIRST.format(task=task, obs=room, adm=adm_s)
    recent = history[-hist_len:]
    start = t - len(recent)
    hist = " ".join(f"[Observation {start + i + 1}: '{o}', Action {start + i + 1}: '{a}']"
                    for i, (o, a) in enumerate(recent))
    return TEMPLATE.format(task=task, t=t, k=len(recent), hist=hist, t1=t + 1, obs=obs, adm=adm_s)


def messages(task, room, history, obs, adm):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(task, room, history, obs, adm)}]


def parse_action(text, adm):
    """Return (action, valid). Takes the last <action>...</action>; falls back to the raw text."""
    m = re.findall(r"<action>(.*?)</action>", text, flags=re.S)
    a = (m[-1] if m else text).strip().lower()
    a = re.sub(r"\s+", " ", a).strip(" .'\"")
    return a, a in adm
