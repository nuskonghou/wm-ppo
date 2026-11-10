"""Prompt construction and action parsing for text Sokoban (the interface of alfworld/prompts.py): verl-agent's default
Sokoban (6x6, 1 box, 15 steps) with its step-by-step reasoning instruction and its invalid-action projection.

The rules/symbols text follows verl-agent's SOKOBAN_TEMPLATE (agent_system/environments/prompts/sokoban.py); the
history keeps the last HIST_LEN (grid, action) pairs, and the model reasons step by step inside <think> before <action>."""
import re

SYSTEM = "You are an expert agent operating in the Sokoban environment."

RULES = (
    "# Symbols and Their Meaning\n"
    "- Walls (`#`): These block movement. You can't move through or push anything into walls.\n"
    "- Floor (`_`): Open spaces where you can walk and move boxes.\n"
    "- Targets (`O`): The spots where boxes need to go.\n"
    "- Boxes (`X`): These are what you need to push onto the targets.\n"
    "- Player (`P`): That's you! You'll move around the grid to push boxes.\n"
    "- Box on Target (`√`): A box successfully placed on a target.\n"
    "- Player on Target (`S`): You standing on a target.\n\n"
    "# Your Goal\n"
    "Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!\n\n"
    "# Rules\n"
    "You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.\n"
    "You can't walk through or push boxes into walls (`#`).\n"
    "To avoid traps, do not push boxes into corners or against walls where they can't be moved again.\n"
    "Rows are listed from top to bottom: 'up' moves to the row above, 'down' to the row below.\n\n"
)

TAIL = (
    "Your admissible actions are [{adm}].\n\n"
    "Now it's your turn to make a move (choose ONE action only for the current step). You should first reason "
    "step-by-step about the current situation — observe the positions of boxes and targets, plan a path to push a "
    "box toward a target, and avoid traps like corners or walls. This reasoning process MUST be enclosed within "
    "<think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for current "
    "step and present it within <action> </action> tags."
)

TASK = "Push the box onto the target."
HIST_LEN = 2


def split_task(first_obs):
    """Sokoban has a fixed goal; the initial grid plays the role of ALFWorld's room description."""
    return TASK, first_obs


def user_message(task, room, history, obs, adm, hist_len=HIST_LEN):
    """history: list of (grid seen at step i, action taken at step i); obs: current grid."""
    adm_s = ", ".join(f'"{a}"' for a in adm)
    t = len(history)
    if t == 0:
        cur = f"# Current Step\nYour current observation is:\n{room}\n"
    else:
        recent = history[-hist_len:]
        start = t - len(recent)
        hist = "\n".join(f"[Observation {start + i + 1}:\n{o}\nAction {start + i + 1}: '{a}']" for i, (o, a) in enumerate(recent))
        cur = (f"# Current Step\nPrior to this step, you have already taken {t} step(s). Below are the most recent "
               f"{len(recent)} observations and the corresponding actions you took:\n{hist}\n"
               f"You are now at step {t + 1} and your current observation is:\n{obs}\n")
    return RULES + cur + TAIL.format(adm=adm_s)


def messages(task, room, history, obs, adm):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(task, room, history, obs, adm)}]


def parse_action(text, adm):
    """Return (action, valid). Takes the last <action>...</action>. As in verl-agent's Sokoban projection, an output
    without an action tag (e.g. truncated reasoning) or with anything but one of the four moves is the invalid action
    'none' (a no-op step), so long invalid text never enters the history or counts as a distinct candidate."""
    m = re.findall(r"<action>(.*?)</action>", text, flags=re.S)
    if not m:
        return 'none', False
    a = re.sub(r"\s+", " ", m[-1].strip().lower()).strip(" .'\"`")
    return (a, True) if a in adm else ('none', False)
