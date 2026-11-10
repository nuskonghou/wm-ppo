"""World-model prompt and target text for ScienceWorld (same interface as alfworld/wm_prompt.py; no torch import).
The world model sees the last 8 actions without their observations. The target is the full next observation as env.py
prints it: the action feedback, the current room, the inventory and the '[score: S]' line, with ' [task completed]' /
' [task failed]' on termination, so that the world model can predict that an action (e.g. focusing on a wrong object)
ends the task."""

SYSTEM = "You are a world model of the ScienceWorld text environment."


def wm_user(r, hist_max=8):
    hist = " ".join(f"[Step {i + 1}: action '{a}']" for i, (_, a) in enumerate(r['hist'][-hist_max:]))
    adm = ", ".join(r['adm'])
    return (f"Task: {r['task']}\nRecent actions: {hist or '(none)'}\nCurrent observation:\n{r['obs']}\n"
            f"Objects the agent can refer to: [{adm}]\nAction taken now: {r['action']}\n\n"
            "Predict the environment's next observation exactly as the environment would print it (action feedback, "
            "[Current room], [Inventory] and [score] lines), inside <obs> </obs>.")


def wm_target(r):
    return f"<obs>{r['next_obs']}</obs>"
