"""World-model prompt and target text (shared by train_wm.py, rollout.py, ct_infer.py and wm_eval.py; no torch import).
ALFWorld: the world model sees all previous observations and actions."""


SYSTEM = "You are a world model of the ALFRED text household environment."


def wm_user(r, hist_max=40):
    hist = " ".join(f"[Step {i + 1}: observation '{o}', action '{a}']" for i, (o, a) in enumerate(r['hist'][-hist_max:]))
    adm = ", ".join(f"'{a}'" for a in r['adm'] if a != 'help')
    return (f"Task: {r['task']}\nInitial observation: {r['room']}\nHistory of steps so far: {hist or '(none)'}\n"
            f"Current observation: {r['obs']}\nAdmissible actions: [{adm}]\nAction taken now: {r['action']}\n\n"
            "Predict the environment's next observation exactly as the environment would print it, inside <obs> </obs>.")


def wm_target(r):
    return f"<obs>{r['next_obs']}</obs>"
