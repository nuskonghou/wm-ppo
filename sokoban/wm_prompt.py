"""World-model prompt and target text for text Sokoban (same interface as alfworld/wm_prompt.py; no torch import).
Sokoban is fully observable, so the world model sees the current grid, the last few actions and the action."""


SYSTEM = "You are a world model of the text Sokoban environment."

LEGEND = ("Symbols: # wall, _ floor, O target, X box, √ box on target, P player, S player on target. "
          "Rows are listed from top to bottom.")


def wm_user(r, hist_max=2):
    hist = " ".join(f"[action '{a}']" for _, a in r['hist'][-hist_max:])
    return (f"{LEGEND}\nRecent actions: {hist or '(none)'}\nCurrent grid:\n{r['obs']}\nAction taken now: {r['action']}\n\n"
            "Predict the grid after this action exactly as the environment would print it, inside <obs> </obs>.")


def wm_target(r):
    return f"<obs>{r['next_obs']}</obs>"
