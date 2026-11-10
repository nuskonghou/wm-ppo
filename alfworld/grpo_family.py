"""GRPO-family advantages for multi-turn rollouts, ported from verl-agent (github.com/langfengQ/verl-agent, commit
20bd331, Apache-2.0) so that every method runs in the same trainer (train_lora.py) with the same policy loss and only
the advantage differs:
  grpo      verl/trainer/ppo/core_algos.py compute_grpo_outcome_advantage: one outcome score per step, mean / unbiased
            std over all steps of the task group (verl-agent default compute_mean_std_cross_steps=True)
  gigpo     gigpo/core_gigpo.py compute_gigpo_outcome_advantage: the episode term above + step-level groups of identical
            anchor observations, discounted return gamma 0.95, step weight 1, mean_std_norm
  hgpo      recipe/hgpo/core_hgpo.py hgpo_advantage_estimate: hierarchical groups keyed by the last k observations,
            k = 1 .. history_length + 1 (history_length 2), biased std inside a group, aggregated with weights
            (k + 1)^alpha (alpha 1) over the non-zero levels, no episode term (base_group False)
Episode reward = 10 * won, verl-agent's ALFWorld scale (all three normalise by a std, so the scale only enters via eps).
Input: eps = the episodes of ONE task group, each {'reward', 'turns': [{'obs': current observation}, ...],
'final_obs': observation after the last action}. Output: per episode, the list of per-turn advantages.
tree_grpo_leaf_adv: Tree-GRPO's default tree estimator (re-implemented from AMAP-ML/Tree-GRPO)."""
from collections import defaultdict

import numpy as np

EPS = 1e-6
HP = {'gigpo': {'gamma': 0.95, 'step_w': 1.0}, 'hgpo': {'gamma': 0.95, 'history': 2, 'alpha': 1.0}}


def _R(e):
    return 10.0 * float(e['reward'] > 0)


def episode_adv(eps):
    """Outcome advantage with mean / unbiased std taken over steps (each step carries its episode's score)."""
    R = [_R(e) for e in eps]
    scores = [R[i] for i, e in enumerate(eps) for _ in e['turns']]
    if len(scores) == 1:
        m, s = 0.0, 1.0
    else:
        m, s = float(np.mean(scores)), float(np.std(scores, ddof=1))
    return [(r - m) / (s + EPS) for r in R]


def step_returns(e, gamma):
    """Discounted return of each turn; the only reward is the terminal one."""
    T = len(e['turns'])
    return [_R(e) * gamma ** (T - 1 - t) for t in range(T)]


def anchor_norm(eps, values):
    """GiGPO step_norm_reward: group turns of the task group by identical anchor observation; size-1 groups give 0."""
    groups = defaultdict(list)
    for i, e in enumerate(eps):
        for t, tu in enumerate(e['turns']):
            groups[tu['obs']].append((i, t))
    out = [[0.0] * len(e['turns']) for e in eps]
    for members in groups.values():
        v = [values[i][t] for i, t in members]
        if len(v) == 1:
            continue
        m, s = float(np.mean(v)), float(np.std(v, ddof=1))
        for (i, t), x in zip(members, v):
            out[i][t] = (x - m) / (s + EPS)
    return out


def adv_grpo(eps):
    A = episode_adv(eps)
    return [[A[i]] * len(e['turns']) for i, e in enumerate(eps)]


def adv_gigpo(eps, gamma=0.95, step_w=1.0):
    A = episode_adv(eps)
    S = anchor_norm(eps, [step_returns(e, gamma) for e in eps])
    return [[A[i] + step_w * S[i][t] for t in range(len(e['turns']))] for i, e in enumerate(eps)]


def adv_hgpo(eps, gamma=0.95, history=2, alpha=1.0):
    ret = [step_returns(e, gamma) for e in eps]
    obs = [[tu['obs'] for tu in e['turns']] for e in eps]
    clusters = defaultdict(list)
    for i in range(len(eps)):
        for t in range(len(obs[i])):
            for k in range(1, min(history + 1, t + 1) + 1):
                clusters[(k, tuple(obs[i][t - k + 1:t + 1]))].append((i, t))
    items = defaultdict(list)
    for (k, _), members in clusters.items():
        if len(members) <= 1:
            continue
        r = np.array([ret[i][t] for i, t in members])
        a = (r - r.mean()) / (r.std() + EPS)                # biased std, as in core_hgpo.py
        for (i, t), x in zip(members, a):
            items[(i, t)].append((k, float(x)))
    out = []
    for i in range(len(eps)):
        row = []
        for t in range(len(obs[i])):
            it = [(k, x) for k, x in items.get((i, t), []) if x != 0]
            if it:
                w = np.array([(k + 1) ** alpha for k, _ in it], float)
                w = w / (w.sum() + EPS)
                row.append(float(sum(wj * x for wj, (_, x) in zip(w, it))))
            else:
                row.append(0.0)
        out.append(row)
    return out


def _z(x):
    """Tree-GRPO's group z-score: sample std (N-1), eps 1e-6; a one-element group has mean 0 and std 1."""
    x = np.asarray(x, float)
    if len(x) == 1:
        return x.copy()
    return (x - x.mean()) / (x.std(ddof=1) + EPS)


def tree_grpo_leaf_adv(R):
    """Tree-GRPO's default estimator adv_estimator=tree (AMAP-ML/Tree-GRPO, commit 19bf3fa): A = z over the question's
    leaves + z over the tree's leaves, native 0/1 reward. Here one tree per task (1 chain + its branches), so both groups
    are the same leaves and A = 2 z, as published."""
    R = [float(r > 0) for r in R]
    return list(_z(R) + _z(R))


def advantages(method, eps):
    return {'grpo': adv_grpo, 'gigpo': adv_gigpo, 'hgpo': adv_hgpo}[method](eps, **HP.get(method, {}))


if __name__ == '__main__':                      # tiny self-check
    eps = [{'reward': 1, 'turns': [{'obs': 'r'}, {'obs': 'a'}, {'obs': 'b'}], 'final_obs': 'x'},
           {'reward': 0, 'turns': [{'obs': 'r'}, {'obs': 'a'}, {'obs': 'c'}], 'final_obs': 'y'},
           {'reward': 0, 'turns': [{'obs': 'r'}, {'obs': 'd'}], 'final_obs': 'z'}]
    for m in ('grpo', 'gigpo', 'hgpo'):
        print(m, [[round(x, 3) for x in row] for row in advantages(m, eps)])
