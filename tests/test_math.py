"""Gate, tree GAE and group advantages on small hand-computed cases, and the gate formula against the gate weights
logged during the reported WM-PPO runs (results/gate.csv)."""
import csv
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stubs  # noqa: E402

stubs.install()
T = stubs.load('alfworld', 'train_ppo_alf')
G = stubs.load('alfworld', 'grpo_family')


def pairs(gains_per_bin):
    return [{'bin': b, 'gain': g} for b, gs in enumerate(gains_per_bin) for g in gs]


class Gate(unittest.TestCase):
    def test_shrinkage_weights(self):
        # bin 0: 20 pairs with gain 0.1 -> shrunken 0.1 * 20 / 60 = 0.0333 -> w = 0.667; bin 1: 40 pairs, gain 0.1 ->
        # 0.05 -> w = 1; bin 2: 19 pairs (< n_min = 20) -> 0
        w, g, se, n, lcb = T.calib_weights(pairs([[0.1] * 20, [0.1] * 40, [1.0] * 19]), min_n=20, prior_n=40.0, g_ref=0.05)
        self.assertEqual(n, [20, 40, 19])
        self.assertAlmostEqual(w[0], 2 / 3, places=9); self.assertEqual(w[1], 1.0); self.assertEqual(w[2], 0.0)

    def test_negative_and_capped(self):
        w, *_ = T.calib_weights(pairs([[-0.2] * 30, [1.0] * 200, []]), min_n=20)
        self.assertEqual(w, [0.0, 1.0, 0.0])

    def test_lcb_rule(self):
        # bin 0: gains 0 / 1 alternating (mean 0.5, population sd 0.5, n = 20): lcb = 0.5 - 1.64 * 0.5 / sqrt(20)
        xs = [0.0, 1.0] * 10
        w, g, se, n, lcb = T.calib_weights_lcb(pairs([xs, [0.0] * 20, [-1.0] * 25]), min_n=20)
        self.assertAlmostEqual(lcb[0], 0.5 - 1.64 * 0.5 / 20 ** 0.5, places=9)
        self.assertEqual(w, [1.0, 0.0, 0.0])
        w, *_ = T.calib_weights_lcb(pairs([[0.0] * 25, [0.0] * 25, [0.0] * 25]), min_n=20)
        self.assertEqual(w, [0.0, 0.0, 0.0])

    def test_logged_gate_weights(self):
        """From iteration 15 on, every logged bin weight equals the shrinkage rule applied to the logged pair counts and
        gains (rounded to 4 decimals in the logs, weights to 3)."""
        n_checked = 0
        with open(os.path.join(stubs.ROOT, 'results', 'gate.csv')) as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            if r['w'] in ('', 'None') or r['calib_hist_n'] in ('', 'None') or int(r['it']) < 15:
                continue
            w, n, g = json.loads(r['w']), json.loads(r['calib_hist_n']), json.loads(r['calib_hist_gain'])
            if r.get('method') in ('gate_ablation_no_gate', 'gate_ablation_top_bin'):
                continue
            pred = [min(1.0, max(0.0, gb * nb / (nb + 40.0) / 0.05)) if nb >= 20 else 0.0 for gb, nb in zip(g, n)]
            for a, b in zip(pred, w):
                self.assertAlmostEqual(a, b, delta=0.0015, msg=f"{r['env']} {r['run']} it {r['it']}")
            n_checked += 1
        self.assertGreater(n_checked, 900)


class TreeGAE(unittest.TestCase):
    def test_small_tree(self):
        # root node 0 -> edge 0 -> node 1 -> edges 1 (terminal, R = 1) and 2 (terminal, R = 0)
        nodes = [{'out': [0]}, {'out': [1, 2]}]
        edges = [{'node': 0, 'child': 1, 'term': False, 'R': 0.0}, {'node': 1, 'child': None, 'term': True, 'R': 1.0},
                 {'node': 1, 'child': None, 'term': True, 'R': 0.0}]
        V = [0.4, 0.5]
        A = T.gae(nodes, edges, V, lam=0.95)
        self.assertAlmostEqual(A[1], 0.5); self.assertAlmostEqual(A[2], -0.5)
        # edge 0: delta = V(1) - V(0) = 0.1, plus lam * mean of the child's edge advantages (0)
        self.assertAlmostEqual(A[0], 0.1)
        V = [0.4, 0.2]
        A = T.gae(nodes, edges, V, lam=0.5)
        self.assertAlmostEqual(A[0], (0.2 - 0.4) + 0.5 * ((0.8 + -0.2) / 2))


class GroupAdvantages(unittest.TestCase):
    def test_grpo(self):
        eps = [{'reward': 1, 'turns': [{'obs': 'a'}]}, {'reward': 0, 'turns': [{'obs': 'a'}]}]
        A = G.advantages('grpo', eps)
        # scores 10 and 0, mean 5, unbiased std 7.0711
        self.assertAlmostEqual(A[0][0], 5 / (50 ** 0.5 + 1e-6), places=6); self.assertAlmostEqual(A[1][0], -A[0][0], places=6)

    def test_tree_grpo_twice_z(self):
        A = G.tree_grpo_leaf_adv([1, 0, 0, 0])
        z = (1 - 0.25) / ((((0.75 ** 2) + 3 * 0.25 ** 2) / 3) ** 0.5 + 1e-6)
        self.assertAlmostEqual(A[0], 2 * z, places=6)

    def test_self_check_runs(self):
        eps = [{'reward': 1, 'turns': [{'obs': 'r'}, {'obs': 'a'}]}, {'reward': 0, 'turns': [{'obs': 'r'}, {'obs': 'b'}]}]
        for m in ('grpo', 'gigpo', 'hgpo'):
            A = G.advantages(m, eps)
            self.assertEqual([len(x) for x in A], [2, 2])


if __name__ == '__main__':
    unittest.main()
