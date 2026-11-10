"""The paper's numbers from results/: tables (all compared cells) and the Section 2 diagnostics."""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Results(unittest.TestCase):
    def test_tables_match_the_paper(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'analysis', 'check_tables.py')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn('total: 981/981 cells match', r.stdout)

    def test_section2_numbers(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'analysis', 'diagnostics.py')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        for s in ('first sample                                                   54.4', 'majority vote                                                  56.9',
                  '61.7', '66.4', '70.4', '59.5', '65.3', '59.1', '+13.5 points (SE 2.8', '+9.5 points (SE 2.5',
                  '195 pairs, critic order correct 57%', 'lower half of the value gaps 53%, top quarter 71%', '11/12',
                  'state 0.008', 'before 57.1, world-model + critic labels 65.3, majority-vote labels 56.9',
                  '+8.2 points (SE 2.0; sign test p = 0.00027'):
            self.assertIn(s, out)


if __name__ == '__main__':
    unittest.main()
