"""Compilation, imports (heavy dependencies stubbed), shared copies and configs."""
import ast
import filecmp
import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stubs  # noqa: E402

stubs.install()
ROOT = stubs.ROOT
ENVS = ('alfworld', 'scienceworld', 'sokoban')


class Compile(unittest.TestCase):
    def test_all_python_files_compile(self):
        files = [f for f in glob.glob(os.path.join(ROOT, '**', '*.py'), recursive=True) if '/third_party/' not in f]
        self.assertGreater(len(files), 50)
        for f in files:
            with self.subTest(f=os.path.relpath(f, ROOT)):
                with open(f) as fh:
                    compile(fh.read(), f, 'exec')


class Imports(unittest.TestCase):
    MODULES = {'alfworld': ['alf_final', 'rollout', 'env', 'prompts', 'wm_prompt', 'grpo_family', 'ct_infer', 'eval_traj',
                            'wm_from_log', 'wm_collect', 'wm_eval', 'offline_collect', 'train_ppo_alf', 'train_ppo_alf_capo',
                            'train_ppo_alf_ppox', 'train_ppo_alf_q', 'train_ppo_alf_opd', 'train_ppo_alf_ocsd', 'train_lora',
                            'train_wm', 'ct_score', 'ct_value_server'],
               'scienceworld': ['alf_final', 'rollout', 'env', 'prompts', 'wm_prompt', 'grpo_family', 'ct_infer', 'eval_traj',
                                'wm_from_log', 'train_ppo_alf', 'train_ppo_alf_capo', 'train_ppo_alf_ppox', 'train_ppo_alf_q',
                                'train_ppo_alf_opd', 'train_ppo_alf_ocsd', 'train_lora', 'train_wm'],
               'sokoban': ['alf_final', 'rollout', 'env', 'prompts', 'wm_prompt', 'ct_infer', 'wm_from_log', 'sokoban_gen',
                           'train_ppo_alf', 'train_ppo_alf_mc', 'train_wm']}

    def test_import(self):
        for env, mods in self.MODULES.items():
            for m in mods:
                with self.subTest(module=f'{env}/{m}'):
                    stubs.load(env, m)

    def test_paw_trainer_needs_the_paw_clone(self):
        repo = os.path.join(ROOT, 'third_party', 'Policy-WorldModel-Agent')
        if os.environ.get('PAW_REPO') or os.path.isdir(repo):
            for env in ('alfworld', 'scienceworld'):
                m = stubs.load(env, 'train_ppo_alf_paw')
                self.assertEqual(len(m.PREDICTION_PROMPTS), 20)
        else:
            with self.assertRaises(FileNotFoundError):
                stubs.load('alfworld', 'train_ppo_alf_paw')


class SharedCopies(unittest.TestCase):
    """Each environment directory is self-contained; these files are identical copies."""
    SAME = [('train_lora.py', ('alfworld', 'scienceworld')), ('grpo_family.py', ('alfworld', 'scienceworld')),
            ('train_ppo_alf_capo.py', ('alfworld', 'scienceworld')), ('train_ppo_alf_ppox.py', ('alfworld', 'scienceworld')),
            ('train_ppo_alf_q.py', ('alfworld', 'scienceworld')), ('train_wm.py', ENVS)]

    def test_identical(self):
        for f, envs in self.SAME:
            for e in envs[1:]:
                with self.subTest(f=f, env=e):
                    self.assertTrue(filecmp.cmp(os.path.join(ROOT, envs[0], f), os.path.join(ROOT, e, f), shallow=False))

    def test_gate_and_gae_identical_in_every_trainer(self):
        def funcs(p):
            with open(p) as fh:
                t = ast.parse(fh.read())
            return {n.name: ast.dump(n) for n in t.body if isinstance(n, ast.FunctionDef) and n.name in ('calib_weights', 'calib_weights_lcb')}
        ref = funcs(os.path.join(ROOT, 'alfworld', 'train_ppo_alf.py'))
        for p in glob.glob(os.path.join(ROOT, '*', 'train_ppo_alf*.py')):
            with self.subTest(p=os.path.relpath(p, ROOT)):
                self.assertEqual(funcs(p), ref)


class Configs(unittest.TestCase):
    def parser(self, env):
        """The argument parser of <env>/alf_final.py (the statements of its __main__ block up to parse_args)."""
        m = stubs.load(env, 'alf_final')
        with open(os.path.join(ROOT, env, 'alf_final.py')) as fh:
            t = ast.parse(fh.read())
        main = [n for n in t.body if isinstance(n, ast.If)][-1]
        body = []
        for s in main.body:
            if 'parse_args' in ast.unparse(s):
                break
            body.append(s)
        ns = dict(m.__dict__)
        exec(compile(ast.Module(body=body, type_ignores=[]), 'parser', 'exec'), ns)
        return ns['ap'], m

    def test_every_config_parses(self):
        import yaml
        sys.path.insert(0, os.path.join(ROOT, 'scripts'))
        import launch
        for env in ENVS:
            ap, m = self.parser(env)
            for p in sorted(glob.glob(os.path.join(ROOT, 'configs', env, '*.yaml'))):
                with open(p) as fh:
                    c = yaml.safe_load(fh)
                with self.subTest(config=os.path.relpath(p, ROOT)):
                    self.assertIn(c['arm'], m.ARMS)
                    self.assertEqual(sorted(c['runs']), list(range(1, len(c['runs']) + 1)))
                    for it in c.get('evaluate') or []:
                        self.assertLessEqual(it, c['iters'])
                    args = ['--arm', c['arm'], '--out', '/tmp/x', '--sglang', 'http://h:1', '--train-gpus', '0',
                            '--master-port', '1', '--iters', str(c['iters'])] + launch.cli_args(c['args'] or {})
                    if c['world_model'] != 'none':
                        args += ['--wm-url', 'http://h:2', '--wm-lora', 'wm_init']
                    if c.get('teacher'):
                        args += [f'--{c["teacher"]}', 'http://h:3']
                    a = ap.parse_args(args)
                    self.assertEqual(a.arm, c['arm'])
                    for k in c['runs']:      # trainer GPUs of every paper run cover all iterations, in order
                        segs = launch.trainer_segments(c, k, c['iters'])
                        self.assertEqual(segs[-1][0], c['iters'])
                        self.assertEqual([x for x, _ in segs], sorted(x for x, _ in segs))

    def test_trainer_segments(self):
        import yaml
        sys.path.insert(0, os.path.join(ROOT, 'scripts'))
        import launch
        with open(os.path.join(ROOT, 'configs', 'alfworld', 'ppo160.yaml')) as fh:
            c = yaml.safe_load(fh)
        self.assertEqual(launch.trainer_segments(c, 1, 55), [(18, 2), (36, 4), (55, 2)])
        self.assertEqual(launch.trainer_segments(c, 1, 14), [(14, 2)])
        self.assertEqual(launch.trainer_segments(c, 1, 30), [(18, 2), (30, 4)])
        self.assertEqual(launch.trainer_segments(c, 2, 55), [(55, 2)])


if __name__ == '__main__':
    unittest.main()
