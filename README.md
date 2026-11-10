# Learning to Act with World Models by Valuing What Comes Next

Code, configurations, per-game evaluation data and checkpoints for **WM-PPO**.
Paper: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX).

## Quick start

Two Docker images, the base model and the checkpoints are needed. Setup is under [Installation](#installation);
the checkpoints are on Hugging Face, [nuskonghou/wm-ppo](https://huggingface.co/nuskonghou/wm-ppo), and described in
[WEIGHTS.md](WEIGHTS.md).

**1. Evaluate the released checkpoints** (1 GPU; about 1 h on ALFWorld and 2 h on ScienceWorld)

```bash
scripts/wmppo data alfworld                                    # once: the ALFWorld games
scripts/wmppo eval alfworld weights/alfworld/wmppo_run1_it036/policy wmppo_run1_it036 --gpu 0
python3 analysis/score_eval.py alfworld wmppo_run1_it036       # compares with the official evaluation
scripts/wmppo eval scienceworld weights/scienceworld/wmppo_run1_it060/policy wmppo_run1_it060 --gpu 0
python3 analysis/score_eval.py scienceworld wmppo_run1_it060
```

**2. Check the paper's numbers from the released evaluation data** (CPU, Python 3 standard library)

```bash
python3 analysis/check_tables.py       # recomputes Tables 2, 7, 8, 10-13, 15-18, 21: "total: 981/981 cells match"
python3 analysis/diagnostics.py        # Section 2 and Figure 1
```

**3. Train WM-PPO from the shared warm-up** (7 GPUs; one iteration takes about 40 min on ALFWorld)

```bash
mkdir -p alfworld/runs/wm
cp -r weights/alfworld/warmup alfworld/runs/ppo                # the warm-up every method starts from
cp -r weights/alfworld/initial_wm alfworld/runs/wm/warmup_log  # the initial world model
scripts/wmppo train alfworld wmppo --run 1 --evaluate          # iterations 13-55, then the evaluation protocol
```

## Released checkpoints

The checkpoints are those of WM-PPO **run 1 (seed 1)** in each environment: policy, critic and world model at the
iteration of the main results, plus the shared warm-up and the initial world model. As in the paper, run k is the run
with the k-th seed; the paper reports three runs on ALFWorld and ScienceWorld, and the results of every run are in its
Table 15.

| Environment | Checkpoint | Greedy | pass@1 | pass@3 | Paper, mean ± SD of three runs (greedy / pass@1) |
|---|---|---|---|---|---|
| ALFWorld (274 games) | run 1, iteration 36 | 88.3 | 89.7 | 95.3 | 87.5 ± 0.9 / 87.8 ± 1.7 |
| ScienceWorld (284 variations) | run 1, iteration 60 | 54.9 | 56.5 | 65.5 | 52.9 ± 2.3 / 51.6 ± 4.8 |
| Text Sokoban (300 levels) | the single run, iteration 36 | 87.0 | 84.4 | -- | one run (Table 18) |

Greedy decoding on the inference server is not bit-exact, so a repeated evaluation can differ from the official one on a
few games.

## Method

A sparse terminal reward says little about individual steps, but each state transition can change the conditions for
success long before the reward arrives. WM-PPO combines a world model, trained on the agent's real transitions to predict
the next observation, with PPO's critic, which estimates the current policy's probability of success:

- for every action the policy samples at a decision point, executed or not, the world model predicts the next
  observation and the critic values the predicted successor;
- **policy channel**: comparisons among these values give the policy auxiliary advantages;
- **value channel**: their mean gives the critic an auxiliary target;
- **gate**: one shared gate weights both channels by how reliably comparisons with similar value margins predicted real
  outcomes earlier in training.

The policy, critic and world model are LoRA adapters on Qwen3-4B-Instruct-2507 and learn only from the agent's own
interaction (no teacher, no privileged information). Only the policy is used at inference.

| Method (Table 2) | ALFWorld, it. 36: greedy / pass@1 | ScienceWorld, it. 60: greedy / pass@1 |
|---|---|---|
| WM-PPO | 87.5 ± 0.9 / 87.8 ± 1.7 | 52.9 ± 2.3 / 51.6 ± 4.8 |
| WM-PPO (policy channel only) | 88.4 ± 2.3 / 89.4 ± 1.6 | 47.2 ± 2.8 / 47.2 ± 1.7 |
| PPO-160 | 63.5 ± 9.8 / 71.0 ± 6.0 | 39.3 ± 1.5 / 42.6 ± 2.5 |
| PPO-256 | 83.1 ± 3.1 / 84.8 ± 1.2 | 43.8 ± 8.7 / 43.5 ± 6.6 |

All baselines of the paper are in `configs/` (see [Baselines](#baselines)); `results/reference_tables/main.tex` is the
full Table 2.

## Repository layout

| Path | Content |
|---|---|
| `alfworld/`, `scienceworld/`, `sokoban/` | training loop (`alf_final.py`), rollouts, environments, prompts, trainers, evaluation scripts; ALFWorld also has the Section 2 diagnostics |
| `wmppo/paths.py` | every machine-specific location (base model, data, images), each overridable by an environment variable |
| `configs/<env>/<method>.yaml` | one file per reported method: arguments, paper run → `--seed`, GPU layout, evaluated iterations |
| `scripts/wmppo` | entry point: runs `scripts/launch.py` in the environment image |
| `scripts/launch.py` | inference servers, training, warm-up, initial world model, evaluation protocol |
| `scripts/export_weights.sh` | collects checkpoints from run directories into a release layout with checksums |
| `docker/` | the two images and the pinned Python packages |
| `results/` | per-game official evaluations of every reported run, training curves, gate logs, diagnostics, the paper's tables ([results/README.md](results/README.md)) |
| `analysis/` | table check, figures, Section 2 numbers, scoring of new evaluations |
| `docs/REPRODUCE.md` | each table and figure: data, command, configs |
| `tests/` | CPU tests |

## Installation

The host needs Linux, NVIDIA GPUs, Docker with the NVIDIA Container Toolkit, and Python 3 for `analysis/`
(`pip install -r analysis/requirements.txt` for the figures and the tests). The paper's runs used A100 40GB nodes
(two early segments of two baseline runs ran on H100 nodes).

```bash
docker build -t wmppo-trainer -f docker/Dockerfile.trainer docker   # sglang + PyTorch + peft: servers and trainers
docker build -t wmppo-env -f docker/Dockerfile.env docker           # ALFWorld, ScienceWorld (Java 11), Sokoban
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507   # hf: pip install huggingface_hub
hf download nuskonghou/wm-ppo --local-dir weights                                   # checkpoints, about 9.5 GB
```

- `wmppo-trainer` is the `slimerl/slime` image of all runs in the paper, pinned by digest (public tag
  `slimerl/slime:nightly-dev-20260707a`), with peft 0.21.2.
- `wmppo-env` is Ubuntu 22.04 with Python 3.10, Java 11 and the packages of `docker/requirements-*.txt` (ALFWorld and
  Sokoban share one environment, ScienceWorld has its own).
- `scripts/wmppo` runs the launcher in `wmppo-env` with the Docker socket mounted; the launcher starts every inference
  server and trainer as a sibling container of `wmppo-trainer`. Directories are mounted at the same absolute paths in all
  containers.

Locations, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `WMPPO_MODEL` | `models/Qwen3-4B-Instruct-2507` | base model directory |
| `ALFWORLD_DATA` | `data/alfworld` | ALFWorld data (holds `json_2.1.1/`) |
| `WMPPO_GPUS` | `0,1,2,3,4,5,6,7` | GPUs the launcher assigns, in order |
| `WMPPO_TMP` | `/tmp` | temporary files of the environment workers (the ALFWorld planner writes many) |
| `WMPPO_TEACHER_MODEL` | `<models>/Qwen3-32B` | teacher of the OCSD (32B) and OPD (32B) baselines |
| `WMPPO_MOUNTS` | | extra host directories for the containers, colon-separated |

## Data

- **ALFWorld**: `scripts/wmppo data alfworld` runs `alfworld-download` into `ALFWORLD_DATA`. Training uses the 3,553
  `train` games; evaluation pools the 140 seen-room (`valid_seen`) and 134 unseen-room (`valid_unseen`) games.
- **ScienceWorld**: the simulator ships with the `scienceworld` package (1.2.3). The 18 tasks, the training variations
  and the 284 evaluation variations (dev 140, test 144) are defined in `scienceworld/env.py`.
- **Text Sokoban**: 6x6 levels with one box and at most 15 steps, generated from seeds; `sokoban/data/levels.json` holds
  the seeds of the 18,908 training and the 300 held-out evaluation levels.

## Training

```bash
scripts/wmppo train ENV METHOD [--run K] [--evaluate] [--iters N]
```

- `configs/ENV/METHOD.yaml` holds the arguments of the reported runs; `runs` maps paper run K to `--seed`.
- Every method starts from the shared 12-iteration warm-up (`ENV/runs/ppo`, task cursor included), so all methods see
  the same tasks from iteration 13 on. Methods with a world model start from the initial world model
  (`ENV/runs/wm/warmup_log`). Copy both from the checkpoints as in the quick start, or retrain them (below).
- Output: `ENV/runs/METHOD_runK/` with `itNNN/{adapter,critic,wm}`, the trees and statistics of each iteration,
  `metrics.jsonl`, and greedy evaluations every three iterations (`eval_NNN.json`). An interrupted run resumes when the
  command is repeated.
- `--evaluate` runs the evaluation protocol afterwards for the iterations reported in the paper; `--iters N` stops
  after iteration N.

GPU layout, from `layout` in each config: one GPU per inference server (policy; world model; two for the 32B teacher),
then the trainer GPUs and, for WM-PPO, the world-model trainer GPUs.

| Method | GPUs | Time per iteration, ALFWorld / ScienceWorld (Table 17) |
|---|---|---|
| WM-PPO | 7: policy server, 2 world-model servers, 2 trainer, 2 world-model trainer | 40 / 55 min |
| PPO-160 and its variants | 3: policy server, 2 trainer | 32 / 34 min (PPO-160) |
| Wide-Tree-PPO | 3 | 31 / 37 min |
| Q-PPO | 3 | 46 / 70 min |
| PPO-256 | 5: policy server, 4 trainer | 28 / 32 min |
| OCSD (32B), OPD (32B) | 5: policy server, 2 teacher, 2 trainer | |

Times are the measured means of Table 17 (A100 40GB, including the in-training evaluation every third iteration).

The number of trainer GPUs sets the data-parallel split of each policy and critic minibatch. Where a reported run used a
different number, the config lists it under `run_trainer_gpus`, and the launcher uses it.

**Warm-up and initial world model.** The paper's runs start from the released warm-up. To train new ones:

```bash
scripts/wmppo warmup alfworld     # iterations 1-12 of PPO -> alfworld/runs/ppo
scripts/wmppo init-wm alfworld    # world model on the warm-up's executed transitions -> alfworld/runs/wm/warmup_log
```

## Baselines

All baselines start from the same warm-up and follow the same evaluation protocol; Appendix D of the paper describes
each one. Config names:

| Group (as in Table 2) | ALFWorld | ScienceWorld |
|---|---|---|
| WM-PPO and its variants | `wmppo`, `wmppo_policy_only` | `wmppo`, `wmppo_policy_only`, `wmppo_terminal_value` |
| Standard training and refinements | `ppo160`, `ppo256`, `grpo`, `grpo_g4`, `gigpo`, `hgpo`, `tree_grpo`, `tree_grpo_8leaves`, `capo`, `capo_clip0.1`, `capo_clip0.001`, `sopo`, `stppo`, `stppo_clip3e-4`, `vapo_lite` | `ppo160`, `ppo256`, `grpo`, `grpo_g4`, `gigpo`, `gigpo_g4`, `capo`, `capo_clip0.3`, `capo_clip0.03`, `sopo`, `sopo_clip0.2`, `sopo_clip0.1`, `stppo` |
| Structural controls | `qppo`, `wide_tree_ppo`, `fixed_wm` | `qppo`, `wide_tree_ppo`, `fixed_wm` |
| Auxiliary feedback added to PPO-160 | `paw`, `majority_vote`, `ocsd_self`, `ocsd_32b`, `opd_32b` | `paw`, `paw_coef1.0`, `paw_coef0.25`, `majority_vote`, `ocsd_self`, `ocsd_32b`, `opd_32b` |

The settings tried for CAPO, SO-PPO, ST-PPO, GRPO, Tree-GRPO and PaW are those of Tables 7 and 8 of the paper; each
config's `name` says which setting it is.

- **PaW** reads its world-model prediction prompts from the authors' repository instead of copying them; clone it into
  `third_party/`:
  ```bash
  git clone https://github.com/ColinLu50/Policy-WorldModel-Agent third_party/Policy-WorldModel-Agent
  git -C third_party/Policy-WorldModel-Agent checkout 4feb964
  ```
- **OCSD (32B)** and **OPD (32B)** serve Qwen3-32B as the teacher (`WMPPO_TEACHER_MODEL`).

## Gate ablation

On ScienceWorld, `gate_ablation_no_gate` fixes the gate weights of the three margin bins to (1, 1, 1) and
`gate_ablation_top_bin` to (0, 0, 1); both are compared with the first WM-PPO run (Table 21 and Figure 9 of the paper).

```bash
scripts/wmppo train scienceworld gate_ablation_no_gate --evaluate
scripts/wmppo train scienceworld gate_ablation_top_bin --evaluate
```

## Text Sokoban

WM-PPO, PPO-160 and ProAct's MC-PPO (`configs/sokoban/`) start from a shared 12-iteration warm-up (Appendix E.6 and
Table 18 of the paper). The Sokoban configs use two policy servers (WM-PPO: 8 GPUs).

```bash
mkdir -p sokoban/runs/wm
cp -r weights/sokoban/warmup sokoban/runs/ppo
cp -r weights/sokoban/initial_wm sokoban/runs/wm/warmup_log
scripts/wmppo train sokoban wmppo --evaluate
```

## Evaluation protocol

`scripts/wmppo eval ENV ADAPTER NAME` starts one inference server with only this adapter, then runs one greedy episode
per evaluation game and three temperature-1 episodes per game (`ct_infer.py --rule random`). It writes
`ENV/runs/final_eval/greedy_NAME.json` and `infer_NAME_random_f{1,2,3}.jsonl`.

- greedy: mean success of the greedy episodes;
- pass@1: mean over games of the success rate of the three sampled episodes;
- pass@3: share of games with at least one sampled success.

Only the policy is used. `python3 analysis/score_eval.py ENV NAME` computes the three numbers. For names of the form
`<config>_run<k>_it<NNN>`, it also compares them with the official evaluation in `results/eval/`.

## Reproducing the tables and figures

```bash
python3 analysis/check_tables.py [--show]                    # tables, cell by cell against results/reference_tables/
python3 analysis/diagnostics.py --plot figures/diagnostics.pdf
python3 analysis/figures.py                                  # figures/{frontier,gate_weights,gate_ablation,curves}.pdf
```

[docs/REPRODUCE.md](docs/REPRODUCE.md) maps every table and figure to its data, command and configs.

## Tests

```bash
python3 -m unittest discover -s tests -v    # CPU only (PyYAML); heavy dependencies are stubbed
```

The tests compile and import every module, check the gate and advantage computations against the logged training
values, parse every config and recompute the paper's tables from `results/`.

## Citation

```bibtex
@article{gao2026wmppo,
  title   = {Learning to Act with World Models by Valuing What Comes Next},
  author  = {Gao, Chenghao and Zhang, Zhi and Jin, Dian and Pan, Yihe and Zhou, Doudou},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Apache License 2.0 ([LICENSE](LICENSE)). Code adapted from verl-agent, gym-sokoban and Self-Distillation-AgentRL is
listed with its license notices in [NOTICE](NOTICE). The base model Qwen3-4B-Instruct-2507 is released by the Qwen team
under Apache 2.0.
