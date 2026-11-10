# Checkpoints

Download from Hugging Face, [nuskonghou/wm-ppo](https://huggingface.co/nuskonghou/wm-ppo) (about 9.5 GB), into `weights/`
in the repository, so that the commands of the README work unchanged:

```bash
hf download nuskonghou/wm-ppo --local-dir weights
```

```
weights/
  SHA256SUMS  MANIFEST.tsv  README.md
  alfworld/      README.md  warmup/  initial_wm/  wmppo_run1_it036/{policy,critic,world_model}/
  scienceworld/  README.md  warmup/  initial_wm/  wmppo_run1_it060/{policy,critic,world_model}/
  sokoban/       README.md  warmup/  initial_wm/  wmppo_run1_it036/{policy,critic,world_model}/
```

Check the files with `(cd weights && sha256sum -c SHA256SUMS)`.

## What is released

All components are LoRA adapters (rank 64, alpha 128, all linear layers) on Qwen3-4B-Instruct-2507. The critic also has
a linear value head (`value_head.pt`) on the last prompt token.

| Directory | Content | Used by |
|---|---|---|
| `<env>/wmppo_run1_itNNN/policy` | WM-PPO policy, paper run 1 (seed 1), at the iteration of the main results | evaluation |
| `<env>/wmppo_run1_itNNN/critic` | its critic (`critic/` adapter, `value_head.pt`) | analysis |
| `<env>/wmppo_run1_itNNN/world_model` | its world model at the same iteration | analysis |
| `<env>/warmup` | the shared 12-iteration PPO warm-up: `it012/adapter`, `it012/critic`, `state.json` (task cursor), `metrics.jsonl` (iterations 1-12) | starting point of every method |
| `<env>/initial_wm` | the initial world model, trained on the warm-up's executed transitions | WM-PPO and its variants, Fixed WM |

The iterations are 36 on ALFWorld, 60 on ScienceWorld and 36 on text Sokoban (Table 2 and Table 18 of the paper).

## Run-1 numbers and the paper's means

The paper reports three runs of WM-PPO on ALFWorld and ScienceWorld and gives the results of each run in Table 15.
The released checkpoints are those of run 1:

| Environment | Run 1: greedy / pass@1 / pass@3 | Paper, mean ± SD of three runs: greedy / pass@1 / pass@3 |
|---|---|---|
| ALFWorld, iteration 36 | 88.3 / 89.7 / 95.3 | 87.5 ± 0.9 / 87.8 ± 1.7 / 93.4 |
| ScienceWorld, iteration 60 | 54.9 / 56.5 / 65.5 | 52.9 ± 2.3 / 51.6 ± 4.8 / 59.6 |
| Text Sokoban, iteration 36 | 87.0 / 84.4 (one run, Table 18) | |

Greedy decoding on the inference server is not bit-exact, so a repeated evaluation can differ from the official one on
a few games. `analysis/score_eval.py` reports the share of games whose greedy outcome equals the official one.

## Using the checkpoints

```bash
# evaluate a policy (evaluation protocol of the paper)
scripts/wmppo eval alfworld weights/alfworld/wmppo_run1_it036/policy wmppo_run1_it036
python3 analysis/score_eval.py alfworld wmppo_run1_it036

# train any method from the shared warm-up
mkdir -p alfworld/runs/wm
cp -r weights/alfworld/warmup alfworld/runs/ppo
cp -r weights/alfworld/initial_wm alfworld/runs/wm/warmup_log
scripts/wmppo train alfworld wmppo --run 1
```

For text Sokoban the warm-up goes to `sokoban/runs/ppo` (or set `WMPPO_SOKOBAN_WARMUP`).

## Exporting checkpoints from your own runs

`scripts/export_weights.sh OUT [LAYOUT]` copies the adapters and value heads named in `LAYOUT` (default
`scripts/weights_layout.tsv`, the layout above) from the run directories into `OUT`. It also writes the warm-up's
`state.json` and `metrics.jsonl`, sets `base_model_name_or_path` to `Qwen/Qwen3-4B-Instruct-2507`, and writes
`SHA256SUMS` and `MANIFEST.tsv`. The run directories are only read.
