# Results

Data of every run reported in the paper. Run names are `<config>_run<k>`: `<config>` is the file in `configs/<env>/` and
`k` is the run of that method in the paper (Table 15). `_rerun` marks a second run of an existing seed, launched later
with the same configuration. These second runs are listed in the appendix tables and enter no mean or test.

## Files

| File | Content |
|---|---|
| `eval/<env>.csv.gz` | official evaluation of every reported run and iteration, per game: `run, iteration, protocol, split, game, success, steps`. `protocol` is `greedy` or `sample1`-`sample3` (the three temperature-1 runs). Game keys: ALFWorld `<split>/<task>/<trial>` (relative to `json_2.1.1/`), ScienceWorld `<task>\|<variation>`, Sokoban `valid/<seed>/level` |
| `real_steps.csv` | executed policy steps per iteration and their sum from iteration 13 (`cum_exec_steps_from_it13`): the "real steps" of the tables and figures |
| `curves.csv` | per run and iteration (ALFWorld, ScienceWorld): greedy success on the dev / test games of the in-training evaluation (every third iteration), training success, cumulative executed steps, wall time, KL and clip fraction of the policy update |
| `curves_sokoban.csv` | the same for text Sokoban |
| `gate.csv` | per iteration of the methods with valued candidates (see below) |
| `cost_per_iteration.csv` | per iteration (Table 17): executed and replayed steps, policy samples, world-model generations, chains and branches (runs with decoded counts); for the three runs of WM-PPO, PPO-160, PPO-256, Wide-Tree-PPO and Q-PPO also the measured wall time (`wall_s`, including the in-training evaluation), the GPU type of the node (`gpu`) and the GPUs allocated to the run (`gpus`: inference servers, trainer and world-model trainer) |
| `diagnostics/` | inputs of Section 2 and Figure 1 (ALFWorld): per-game outcomes of the candidate-selection rules and of the offline update (`select_*.jsonl`, `offline_update_*.jsonl`), and the states of the pairwise test with their Monte-Carlo returns and critic values (`pairwise_test_states.json`) |
| `reference_tables/` | the paper's tables as LaTeX (below) |

`gate.csv` columns:

- `distill_states`: decision points with at least two distinct sampled actions whose predicted successors were valued;
  `per_bin`: their number per margin bin. The margin is the gap between the two highest values; the bins split at
  this iteration's 50th and 80th margin percentiles (`thr`).
- `w`: gate weight of each bin; `rows`: auxiliary policy rows.
- `calib_now_acc`: share of this iteration's checkable comparisons whose sign agreed with the real outcome difference,
  and their number. `calib_now_gain`: mean signed gain per bin of this iteration's comparisons.
- `calib_hist_n`, `calib_hist_gain`: number of comparisons and mean gain per bin over the preceding four iterations,
  which set `w`.
- ScienceWorld only: `ties` (decision points whose two highest values are equal; they carry no preference) and
  `term_overrides` (predicted successors whose predicted ending replaced the critic's value).
- `policy_samples`, `adv_raw_std` and `kl` of the policy update.

## Statistics

All definitions are in `analysis/evaldata.py`:

- greedy: mean success of the greedy episodes; pass@1: mean over games of the three sampled runs; pass@3: share of
  games with at least one sampled success;
- method values are means over the method's runs, and ± is the sample standard deviation over runs;
- significance marks: per-game paired z-test (two-sided, p < 0.05) on the per-game success averaged over each method's
  runs, against WM-PPO at the same iteration;
- Section 2 reports exact sign tests on the games where two rules differ (`analysis/diagnostics.py`,
  `alfworld/ct_paired.py`).

## Reference tables

| File | Paper | Recomputed by `analysis/check_tables.py` |
|---|---|---|
| `main.tex` | Table 2 | yes |
| `lookahead_alfworld.tex` | Table 6 | no (Appendix C analysis) |
| `baseline_tuning.tex`, `baseline_tuning_sw.tex` | Tables 7, 8 | yes |
| `full_alfworld_a.tex`, `full_alfworld_b.tex` | Tables 10, 11 | yes |
| `full_scienceworld_a.tex`, `full_scienceworld_b.tex` | Tables 12, 13 | yes |
| `reach.tex` | Table 14 | no |
| `seeds.tex` | Table 15 | yes |
| `sw_tasks.tex` | Table 16 | yes |
| `cost.tex` | Table 17 | yes |
| `sokoban.tex` | Table 18 | yes |
| `gate_ablation_sw.tex` | Table 21 | yes |
| `gate.tex` | Table 22 | no (from `gate.csv`) |
