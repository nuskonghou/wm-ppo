# Reproducing the paper

For each table and figure: the data in `results/`, the command that recomputes it, and the configs whose runs produced
the data. Table and figure numbers follow the arXiv version of the paper. Training commands are shown for ALFWorld;
replace `alfworld` with `scienceworld` or `sokoban` for the other environments.

## From the released data (CPU)

| Paper | Data | Command |
|---|---|---|
| Table 2 (main results) | `results/eval/*.csv.gz`, `results/real_steps.csv` | `python3 analysis/check_tables.py` |
| Tables 7, 8 (baseline settings) | `results/eval/` | same |
| Tables 10-13 (all methods and iterations) | `results/eval/`, `results/real_steps.csv` | same |
| Table 15 (individual runs) | `results/eval/` | same |
| Table 16 (ScienceWorld task types) | `results/eval/scienceworld.csv.gz` | same |
| Table 17 (computation per iteration) | `results/cost_per_iteration.csv`, `results/real_steps.csv` | same |
| Table 18 (text Sokoban) | `results/eval/sokoban.csv.gz`, `results/real_steps.csv` | same |
| Table 21 (gate ablation) | `results/eval/scienceworld.csv.gz` | same |
| Section 2 numbers, Figure 1 | `results/diagnostics/` | `python3 analysis/diagnostics.py --plot figures/diagnostics.pdf` |
| Figures 4 and 12 (success against real steps) | `results/eval/`, `results/real_steps.csv` | `python3 analysis/figures.py --only frontier` |
| Figure 8 (gate weights) | `results/gate.csv` | `python3 analysis/figures.py --only gate` |
| Figure 9 (gate ablation curves) | `results/curves.csv` | `python3 analysis/figures.py --only ablation` |
| Figure 13 (learning curves) | `results/curves.csv` | `python3 analysis/figures.py --only curves` |

`check_tables.py` recomputes every compared cell and prints `total: 981/981 cells match`; `--show` lists them.
Bold face is not compared. The figures use the same numbers as the paper in a plain style.

Tables 6, 14 and 22 are in `results/reference_tables/` (`lookahead_alfworld.tex`, `reach.tex`, `gate.tex`); Table 22
summarises `results/gate.csv`. Tables 1, 3, 4 and 5
describe the protocol, related work, roles and hyperparameters; the hyperparameters are the arguments in `configs/`
and the defaults of the trainers. Figures 2 and 3 are diagrams of the method.

The remaining tables and figures (Tables 9, 19, 20, 23-29; Figures 5-7, 10, 11, 14-19) are analyses of the per-iteration
training trees, trajectories and model outputs of the runs. They are not part of `results/`.

## Training runs behind the tables

Each config lists its paper runs (`runs`: paper run → `--seed`) and the iterations evaluated in the paper
(`evaluate`). `--evaluate` runs the evaluation protocol on those iterations after training. The output lands in
`ENV/runs/final_eval/greedy_<config>_run<k>_it<NNN>.json` and `infer_<config>_run<k>_it<NNN>_random_f{1,2,3}.jsonl`;
`analysis/score_eval.py` compares it with `results/eval/`.

```bash
# Table 2 / Tables 10-13 / Table 15: methods with three runs
for k in 1 2 3; do
  for m in wmppo wmppo_policy_only ppo160 ppo256 qppo wide_tree_ppo; do
    scripts/wmppo train alfworld $m --run $k --evaluate
  done
done
scripts/wmppo train alfworld ocsd_self --run 2 --evaluate      # OCSD (self) has three runs on ALFWorld
scripts/wmppo train alfworld ocsd_self --run 3 --evaluate
# methods with one run (Table 2, Tables 7-8, 10-13)
for m in grpo gigpo capo sopo stppo fixed_wm paw majority_vote ocsd_self ocsd_32b opd_32b \
         grpo_g4 hgpo tree_grpo tree_grpo_8leaves vapo_lite capo_clip0.1 capo_clip0.001 stppo_clip3e-4; do
  scripts/wmppo train alfworld $m --evaluate
done
```

| Paper | ALFWorld configs | ScienceWorld configs |
|---|---|---|
| Table 2 | `wmppo`, `wmppo_policy_only`, `ppo160`, `ppo256`, `grpo`, `gigpo`, `capo`, `sopo`, `stppo`, `qppo`, `wide_tree_ppo`, `fixed_wm`, `paw`, `majority_vote`, `ocsd_self`, `ocsd_32b`, `opd_32b` | the same names (CAPO at iteration 18) |
| Table 7 / Table 8 | `capo_clip0.001`, `capo_clip0.1`, `capo`, `stppo_clip3e-4`, `stppo`, `grpo_g4`, `grpo`, `tree_grpo_8leaves`, `tree_grpo` | `capo_clip0.3`, `capo`, `capo_clip0.03`, `sopo_clip0.2`, `sopo`, `sopo_clip0.1`, `stppo`, `paw_coef1.0`, `paw`, `paw_coef0.25` |
| Tables 10-13 | all of the above, plus `hgpo`, `vapo_lite` | all of the above, plus `wmppo_terminal_value`, `grpo_g4`, `gigpo_g4` |
| Table 15 | runs 1-3 of the methods with three runs | same |
| Table 18 | `sokoban/`: `wmppo`, `mcppo`, `ppo160` | |
| Table 21, Figure 9 | | `gate_ablation_no_gate`, `gate_ablation_top_bin` (the first runs of WM-PPO and PPO-160 as references) |

Before training, put the shared warm-up and the initial world model in place (from the checkpoints, see WEIGHTS.md,
or retrained with `scripts/wmppo warmup ENV` and `scripts/wmppo init-wm ENV`). Iteration counts are those of the
paper: ALFWorld 36 (55 for the methods of Table 11), ScienceWorld 60, Sokoban 36.

## Section 2 diagnostics (ALFWorld)

The diagnostics use the policy and critic of PPO continued from the warm-up to iteration 24
(`configs/alfworld/diagnostics_ppo.yaml`). Each step below uses one script of `alfworld/`; the docstring of each script
describes its inputs and outputs.

- `scripts/wmppo run alfworld SCRIPT ...` runs an environment-side script with paths relative to `alfworld/`.
- `scripts/wmppo gpu [--detach] [--name NAME] alfworld GPUS SCRIPT ...` runs a trainer-image script on GPUS. It adds
  `--base` and uses torchrun for the trainers.
- `scripts/wmppo serve NAME GPU PORT --lora NAME=DIR` starts an inference server and loads adapters into it.

```bash
scripts/wmppo train alfworld diagnostics_ppo                                   # 3 GPUs
P=$PWD/alfworld/runs/diagnostics_ppo_run1/it024
POL=http://127.0.0.1:30012 VAL=http://127.0.0.1:30031 WM=http://127.0.0.1:30025

# servers: policy (GPU 0), critic values (GPU 1)
scripts/wmppo serve wmppo_diag_pol 0 30012 --lora ct_ppo=$P/adapter
scripts/wmppo gpu --detach --name wmppo_diag_value alfworld 1 ct_value_server.py --critic $P/critic --port 30031

# world model of Section 2.2: transitions under this policy (train shards 0-1, evaluation games), 1 epoch on 2 GPUs
# (in the paper's run: 48,440 training transitions, 44% of them unexecuted candidates)
for s in 0 1; do scripts/wmppo run alfworld wm_collect.py --lora ct_ppo --sglang $POL --shard $s --n-games 500 --out runs/wm/train_s$s.jsonl; done
for sp in valid_seen valid_unseen; do scripts/wmppo run alfworld wm_collect.py --lora ct_ppo --sglang $POL --split $sp --out runs/wm/$sp.jsonl; done
scripts/wmppo gpu alfworld 2,3 train_wm.py --data runs/wm/train_s0.jsonl runs/wm/train_s1.jsonl --out runs/wm/obs
scripts/wmppo serve wmppo_diag_wm 2 30025 --lora wm_obs=$PWD/alfworld/runs/wm/obs
scripts/wmppo run alfworld wm_eval.py --url $WM --lora wm_obs --name obs --n 1500     # prediction accuracy (Appendix C)

# (a) candidate selection, one episode per evaluation game
R="scripts/wmppo run alfworld ct_infer.py --arm ppo --lora ct_ppo --sglang $POL --out runs/critic_test"
$R --rule random; $R --rule random --tag _r2; $R --rule sc; $R --rule peek
for r in critic_only critic_parent_adm critic_masked; do $R --rule $r --value-url $VAL; done
$R --rule critic_wm --value-url $VAL --wm-url $WM --wm-lora none   --tag _untrained_wm
$R --rule critic_wm --value-url $VAL --wm-url $WM --wm-lora wm_obs --tag _trained_wm

# (b) pairwise test: three random states per game, three continuations per candidate, scored by the critic
for s in 0 1 2; do scripts/wmppo run alfworld ct_collect.py --arm ppo --lora ct_ppo --sglang $POL --out runs/critic_test --seed $s; done
for f in ppo ppo_s1 ppo_s2; do scripts/wmppo gpu alfworld 3 ct_score.py --data runs/critic_test/$f.json --critic $P/critic; done

# (c) offline update: labels on unseen training games (shards 4-5), one PPO-clip epoch, two sampled runs per game
for s in 4 5; do scripts/wmppo run alfworld offline_collect.py --lora ct_ppo --sglang $POL --value-url $VAL --wm-url $WM \
  --wm-lora wm_obs --shard $s --n-games 250 --out runs/offline/collect_s$s.jsonl; done
for l in la sc; do
  scripts/wmppo run alfworld offline_build.py --label $l --out runs/offline/samples_$l.jsonl runs/offline/collect_s4.jsonl runs/offline/collect_s5.jsonl
  scripts/wmppo gpu alfworld 3,4 train_lora.py --data runs/offline/samples_$l.jsonl --adapter-in $P/adapter \
    --adapter-out runs/offline/adapter_$l --lr 1e-5 --mb 64 --pack-tokens 16384
  curl -s -X POST $POL/load_lora_adapter -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"offline_$l\", \"lora_path\": \"$PWD/alfworld/runs/offline/adapter_$l\"}"
  for t in _r1 _r2; do scripts/wmppo run alfworld ct_infer.py --arm offline_$l --rule random --lora offline_$l --sglang $POL --out runs/critic_test --tag $t; done
done

python3 analysis/diagnostics.py --runs alfworld/runs/critic_test --plot figures/diagnostics.pdf
python3 alfworld/ct_paired.py ppo_critic_wm_trained_wm ppo_sc     # paired comparison with exact sign test (CT_DIR)
scripts/wmppo stop
```

`analysis/diagnostics.py --runs` reads these outputs instead of `results/diagnostics/`, which holds the same files from
the paper's runs (renamed; for example `infer_ppo_critic_wm_trained_wm.jsonl` is `select_critic_trained_wm.jsonl`).
