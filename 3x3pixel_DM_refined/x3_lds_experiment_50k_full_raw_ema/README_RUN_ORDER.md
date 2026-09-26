# X3 50k LDS — RAW + EMA Traj-TracIn

## 100-query projected first/second-order run

The current configuration reuses the already-trained base/LDS models and mask
bank. It samples 100 queries (75 prompted, 25 unprompted), then runs:

- raw first-order next-checkpoint Traj-TracIn, 100 timestamps x 10 train-noise MC;
- raw second-order next-delta HVP Traj-TracIn with coefficient 0.5;
- shared 4096-D CountSketch projection for both Traj orders;
- `linear`, `timestamp_sum_squared`, and `termwise_squared` contractions;
- EMA DAS at 100 timestamps x 10 query MC, with train-gradient MC=10;
- the existing DAS lambda sweep unchanged.

An additional backward first-order mode evaluates checkpoints `c=1..49`
(zero-based indices, 49 transitions including the final checkpoint) at
`theta_c`, targets `theta_{c-1}`, and weights the transition with the previous
checkpoint learning rate `eta_{c-1}`. It uses the same 100 queries, 100x10
train-gradient sampling, 4096-D projection, and three contractions:

```bash
python 04_launch_projected_backward_100q_4gpu.py
```

AdamW optimizer-history correction is intentionally disabled. No model or LDS
subset retraining is part of this run.

Run from this directory:

```bash
python 00_verify_50k_config.py
python 02_build_queries.py
python 04_launch_projected_100q_4gpu.py
python 05_collect_subset_outputs.py
python 07_run_all_lds.py
```

Completed family banks are skipped on rerun. The query and LDS response stages
replace the old 16-query bank with the requested 100-query bank.

## Query-specific top-1000 removal retraining

This follow-up trains 200 models: one removal model per query for exact,
unprojected raw first-order next-checkpoint Traj-TracIn linear scores, and one
per query for EMA DAS at lambda 10. Both methods rank their saved scores directly in descending
order, without applying the LDS prediction sign, and remove the largest 1000
training indices. Each retrain keeps the original seed and optimization
settings on the remaining 49,000 examples.

The Traj ranking comes from a raw-parameter exact-gradient score bank (no
projection), while DAS ranking comes from its EMA score bank. Both retrained models are
evaluated from their EMA weights. The evaluator replays the query's exact saved
initial noise and prompt, compares directly against the saved base-EMA query
trajectory, and records full-trajectory and endpoint changes.

For a controlled projected-versus-exact comparison, the exact bank reuses the
projected bank's train-noise seeds, checkpoint/timestamp weights, MC averaging,
and loss reductions. Exact dots are divided by the projected dimension because
both CountSketch vectors use `1/sqrt(d)` scaling; this positive constant does
not change the top-1000 ordering.

```bash
python 04_launch_exact_traj_next_100q_4gpu.py
python 09_prepare_topk_removal.py
python -u 10_launch_topk_removal_8gpu.py --gpus 0,1,2,3 2>&1 \
  | tee x3_lds_exp_50k/logs/topk_removal_200_models.log
```

The launcher is restartable at the completed-model level and saves a training
resume checkpoint every 10 epochs. Final aggregate outputs are written to
`x3_lds_exp_50k/topk_removal_retrain/summary.json` and
`per_query_results.csv`. A paired Traj-versus-DAS table is saved as
`paired_comparison.csv`.

## Earlier raw/EMA baseline

This version runs FIVE attribution families per query:

1. `traj_ref_raw`
2. `traj_next_raw`
3. `traj_ref_ema`
4. `traj_next_ema`
5. `das` (EMA)

For 16 queries this is 80 attribution jobs total.

Important:
- Dataset seed = 67
- Every model training seed = 67
- N = 50,000
- LDS subset size = 12,500
- 192 masks
- LDS prediction orientation is `-(membership @ attr)`
- DAS lambda sweep extends through 100000

Run order:

```bash
python 00_verify_50k_config.py
python 00_prepare_experiment.py
python 01_launch_train_4gpu.py
python 02_build_queries.py
python 03_cache_traj_query_gradients.py   # optional diagnostic cache
python 04_launch_attribution_4gpu.py
python 05_collect_subset_outputs.py
python 07_run_all_lds.py
python 08_print_best_comparison.py
```

# X3 50k LDS Experiment

This folder is the 50,000-datapoint scale-up of the 5k experiment.

Fixed design:
- dataset generation seed: 67
- training seed for every model: 67
- N_TRAIN: 50,000
- LDS subset fraction: 25%
- subset size: 12,500
- LDS masks: 3 seeds × 64 = 192
- prompted + unprompted subset models
- 16 queries
- Traj-Ref / Traj-Next
- DAS projection dim: 4096
- fast batched attribution backend

All outputs go under `x3_lds_exp_50k/`, so the 5k experiment is untouched.

# X3 5000 / LDS experiment

## Training seed

All full and subset models use the SAME training seed:

```python
TRAIN_SEED = 67
```

The LDS mask-bank seeds (0, 1, 2) only determine which 25% datapoints are selected.
They do NOT change model initialization / loader shuffle / diffusion timestep-noise seed.


## Intended experiment

- Base data: 5000 generated points, generator seed 67.
- Full models: prompted + unprompted.
- LDS masks: 3 seeds × 64 masks, each 25% = 1250 points.
- The 192 masks are shared across model families.
- Default config trains both prompted and unprompted final subset checkpoints per mask.
- 7 initial noise seeds:
  - seeds 0,1,2 are prompted; 4 random prompts each => 12 queries
  - seeds 3,4,5,6 are unprompted => 4 queries
  - total = 16
- Traj-TracIn: reference target + next-checkpoint target.
- DAS: projected dimension 4096, later-JAX logic, lambda sweep.
- LDS: Spearman correlation of subset attribution sums vs actual subset-model response.

## Run order

```bash
python 00_prepare_experiment.py
python 01_launch_train_4gpu.py
python 02_build_queries.py
python 03_cache_traj_query_gradients.py
```

Recommended 4-GPU attribution launch:

```bash
python 04_launch_attribution_4gpu.py
```

Or run methods manually:

```bash
python 04_run_attribution.py --gpu 0 --method traj_ref
python 04_run_attribution.py --gpu 1 --method traj_next
python 04_run_attribution.py --gpu 2 --method das
```

Then:

```bash
python 05_collect_subset_outputs.py

python 06_lds_eval.py --method traj_ref  --metric traj_ref
python 06_lds_eval.py --method traj_next --metric traj_ref

python 06_lds_eval.py --method das --metric traj_ref --lambda 0.1
python 06_lds_eval.py --method das --metric traj_ref --lambda 0.2
# ... sweep ...
```

You can also evaluate against simple-loss response:

```bash
python 06_lds_eval.py --method traj_ref --metric simple_loss
python 06_lds_eval.py --method das --metric simple_loss --lambda 2.0
```

## Important count

`3 × 64 = 192` refers to subset MASKS / subset JOB SLOTS.

The default:
```python
SUBSET_TRAIN_FAMILIES = ("prompted", "unprompted")
```
trains two final models for each mask, because a prompted LDS query should be compared
against prompted subset models and an unprompted LDS query against unprompted subset models.

If you truly want 192 trained subset models total, change it to one family.


## Eta convention

There are two unrelated eta symbols in the code:

1. **Traj-TracIn checkpoint eta**
   - `eta_c = learning rate at checkpoint c`
   - saved directly inside every newly trained checkpoint as `ckpt["eta"]`
   - both Traj-Ref and Traj-Next multiply each checkpoint contribution by `eta_c`
   - query gradients themselves are NOT multiplied when cached; eta is applied when aggregating TracIn scores.

2. **DDIM sampling eta**
   - remains `0.0`
   - this controls DDIM stochasticity and is unrelated to TracIn checkpoint weighting.

DAS does not use the TracIn checkpoint eta weighting.



## Progress / ETA

The updated scripts print approximate ETA at multiple levels:

- `train_worker.py`: epoch-level elapsed/ETA for each model.
- `01_launch_train_4gpu.py`: overall completed-model count and queue ETA.
- Traj attribution: every 100 training points plus checkpoint/global ETA.
- DAS attribution: feature-build percentage and ETA.
- `04_launch_attribution_4gpu.py`: overall 48-job attribution queue ETA.
- `05_collect_subset_outputs.py`: LDS query/subset progress and ETA.

Early ETA values can be noisy; they stabilize after several completed units.


## Fast attribution backend

This package replaces the low-GPU-utilization attribution loop with:

- Traj-TracIn: batched MC + batched datapoints + `torch.func.jvp`.
  The JVP returns the exact per-example dot product
  `<query_gradient, train_loss_gradient>` without forming one train gradient
  with one backward per datapoint.
- DAS: `torch.func.vmap(grad)` across a train batch, batched CountSketch,
  GPU-resident 4096-D Gram, GPU lambda solve.

Main tuning knobs:

```python
TRACIN_SCORE_BATCH_SIZE = 512
DAS_FEATURE_BATCH_SIZE = 64
```

If memory allows, increase them. If OOM occurs, reduce them.

## Adam/clipping-aware raw SOURCE-DAS

This is a separate ablation and does not overwrite the SGD-style SOURCE-DAS
artifacts. Each 20-epoch segment uses one midpoint checkpoint for EK-FAC
curvature, diagonal curvature, and datapoint gradients; the final segment also
includes the epoch-200 endpoint. This gives 10 midpoints + 1 final endpoint =
11 expensive checkpoints total. Adam's bias-corrected `exp_avg_sq` and the
clipping scale use all five saved checkpoints per segment and are averaged
together as an LR-weighted effective `c*p`. The query Jacobian is evaluated at
the final raw model on the cached EMA-generated DDIM trajectory.

The clipping replay is a frozen-scale approximation: it includes the estimated
`c = min(1, C / ||g_batch||)` in each segment decay but omits `dc/dtheta`, since
the original shuffled batch gradients were not saved.

```bash
python 20_verify_adam_clip_source_das.py
python -u 22_launch_adam_clip_source_das_4gpu.py
```

One run saves and evaluates two methods:

- `source_das_adam_clip_raw_11h50p_100t_mc10_unnormalized`
- `source_das_adam_clip_raw_11h50p_100t_mc10_jacobian_fro_rms`

`jacobian_fro_rms` computes the exact Jacobian Frobenius norm over the selected
attribution parameters and all 27 predicted-noise components, then divides all
components for that query/timestamp by the shared `||J||_F / sqrt(27)` value.
It never normalizes the SOURCE parameter delta.  Both methods square the
resulting 27 component effects, sum them, and average over 100 timestamps.
