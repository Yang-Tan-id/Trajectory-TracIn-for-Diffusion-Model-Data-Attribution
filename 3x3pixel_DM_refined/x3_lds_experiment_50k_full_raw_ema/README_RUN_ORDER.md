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

Curvature, diagonal-Fisher, and SOURCE score passes use compute batch 512.
Clipping coefficients are estimated through a separate loader at the original
training batch size 256, so increasing compute throughput does not change the
definition of `c = min(1, C / ||g_batch||)`.

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

## 50-checkpoint forward-loss alignment (q00-q49)

This experiment replays the four realized training events for every datapoint
inside each four-epoch checkpoint interval.  At raw checkpoint `c`, it first
uses all 1000 states of the existing final-EMA reference trajectory to form the
noise-imitation loss

```text
L_ref(theta_c) = mean_k ||eps_theta_c(x_k, t_k, cond_q)
                              - eps_final_EMA(x_k, t_k, cond_q)||^2.
```

It evaluates two artificial one-step updates:

```text
raw:        theta+ = theta_c - lr_c * grad L_ref
normalized: theta+ = theta_c - lr_c * grad L_ref / ||grad L_ref||_2
```

For each datapoint and checkpoint, the score contribution is its mean loss
reduction over the exact four shuffled `(t, noise)` training events:

```text
mean_event [loss(theta_c) - loss(theta+)].
```

For each of the two update rules, the same forward pass produces three scores:

- `absolute`: mean `loss_before - loss_after` over four events;
- `log_relative`: mean `log((loss_before + eps) / (loss_after + eps))`;
- `loss_conditioned_robust`: split datapoints into 20 equal-count bins by
  baseline loss, robust-standardize the log-relative score with median/MAD
  inside each bin, and clip to `[-5, 5]`.

The final score averages each contribution over all 50 checkpoints. Parameters
remain raw; the trajectory and predicted-noise targets come from the final EMA
model. LDS evaluates both score signs on the existing subset models and
observations. Event-level baseline losses are retained so the three scores do
not require three model runs.

Run preparation once on one GPU.  The replay cache is about 1.1 GB because the
original float32 noise is retained:

```bash
python 23_verify_forward_loss_alignment.py
python -u 24_prepare_forward_loss_alignment.py --gpu 0 \
  2>&1 | tee x3_lds_exp_50k/logs/forward_loss_alignment_prepare.log
```

Then run the 50 queries on exactly four GPUs.  The launcher automatically runs
LDS after all four shards succeed and can resume each query from its last
completed checkpoint:

```bash
python -u 27_launch_forward_loss_alignment_4gpu.py \
  2>&1 | tee x3_lds_exp_50k/logs/forward_loss_alignment_4gpu.log
```

The six attribution methods are the Cartesian product of:

- update: `raw_sgd` or `normalized_sgd`;
- normalization: the base method name (`absolute`), suffix `_log_relative`, or
  suffix `_loss_conditioned_robust`.

If preparation was run with the earlier absolute-only implementation, rerun
`24_prepare_forward_loss_alignment.py`. Existing replay/query caches are kept;
only the missing event-level baseline cache is generated. The scorer uses a
new v2 partial file, so an old absolute-only partial cannot be mixed into these
six results.

## Four-step trajectory unlearning (q00-q49)

This is a separate experiment and does not overwrite one-step learning. At
every raw checkpoint `c`, it performs four trajectory-gradient ascent steps.
Each step recomputes the mean gradient over all 1000 fixed reference states at
the already-updated parameters. The reference states and final-EMA
predicted-noise targets remain fixed.

The four learning rates are the final real optimizer-step LR from each epoch,
used in reverse chronological order. For checkpoint 200, the order is:

```text
eta_(199->200), eta_(198->199), eta_(197->198), eta_(196->197)
```

This is not the LR saved at checkpoint 196. All values are reconstructed from
the original warmup/cosine schedule. The experiment uses trajectory-SGD rather
than the checkpoint AdamW moments, because historical first moments contain
training-data directions and negating a gradient through AdamW is not an exact
optimizer reversal.

After four steps, each datapoint's four realized events from the matching
four-epoch interval are evaluated. Positive score means loss growth after
unlearning. Four step-size multipliers are evaluated:

```text
alpha = 1, 1/4, 1/16, 1/64
```

Every alpha follows its own nonlinear four-step path and recomputes the
1000-timestamp gradient after each update; this is not a post-hoc rescaling of
the alpha-1 parameter delta. The two update forms (raw and
global-gradient-normalized), four alphas, and three score normalizations
(absolute, log-relative, loss-conditioned robust) produce 24 methods.

It reuses all caches from script 24:

```bash
python -u 30_launch_trajectory_unlearning_4step_4gpu.py \
  2>&1 | tee x3_lds_exp_50k/logs/trajectory_unlearning_4step_4gpu.log
```

The launcher runs LDS automatically. The 24-method combined result is:

```text
x3_lds_exp_50k/lds/trajectory_unlearning_4step_all_normalizations_q00_q49.json
```

The scorer uses its own checkpoint-resumable partials under
`forward_loss_alignment/partials/trajectory_unlearning_4step/`. Alpha 1 keeps
the original artifact names; smaller alphas add suffixes such as
`_alpha_0p25`, `_alpha_0p0625`, and `_alpha_0p015625`. The alpha sweep uses a
new v2 partial file, so earlier alpha-1-only progress is not mixed in.

## q00-q09 top-1000 removal: normalized unlearning alpha=.25 vs DAS

This comparison uses the first ten queries from the same original 100-query
bank. All ten are prompted, and their saved initial noise and prompt are reused.
For each query it removes the 1000 largest saved scores from:

- `trajectory_unlearning_normalized_sgd_4step_50ckpt_1000t_4event_alpha_0p25`
  (absolute loss-growth score);
- `das_ema/lambda_10p0`.

No LDS sign is applied to the ranking. Each 49k model is retrained from the
same training seed for 200 epochs, then its final EMA trajectory is compared
with the saved base-EMA query trajectory. This is 10 queries x 2 methods = 20
models on four GPUs. Training has 10-epoch resume checkpoints.

```bash
python -u 32_launch_unlearning_vs_das_topk_removal_4gpu.py \
  2>&1 | tee x3_lds_exp_50k/logs/topk_unlearning_alpha_0p25_vs_das.log
```

Artifacts and the paired trajectory/endpoint comparison are saved under:

```text
x3_lds_exp_50k/topk_removal_unlearning_alpha_0p25_vs_das_q00_q09/
```

The main outputs are `summary.json`, `per_query_results.csv`,
`paired_comparison.csv`, and `method_overlap.json`. In the paired summary,
`right_minus_left` is unlearning minus DAS because the method tags sort as
`das_...` then `unlearning_...`.

## Endpoint-MC100 joint FT+GA MUCS score (q00-q09)

This pilot starts from the epoch-200 raw model and treats the epoch-4 raw model
as the null model. For each query's saved final-EMA endpoint, it fixes 100
`(t, noise)` draws and measures endpoint diffusion loss under both models. The
stopping target recovers 95% of the final-to-null loss gap:

```text
target = L_final + 0.95 * (L_null - L_final)
```

The query-specific `F2` copies final `F1`'s raw parameters, then creates a
fresh AdamW optimizer (`m=0`, `v=0`, step 0). Every continuation step draws
one deterministically shuffled retain batch containing 100 distinct training
datapoints, with one diffusion draw per datapoint, and optimizes the true joint
objective:

```text
L_joint = L_FT - lambda * L_GA, lambda=1
```

`L_FT` is the normal prompted diffusion training loss. `L_GA` uses 100 fixed
diffusion draws of the same query endpoint and caps each draw's loss at the
corresponding epoch-4 null-model loss. Both terms participate in one backward
and one AdamW step. The run keeps the original AdamW hyperparameters and global
clip norm 1, and replaces the near-zero final scheduled LR with the requested
constant `0.1 * PEAK_LR = 1e-5`. It stops immediately when the uncapped mean
endpoint loss reaches the target. If `L_null <= L_final`, the job fails
explicitly because ascent toward the null loss is not defined by this
criterion.

The older method name
`mucs_endpoint_mc100_adamw_lr0p1_nullgap95_raw_q00_q09` is reserved for the
GA-only ablation and is not overwritten. The launcher now writes the true
joint method
`mucs_joint_ft_ga_endpoint_mc100_retain100_freshadamw_lambda1_lr0p1_nullgap95_raw_q00_q09`.

After stopping, every one of the 50,000 training points is evaluated with 100
paired MC draws under the original final raw model `theta` and the unlearned
model `theta_prime`. The saved score is exactly:

```text
mean_m ((L_i,m(theta_prime) - L_i,m(theta)) /
        (L_i,m(theta_prime) + L_i,m(theta) + eps))
```

The training-point condition is its own dataset condition; only the endpoint
unlearning objective uses the query prompt. Run on four GPUs:

```bash
python -u 35_launch_mucs_endpoint_unlearning_4gpu.py \
  2>&1 | tee x3_lds_exp_50k/logs/mucs_endpoint_unlearning_4gpu.log
```

AdamW unlearning checkpoints/history are saved under
`x3_lds_exp_50k/mucs_endpoint_unlearning/`. Scores, baseline MC100 means, and
unlearned MC100 means are saved under the normal attribution directory. LDS is
run automatically for q00-q09 and both score signs.

The joint MUCS update starts from the epoch-200 raw parameters but initializes
a fresh AdamW state. Each update uses 100 distinct retain datapoints with one
diffusion draw each, plus 100 fixed diffusion draws of the same generated
endpoint. Its objective is `L_FT - lambda * L_GA`; each per-draw query loss is
capped by the corresponding epoch-4 null-model loss. The learning rate is
`PEAK_LR / 10 = 1e-5` and the toy model's original AdamW hyperparameters are
retained.

After MUCS LDS, the launcher also compares against original `DAS EMA,
lambda=10` on exactly q00-q09. It reports both LDS signs for both methods,
marks the existing DAS convention `-(membership @ score)` as canonical, and
computes direct 50k-score Spearman plus top-1000 overlap. Outputs:

```text
x3_lds_exp_50k/lds/joint_mucs_vs_original_das_lambda10_q00_q09.json
x3_lds_exp_50k/lds/joint_mucs_vs_original_das_lambda10_q00_q09.csv
x3_lds_exp_50k/lds/joint_mucs_vs_original_das_lambda10_q00_q09_score_overlap.csv
```

The terminal and score-overlap CSV report each query's overlap separately.
Top-1000 means the 1,000 largest saved scores from each method; it does not
apply the LDS-only DAS sign convention.

If MUCS scores already exist, the comparison alone can be rerun with:

```bash
python 36_compare_mucs_vs_original_das.py
```

## Joint MUCS vs original DAS top-1000 removal (q00-q09)

This launches 20 independent 49k-example retraining jobs: ten remove each
query's 1,000 largest joint-MUCS saved scores, and ten remove each query's
1,000 largest original DAS EMA lambda-10 saved scores. No LDS sign is applied
to the ranking. All jobs use the same training seed and full original 3x3
training recipe. Evaluation uses each query's identical cached `x_T`, prompt,
and base EMA reference trajectory; removal models are evaluated with EMA.

Run on exactly four GPUs with:

```bash
python -u 38_launch_mucs_vs_das_topk_removal_4gpu.py \
  --gpus 0,1,2,3 2>&1 | \
  tee x3_lds_exp_50k/logs/topk_removal_joint_mucs_vs_das_4gpu.log
```

The launcher is resumable and automatically prepares, trains, evaluates, and
summarizes the 20 jobs. Outputs are under:

```text
x3_lds_exp_50k/topk_removal_joint_mucs_vs_original_das_q00_q09/
```

The main comparison files are `per_query_results.csv`,
`paired_comparison.csv`, and `summary.json`. Since method tags sort as DAS then
MUCS, each paired `delta_*` is MUCS minus DAS; positive values mean MUCS
removal caused a larger trajectory or endpoint change.

## First-order raw timestamp-sum-squared top-1000 removal (q00-q09)

The LDS result
`traj_projected_first_raw_timestamp_sum_squared_traj_ref_raw.json` maps to the
50k-dimensional attribution files under
`attribution/traj_projected_first_raw_timestamp_sum_squared/qXX/scores.npy`.
The following command removes the 1,000 largest saved scores for q00-q09 and
trains ten 49k-example models on four GPUs:

```bash
python -u 41_launch_timestamp_square_topk_removal_4gpu.py \
  --gpus 0,1,2,3 2>&1 | \
  tee x3_lds_exp_50k/logs/topk_removal_timestamp_square_4gpu.log
```

Outputs are stored under:

```text
x3_lds_exp_50k/topk_removal_traj_first_raw_timestamp_square_q00_q09/
```

The launcher writes `per_query_results.csv` and `summary.json`. If all ten DAS
removal evaluations from the preceding joint-MUCS-vs-DAS experiment exist, it
also writes `paired_with_existing_das.csv` without retraining those identical
DAS models. Its deltas are timestamp-square minus DAS.

## Twelve-probe Traj contractions with query-side L2 (q00-q19)

This experiment uses the first 49 raw checkpoints, all 100 cached reference
trajectory timestamps, train-gradient MC10, and a 4096-D CountSketch parameter
projection. At timestamp `t`, twelve Gaussian output probes are shared across
all checkpoints and queries. For probe `r`, the query feature is the projected
parameter gradient of

```text
<epsilon_theta(x_q,t), v_t,r> / sqrt(27).
```

Let `z[c,t,r,i]` be its dot product with training point `i`'s aligned projected
loss gradient. Four scores are produced:

```text
termwise raw:
  sum_(t,c) eta_c/100 * mean_r z[c,t,r,i]^2

termwise query-L2:
  the same after dividing each query feature by its exact full-parameter L2 norm

timestamp-wise raw:
  sum_t mean_r (sum_c eta_c/100 * z[c,t,r,i])^2

timestamp-wise query-L2:
  the same checkpoint-before-square contraction using query-L2 features
```

The query L2 norm is computed from the full parameter gradient before
CountSketch; the training gradient is never normalized.

The twelve probes estimate output-direction energy; they are not a projection
dimension of 12. The parameter projection remains 4096-D. Every method saves
both a final `(50000,)` `scores.npy` and a `(100,50000)`
`per_timestamp_scores.npy` for each query.

Verify and run on four GPUs:

```bash
python 42_verify_traj_probe12.py

python -u 46_launch_traj_probe12_20q_4gpu.py \
  --gpus 0,1,2,3 --batch-size 128
```

The four method namespaces are:

```text
traj_probe12_first_raw_termwise_squared
traj_probe12_first_raw_termwise_squared_query_l2
traj_probe12_first_raw_timestamp_sum_squared
traj_probe12_first_raw_timestamp_sum_squared_query_l2
```

The launcher is timestamp-level resumable, merges the per-timestamp artifacts,
and evaluates q00-q19 with both score signs on all existing LDS targets. LDS
outputs are:

```text
x3_lds_exp_50k/lds/traj_probe12_q00_q19_both_signs.json
x3_lds_exp_50k/lds/traj_probe12_q00_q19_both_signs.csv
```

## Previous-checkpoint Traj and signed next/previous alpha sweep

The original forward methods use checkpoint pairs `(c,c+1)` for the first 49
checkpoints, so the final checkpoint is excluded because it has no next model.
The backward/previous methods use `(c,c-1)` for checkpoints 2 through 50, so
they exclude the first checkpoint and include the final checkpoint. Previous
scores are saved under:

```text
traj_projected_backward_first_raw_termwise_squared
traj_projected_backward_first_raw_timestamp_sum_squared
```

After previous scoring, fixed global alphas are evaluated with next as the
base for `linear`, `termwise_squared`, and `timestamp_sum_squared`:

```text
combined(alpha) = (1-alpha) * next + alpha * previous
alpha = -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1
```

Positive alpha between zero and one is a convex interpolation. Negative alpha
is an affine extrapolation that subtracts the previous direction. Alpha is
shared by all queries; it is never selected separately per query. Each alpha's
50k attribution is saved as float32 under a method namespace such as:

```text
traj_next_previous_affine_timestamp_sum_squared_alpha_p0p25/q00/scores.npy
traj_next_previous_affine_timestamp_sum_squared_alpha_m0p25/q00/scores.npy
```

Run previous scoring and the alpha sweep on four GPUs:

```bash
python -u 48_launch_previous_and_alpha_4gpu.py
```

The sweep uses the existing LDS convention
`-(membership @ combined_score)`, evaluates all 100 queries and all existing
LDS targets, and reports positive-alpha versus negative-alpha improvement over
the next-only alpha-zero baseline. The primary printed target is
`traj_ref_raw`, matching
`traj_projected_first_raw_timestamp_sum_squared_traj_ref_raw.json`. Outputs:

```text
x3_lds_exp_50k/lds/traj_next_previous_alpha_sweep.json
x3_lds_exp_50k/lds/traj_next_previous_alpha_sweep_per_query.csv
x3_lds_exp_50k/lds/traj_next_previous_alpha_selection_traj_ref_raw.csv
```

For each contraction, the primary `traj_ref_raw` result additionally saves:

```text
traj_next_previous_affine_<contraction>_global_best_traj_ref_raw/qXX/scores.npy
traj_next_previous_affine_<contraction>_per_query_best_traj_ref_raw/qXX/scores.npy
```

`global_best` uses one alpha shared by all 100 queries and maximizes their mean
LDS. `per_query_best` selects alpha separately for every query on the same LDS
target and is therefore explicitly an oracle diagnostic rather than a fair
held-out hyperparameter result.
