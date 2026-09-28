# Endpoint noise-aligned TracIn-DAS

This experiment uses the original non-staged 50-checkpoint model bank and
prompted queries q00-q09.

For every `(checkpoint, timestamp)` pair, an independent deterministic noise
tensor is sampled. Within that pair, the noise is shared by every query
endpoint and every training example so the two gradient sides remain exactly
noise-aligned. At raw checkpoint `c`, the query output direction is

```text
normalize(epsilon_{c+1}(x_qt,t) - epsilon_c(x_qt,t)).
```

The scalar query objective is the current predicted noise projected onto this
direction. Its exact full-parameter gradient is dotted with each training
point's diffusion-loss gradient at the same timestamp and with the same noise.
No CountSketch parameter projection is used.

An optional `projected4096` mode applies one checkpoint-specific CountSketch
map to both query and training gradients before their parameter-space dot
product. It does not normalize either projected gradient.

Run on four GPUs:

```bash
python -u 75_launch_tracin_das_4gpu.py --batch-size 128
```

Alternatively, use one identical noise across all 49 checkpoint transitions
within each timestamp:

```bash
python -u 75_launch_tracin_das_4gpu.py \
  --noise-mode timestamp-shared --batch-size 128
```

This second mode still uses a different noise for every timestamp and still
shares the term noise between the query and training-loss sides.

Run the two 4096-dimensional projected variants with:

```bash
# Different noise for each checkpoint/timestamp pair.
python -u 75_launch_tracin_das_4gpu.py \
  --noise-mode checkpoint \
  --parameter-projection projected4096 \
  --batch-size 128

# One noise shared across checkpoints within each timestamp.
python -u 75_launch_tracin_das_4gpu.py \
  --noise-mode timestamp-shared \
  --parameter-projection projected4096 \
  --batch-size 128
```

The launcher automatically merges three contractions:

```text
tracin_das_endpoint_next_delta_checkpoint_noise_linear
tracin_das_endpoint_next_delta_checkpoint_noise_termwise_squared
tracin_das_endpoint_next_delta_checkpoint_noise_timestamp_sum_squared
```

Then evaluate all eight existing LDS response metrics and both score signs:

```bash
python -u 76_eval_tracin_das_lds.py
```

Evaluate the timestamp-shared variant with:

```bash
python -u 76_eval_tracin_das_lds.py --noise-mode timestamp-shared
```

Add `--parameter-projection projected4096` to evaluate either projected
variant.

## Independent MC10 training loss

To keep checkpoint/timestamp alignment but stop forcing the training loss to
reuse the query noise, add `--train-noise-mode independent-mc10`. Each training
point then receives ten independent noises, the ten losses are averaged, and
one gradient of that mean loss is computed. For example, the projected4096
timestamp-shared query-noise variant is:

```bash
python -u 75_launch_tracin_das_4gpu.py \
  --noise-mode timestamp-shared \
  --parameter-projection projected4096 \
  --train-noise-mode independent-mc10 \
  --batch-size 128
```

Evaluate it with the same three switches:

```bash
python -u 76_eval_tracin_das_lds.py \
  --noise-mode timestamp-shared \
  --parameter-projection projected4096 \
  --train-noise-mode independent-mc10
```

All aligned and independent-MC10 variants use separate shard, score, log, and
LDS-summary namespaces.

The LDS summary is saved at
`x3_lds_exp_50k/lds/tracin_das_endpoint_next_delta_checkpoint_noise_q00_q09.json`.

## Stable aligned projected4096 run on all 100 queries

The all-query launcher keeps the checkpoint-specific random noise and reuses
that exact `(timestamp, noise)` on the query and training-loss sides. It uses
the same checkpoint-specific CountSketch4096 map for both parameter gradients.
It runs prompted q00-q74 and unprompted q75-q99 as separate family banks on
four GPUs, without sharing or overwriting partial shards from the q00-q09 run:

```bash
python -u 81_launch_tracin_das_aligned_projected4096_100q_4gpu.py \
  --batch-size 128
python -u 82_eval_tracin_das_aligned_projected4096_100q_lds.py
```

The launcher saves all three contractions (`linear`, `termwise_squared`, and
`timestamp_sum_squared`) under their existing projected4096 method names for
q00-q99. The LDS summary is saved as
`x3_lds_exp_50k/lds/tracin_das_endpoint_next_delta_checkpoint_noise_projected4096_aligned_q00_q99.json`.

## q00-q98 interval-mean-LR checkpoint-count sweep

This variant keeps checkpoint-specific query noise, exact query/train loss-noise
alignment, and checkpoint-specific CountSketch4096 projection. Its transition
weight is the exact mean of every scheduled optimizer learning rate between the
two checkpoint global steps:

```text
mean(eta_k for global_step_c <= k < global_step_{c+1}) / 100 timestamps
```

One run simultaneously accumulates checkpoint-bank counts
`50,40,25,20,15,10,5`. The 50-checkpoint label means the complete 50-model
bank and therefore all 49 available next transitions. Every smaller bank uses
`round(linspace(0,48,count))`; all include source position 48, the penultimate
checkpoint whose target is the final checkpoint. The query bank is q00-q98.

Run attribution, merge, and LDS evaluation on four GPUs with:

```bash
python -u 84_launch_tracin_das_avg_pair_lr_99q_4gpu.py --batch-size 128
```

The launcher saves all three contractions for every checkpoint count and then
evaluates both LDS signs against four target families (`simple_loss`,
`traj_ref`, `endpoint_deviation`, and `trajectory_state_mse`), retaining both
EMA and raw observed variants. The combined summary is:

```text
x3_lds_exp_50k/lds/tracin_das_interval_mean_lr_99q_checkpoint_sweep.json
```

## Final-EMA DAS with aligned query/train noise (q00-q98)

The original DAS samples independent train-side MC10 noise inside each query
term. This comparison instead uses one shared noise draw for each
`(timestamp, MC)` term on both sides: every query endpoint and every training
point feature/residual use that same draw. Ten aligned terms per timestamp
still provide effective MC10, but there is no independent train-side MC loop.
Projection remains CountSketch4096 and all configured lambdas are saved.

Run four timestamp/family shards, merge, and evaluate LDS automatically:

```bash
python -u 88_launch_das_aligned_noise_99q_4gpu.py --batch-size 64
```

The method namespace is `das_ema_aligned_noise`. LDS covers the four target
families with both EMA/raw observed variants and both score signs:

```text
x3_lds_exp_50k/lds/das_ema_aligned_noise_99q_lambda_sweep.json
```

## q00-q98 timestamp-count sweep from 100 to 10

This sweep fixes the complete 50-model/49-transition bank and simultaneously
accumulates `100,90,80,...,10` evenly spaced timestamp budgets. Each budget
uses `round(linspace(0,99,count))`, so timestamp positions 0 and 99 are always
included. All other settings match the stable TracIn-DAS variant: raw next
checkpoint transitions, the source checkpoint's saved `eta`, checkpoint-specific
aligned noise, CountSketch4096, and q00-q98.

Run all three contractions, merge, and LDS automatically:

```bash
python -u 91_launch_tracin_das_timestamp_sweep_99q_4gpu.py --batch-size 500
```

The LDS summary covers four target families, EMA/raw observed variants, and
both score signs:

```text
x3_lds_exp_50k/lds/tracin_das_checkpoint_lr_99q_timestamp_sweep.json
```
