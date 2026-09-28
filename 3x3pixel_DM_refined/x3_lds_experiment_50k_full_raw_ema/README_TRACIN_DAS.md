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
