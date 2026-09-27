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

The LDS summary is saved at
`x3_lds_exp_50k/lds/tracin_das_endpoint_next_delta_checkpoint_noise_q00_q09.json`.
