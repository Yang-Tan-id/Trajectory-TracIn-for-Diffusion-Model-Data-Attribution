# Endpoint noise-aligned TracIn-DAS

This experiment uses the original non-staged 50-checkpoint model bank and
prompted queries q00-q09.

For each of 100 timestamps, one deterministic noise tensor is shared by every
query endpoint and every training example. At raw checkpoint `c`, the query
output direction is

```text
normalize(epsilon_{c+1}(x_qt,t) - epsilon_c(x_qt,t)).
```

The scalar query objective is the current predicted noise projected onto this
direction. Its exact full-parameter gradient is dotted with each training
point's diffusion-loss gradient at the same timestamp and with the same noise.
No CountSketch parameter projection is used.

Run on four GPUs:

```bash
python -u 75_launch_tracin_das_4gpu.py --batch-size 128
```

The launcher automatically merges three contractions:

```text
tracin_das_endpoint_next_delta_linear
tracin_das_endpoint_next_delta_termwise_squared
tracin_das_endpoint_next_delta_timestamp_sum_squared
```

Then evaluate all eight existing LDS response metrics and both score signs:

```bash
python -u 76_eval_tracin_das_lds.py
```

The LDS summary is saved at
`x3_lds_exp_50k/lds/tracin_das_endpoint_next_delta_q00_q09.json`.
