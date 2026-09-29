# Timestamp-diagonal predicted-clean aligned DAS

For each of the 100 cached reference-trajectory snapshots, this experiment uses
the final EMA model's current predicted noise to form a clean endpoint estimate:

```text
x0_hat_k = (x_k - sqrt(1-alpha_bar_k) * eps_ema(x_k,k)) / sqrt(alpha_bar_k)
```

Unlike the ten-anchor Cartesian experiment, endpoint `x0_hat_k` is used only at
its own diffusion noise level `k`.  Query and train sides use the same noise and
output probe for that `(k, MC)` term.  The final attribution is the mean of the
100 timestamp by MC10 squared DAS terms.  CountSketch4096, projected-gradient
normalization, final EMA parameters, and the full lambda sweep are retained.

Run q00-q99 on GPUs 4-7 with the requested feature batch size:

```bash
python -u 125_launch_diagonal_clean_aligned_das_100q_4gpu.py \
  --gpus 4,5,6,7 \
  --batch-size 2560
```

Then evaluate LDS:

```bash
python -u 126_eval_diagonal_clean_aligned_das_100q_lds.py
```

The LDS sweep is saved to:

```text
x3_lds_exp_50k/lds/diagonal_clean_aligned_das_100q_100timestamp_lambda_sweep.json
```

For the cheaper ten-timestamp one-to-one version, run:

```bash
python -u 127_launch_diagonal_clean_aligned_das_10timestamp_100q_4gpu.py \
  --gpus 4,5,6,7 \
  --batch-size 2560

python -u 128_eval_diagonal_clean_aligned_das_10timestamp_100q_lds.py
```
