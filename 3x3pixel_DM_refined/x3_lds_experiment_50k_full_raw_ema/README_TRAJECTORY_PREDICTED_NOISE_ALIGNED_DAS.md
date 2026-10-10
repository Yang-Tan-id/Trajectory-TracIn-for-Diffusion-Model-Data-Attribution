# Trajectory-predicted-noise aligned DAS

For every cached non-endpoint query trajectory state `x_t^q`, the final EMA
model first predicts and detaches

```text
epsilon_qt = epsilon_theta(x_t^q, t, c_q).
```

Every training point is then noised to the same timestamp using that exact
query-predicted noise:

```text
x_it^q = sqrt(alpha_bar_t) x_i + sqrt(1-alpha_bar_t) epsilon_qt.
```

Its DAS feature and residual are evaluated at `x_it^q`, with `epsilon_qt` as
the residual target. Query and training use the same timestamp, target noise,
output probe, final EMA parameters, and CountSketch4096 map. The endpoint
snapshot at `t=0` is excluded. Ten shared Gaussian output probes are averaged
for each of the 99 trajectory states, and all configured DAS lambdas are
evaluated.

```bash
python -u 157_launch_trajectory_predicted_noise_aligned_das_4gpu.py \
  --gpus 0,1,2,3 \
  --feature-batch-size 64
```

The launcher merges scores and runs LDS automatically. Output:

```text
x3_lds_exp_50k/lds/trajectory_predicted_noise_aligned_das_q00_q09_lambda_sweep.json
```
