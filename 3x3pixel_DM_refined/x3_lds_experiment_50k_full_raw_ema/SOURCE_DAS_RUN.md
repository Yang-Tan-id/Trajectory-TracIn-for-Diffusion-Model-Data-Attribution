# Timestamp-aligned SOURCE-DAS

This experiment reuses the trained X3 base checkpoints, 100 saved EMA query
trajectories, 50k training points, 192 LDS masks, and existing observed LDS
responses.

For each saved trajectory timestamp `t`, each training loss and EK-FAC
curvature estimate uses that exact diffusion timestamp and ten aligned noise
samples. Ten raw checkpoints (`epoch 20, 40, ..., 200`) define ten SOURCE
segments. Each segment contains 3920 optimizer iterations and uses the exact
mean learning rate from the original warmup/cosine schedule.

For datapoint `i`, query `q`, and timestamp `t`, the score contribution is

```text
delta_theta_source[i,t] = sum_l product_{l'>l}(S[l',t]) r[l,i,t]
response[i,q,t] = J(final_ema predicted_noise at query trajectory [q,t])
                  @ delta_theta_source[i,t]
score[i,q] = mean_t ||response[i,q,t]||_2^2
```

The predicted-noise output has 27 components, and the implementation computes
all 27 SOURCE linear responses and sums their squares. This is exactly the JVP
norm squared; no output probe or parameter projection is used.

SOURCE's current matrix-function implementation supports `Linear` and
`Conv2d`, so GroupNorm parameters are excluded. SOURCE curvature and residuals
use raw checkpoints; the query Jacobian uses final EMA parameters. As requested,
the AdamW preconditioner is ignored and SOURCE uses its SGD-style dynamics.

Server layout defaults to sibling repositories:

```text
/home/yt7447/diffusion_model/simple-influence
/home/yt7447/diffusion_model/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution
```

Override the first path with `SIMPLE_INFLUENCE_ROOT` if necessary.

Run:

```bash
python 17_verify_source_das.py
python -u 19_launch_source_das_4gpu.py
```

Detailed log:

```text
x3_lds_exp_50k/logs/source_das_10ckpt_100q_100t_mc10_4gpu.log
```

Final scores and LDS:

```text
x3_lds_exp_50k/attribution/source_das_raw_to_ema_10ckpt_100t_mc10_exact/qXX/scores.npy
x3_lds_exp_50k/lds/source_das_raw_to_ema_10ckpt_100t_mc10_exact_both_signs.json
```
