# Trajectory-state inverse-noise TracIn-DAS

For q00-q09, each saved reference trajectory state `x_t^q` is treated as if it
were produced by forward noising every training point `x_i` at the same
diffusion level. The corresponding deterministic noise target is inverted from
the forward equation:

```text
epsilon*_(i,q,t) =
    (x_t^q - sqrt(alpha_bar_t) x_i) / sqrt(1 - alpha_bar_t).
```

The query-dependent training loss is evaluated directly at `x_t^q` using the
training point's condition:

```text
L*_(i,q,t)(theta_c) =
    mean((epsilon_theta_c(x_t^q, t, c_i) - epsilon*_(i,q,t))^2).
```

The query side is the normalized current-to-next-checkpoint predicted-noise
delta direction at the same `x_t^q`. Query and training gradients share one
CountSketch4096 map per checkpoint. The final endpoint index 99 (`t=0`) is
excluded, and indices 0..98 are averaged with weight `1/99`. Linear,
termwise-square, and timestamp-sum-square contractions are saved.

Because every training loss depends on the query state, this experiment is
roughly ten times more expensive than a ten-query bank whose training features
can be shared. It checkpoints after every completed timestamp.

```bash
python -u 138_launch_trajectory_inverse_noise_tracin_das_10q_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 64
```

The launcher merges shards and evaluates all LDS targets automatically. Output:

```text
x3_lds_exp_50k/lds/trajectory_inverse_noise_tracin_das_q00_q09.json
```
