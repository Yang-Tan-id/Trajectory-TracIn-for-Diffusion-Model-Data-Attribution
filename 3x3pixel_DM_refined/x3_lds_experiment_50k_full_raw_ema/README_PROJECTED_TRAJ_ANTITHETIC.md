# Projected Traj TracIn with antithetic training noise

This q00-q09 experiment retains the existing first-order raw projected Traj
TracIn query side: query gradients are evaluated along the cached reference
trajectory and target the next checkpoint's predicted noise. Query and training
sides share only the diffusion timestep; their noise directions are not aligned.

For every training datapoint and aligned timestep, ten independent base noises
are sampled. Each is paired with its negative. The mean of all twenty diffusion
loss terms is differentiated once:

```text
g_i = grad mean_m [L_i(+epsilon_m) + L_i(-epsilon_m)] / 2
```

The same global CountSketch-4096 construction and checkpoint LR weighting are
used. Linear, termwise-square, and timestamp-sum-square scores are saved.

```bash
python -u 164_launch_projected_traj_antithetic10pairs_10q_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 16
```

The launcher merges all four timestamp shards and runs LDS automatically. The
summary is written to:

```text
x3_lds_exp_50k/lds/projected_traj_antithetic10pairs_q00_q09.json
```
