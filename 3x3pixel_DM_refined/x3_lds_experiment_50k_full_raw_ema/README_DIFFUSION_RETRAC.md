# Replayed Diffusion-TracIn and Diffusion-ReTrac

This experiment implements the practical checkpoint form from Xie et al.,
*Data Attribution for Diffusion Models: Timestep-induced Bias in Influence
Estimation*.

For each of the 50 raw prompted checkpoints, the query side uses the simple
diffusion loss on generated endpoint queries, 100 evenly spaced diffusion
timesteps, and MC10 query noise.  The train side does **not** use query-aligned
noise: it replays the four exact `t_train` and `epsilon_train` events realized
for every training datapoint in the corresponding four-epoch checkpoint
interval.

Two scores are saved:

1. Diffusion-TracIn uses raw query and train loss gradients.
2. Diffusion-ReTrac first averages query gradients over MC at each timestep,
   L2-normalizes that timestep gradient, and averages over timesteps.  Each
   replayed training-event gradient is independently L2-normalized before the
   four events are averaged.

Both use the raw checkpoint parameters, the learning rate saved in that
checkpoint, and a shared CountSketch4096 map on the query/train sides of each
checkpoint.  The projection changes dot-product approximation, not the
normalization contract: ReTrac L2 norms are computed from full gradients before
projection.

Run from this experiment directory:

```bash
python -u 115_launch_diffusion_retrac_10q_4gpu.py \
  --batch-size 8 \
  --query-batch-size 2
```

The launcher verifies inputs, runs four checkpoint shards, resumes at completed
checkpoint boundaries, merges q00-q09 scores, and evaluates all eight existing
LDS response targets with both signs.

If the exact training-event replay cache is absent, first run:

```bash
python -u 24_prepare_forward_loss_alignment.py \
  --skip-queries \
  --skip-baseline
```

Outputs:

- `x3_lds_exp_50k/attribution/diffusion_tracin_replayed_raw_50ckpt_100t_mc10_projected4096/qXX/scores.npy`
- `x3_lds_exp_50k/attribution/diffusion_retrac_replayed_raw_50ckpt_100t_mc10_projected4096/qXX/scores.npy`
- `x3_lds_exp_50k/lds/diffusion_tracin_vs_retrac_replayed_10q.json`
- `x3_lds_exp_50k/logs/diffusion_retrac_10q_gpu{0,1,2,3}.log`
