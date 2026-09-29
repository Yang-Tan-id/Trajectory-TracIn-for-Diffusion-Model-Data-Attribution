# Fully-unrolled, higher-noise-aligned trajectory DAS

For q00-q09, differentiate through the complete deterministic 1000-step DDIM
sampler at the final prompted EMA parameters. At each of 100 saved reference
trajectory states, 27 projected RGB output-basis gradients are cached (3 color
channels x 3 x 3 pixels). By linearity, combining those 27 gradients with the
current loss-side probe is exactly equal to directly differentiating the scalar
`<x_t, probe> / sqrt(27)`;
this avoids repeating the full unroll for every loss term. The full chain rule
through all preceding denoising steps is retained.

For a trajectory state at diffusion level `t`, only training losses at levels
`s >= t` are paired with that state. For example, the state near `t=888` is
matched to training-loss timestamps from 888 through 999. Its responses are
averaged over those higher-noise timestamps and MC10; the resulting 100
per-state responses are then averaged.

For each `(s, m)` term, one Gaussian output probe is sampled. Exactly that same
probe is used to construct the training gradient, the residual, and every
aligned query-state VJP. There is no independent query probe4 average:

```text
(1 / 100) sum_t (1 / |S(t)|) sum_{s in S(t)} (1 / 10)
    sum_{m=1}^{10}
    [r_{i,s,m} phi_{i,s,m}^T (G_{s,m} + lambda I)^-1 psi_{q,t,s,m}]^2,
S(t) = {s in the 100 DAS timestamps : s >= t}.
```

The first cached state (the fixed initial noise) contributes zero parameter
gradient naturally. Query output-basis gradients and every training-loss feature use the
same global 4096-dimensional CountSketch. Query features are not L2 normalized,
preserving trajectory-Jacobian magnitude; training features retain the normal
DAS normalization. Each training gradient internally averages MC10 noise draws,
and each timestamp also has 10 outer DAS Monte Carlo terms. All configured
damping lambdas and all LDS targets are evaluated.

The launcher shards q00-q09 across all four requested GPUs during the unroll
cache phase, merges the four query shards, and then reuses the same four GPUs
for DAS timestamp/MC shards. Completed query shards are restartable.

```bash
python -u 132_launch_unrolled_traj_das_10q_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 128
python -u 133_eval_unrolled_traj_das_10q_lds.py
```
