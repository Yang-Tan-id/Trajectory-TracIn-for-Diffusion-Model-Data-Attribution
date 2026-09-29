# Fully-unrolled, higher-noise-aligned trajectory DAS

For q00-q09, differentiate through the complete deterministic 1000-step DDIM
sampler at the final prompted EMA parameters. At each of 100 saved reference
trajectory states, four independent Rademacher probes produce a projected
fully-unrolled state-Jacobian feature. This keeps the full chain rule through
all preceding denoising steps.

For a trajectory state at diffusion level `t`, only training losses at levels
`s >= t` are paired with that state. For example, the state near `t=888` is
matched to training-loss timestamps from 888 through 999. Its responses are
averaged over those higher-noise timestamps and MC10; the resulting 100
per-state responses are then averaged. In symbols, the contraction is

```text
(1 / 100) sum_t (1 / |S(t)|) sum_{s in S(t)} (1 / (10 * 4))
    sum_{m=1}^{10} sum_{p=1}^{4}
    [r_{i,s,m} phi_{i,s,m}^T (G_{s,m} + lambda I)^-1 psi_{q,t,p}]^2,
S(t) = {s in the 100 DAS timestamps : s >= t}.
```

The first cached state (the fixed initial noise) contributes zero parameter
gradient naturally. Every state/probe and every training-loss feature use the
same global 4096-dimensional CountSketch. Query features are not L2 normalized,
preserving trajectory-Jacobian magnitude; training features retain the normal
DAS normalization. Each training gradient internally averages MC10 noise draws,
and each timestamp also has 10 outer DAS Monte Carlo terms. All configured
damping lambdas and all LDS targets are evaluated.

```bash
python -u 132_launch_unrolled_traj_das_10q_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 128
python -u 133_eval_unrolled_traj_das_10q_lds.py
```
