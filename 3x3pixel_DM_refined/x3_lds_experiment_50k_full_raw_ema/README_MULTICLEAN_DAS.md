# Ten-anchor triangular predicted-clean aligned DAS

For q00-q99, select ten evenly spaced snapshots in increasing diffusion-time
order (approximately `t=0,111,...,888,999`) from each 100-state cached final-EMA
reference trajectory. At each anchor `k`, make one predicted clean
image from the current predicted noise:

```text
x0_hat[k] = (x_k - sqrt(1-alpha_bar[k]) * eps_ema(x_k,k))
             / sqrt(alpha_bar[k])
```

Anchor `j=1..10` runs aligned final-EMA DAS on the first `10*j` of the 100 DAS
timestamps with MC10. Thus the `t≈888` ninth anchor uses 90 timestamps and the
last `t≈999` anchor uses all 100. Each anchor is averaged over its own timestamp
count and MC10, then the ten squared DAS scores are summed. For a fixed
`(DAS timestamp, MC)` term, the projected training features, Gram matrix,
residuals, and linear solves are computed once and reused by every eligible
anchor. All configured damping lambdas are retained.

```bash
python -u 127_launch_multiclean_aligned_das_100q_4gpu.py \
  --gpus 4,5,6,7 \
  --batch-size 2560
python -u 128_eval_multiclean_aligned_das_100q_lds.py
```
