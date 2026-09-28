# Ten-anchor predicted-clean aligned DAS

For q00-q09, select ten evenly spaced snapshots from each 100-state cached
final-EMA reference trajectory. At each anchor `k`, make one predicted clean
image from the current predicted noise:

```text
x0_hat[k] = (x_k - sqrt(1-alpha_bar[k]) * eps_ema(x_k,k))
             / sqrt(alpha_bar[k])
```

Run aligned final-EMA DAS with 100 DAS timestamps and MC10 on all ten clean
estimates. For a fixed `(DAS timestamp, MC)` term, the projected training
features, Gram matrix, residuals, and linear solves are computed once and
reused for all ten clean anchors. The final attribution is the sum of the ten
per-anchor squared DAS scores. All configured damping lambdas are retained.

```bash
python -u 109_launch_multiclean_aligned_das_10q_4gpu.py --batch-size 64
python -u 110_eval_multiclean_aligned_das_10q_lds.py
```
