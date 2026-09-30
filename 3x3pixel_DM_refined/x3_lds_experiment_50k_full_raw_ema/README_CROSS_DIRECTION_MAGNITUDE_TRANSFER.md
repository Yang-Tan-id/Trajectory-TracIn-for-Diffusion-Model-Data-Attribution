# Cross-direction finite-difference magnitude transfer

This diagnostic asks whether a one-sided fresh-SGD update that produces a large
predicted-noise change on its trained source axis also produces large changes
elsewhere. It reuses the 40 saved fresh-SGD branches and performs no retraining.

For every source and timestamp block it measures exact finite updated-minus-null
output changes on:

1. the source datapoint and trained `+epsilon` axis (the reference magnitude),
2. the same source datapoint and `-epsilon` axis,
3. a different target datapoint, its own prompt, and five independent axes.

It reports Pearson/Spearman magnitude correlations, other/original magnitude
ratios, the fraction of branches where the other direction is smaller, and
within-branch correlations across all 1000 timestamps. It also computes
cross-source correlations separately at each fixed timestamp, which controls
for magnitude patterns caused only by the common diffusion schedule.

```bash
python 175_verify_cross_direction_magnitude_transfer.py
python -u 177_launch_cross_direction_magnitude_transfer_4gpu.py --gpus 0,1,2,3
```

Summary:

```text
x3_lds_exp_50k/null_tblock_cross_direction_magnitude_transfer_fresh_sgd/summary.json
```
