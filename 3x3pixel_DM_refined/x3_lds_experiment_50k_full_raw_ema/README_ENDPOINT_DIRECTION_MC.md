# Endpoint multi-direction response experiment

This experiment reuses the 40 saved one-step fresh-SGD branches. For every
paired target endpoint and all 1000 diffusion noise levels, it forward-pollutes
the endpoint using 100 known Gaussian directions. It measures:

- the true finite updated-minus-null predicted-noise squared L2 response;
- the exact JVP using the saved parameter delta;
- the absolute first-order diffusion-loss change for each direction.

The 100-direction finite response is the Monte Carlo ground truth. The analysis
tests whether one direction predicts the mean of the other 99 and sweeps
direction counts `1,2,4,8,10,20,50,100`. It reports correlation across noise
levels, relative error, and a false-small rate.

The analysis also quantifies finite-response L2 heterogeneity across the 100
directions: fixed-t coefficient of variation, q90/q10 spread, direction-wise
mean variation, direction-to-mean and pairwise curve correlations, and the
fractions below 0.5x or above 1.5x the fixed-t direction mean.

```bash
python 178_verify_endpoint_direction_mc.py
python -u 180_launch_endpoint_direction_mc_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 5120
```

Results can be re-analyzed without GPU work:

```bash
python -u 180_launch_endpoint_direction_mc_4gpu.py --analyze-only
```

For a stricter assessment of whether direction variation is genuinely small,
including tolerance coverage, 95%/99% deviation bands, worst-timestep behavior,
source-cluster bootstrap confidence intervals, and separate L2/squared-L2
results, run:

```bash
python -u 184_quantify_endpoint_direction_variation.py
```

Summary:

```text
x3_lds_exp_50k/endpoint_direction_mc_fresh_sgd_100dir_1000t/summary.json
```
