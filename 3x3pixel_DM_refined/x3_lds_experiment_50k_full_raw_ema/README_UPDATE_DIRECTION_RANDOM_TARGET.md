# Update-direction gradient to random target directions

This analysis asks whether the single source noise direction actually used by a
fresh-SGD update can predict finite predicted-noise changes on a different
target datapoint polluted along 100 random directions.

For every one of the 40 existing source-update branches,

```text
10 source datapoints x 4 independent 250-timestamp update blocks,
```

the cached scalar

```text
|g_target(random direction r, t)^T Delta-theta-source|
```

is proportional to

```text
|g_target(random direction r, t)^T g_source(update +epsilon direction)|
```

because each branch is exactly one fresh-SGD step. The target is the finite
updated-minus-null predicted-noise L2 response on that random target direction.

No GPU recomputation is required after experiment 180 has completed:

```bash
python -u 187_eval_update_direction_to_random_target_response.py
```

The script reports prediction quality for fixed random directions across all
timesteps, fixed timesteps across random directions, pooled direction/timestep
pairs, all-t integrated direction scores, the 100-direction mean curve, and the
40 source-update branches. It writes:

```text
x3_lds_exp_50k/endpoint_direction_mc_fresh_sgd_100dir_1000t/update_direction_to_random_target_response.json
```
