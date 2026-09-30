# Trajectory Bundle TracIn

This experiment retains a vector-valued TracIn response instead of reducing
each training point to an additive scalar.

For checkpoint `c`, training point `i`, query `q`, and trajectory timestamp
`t`, it estimates

```text
g_i,c = grad_theta mean_{m=1..10} L_simple(i; random t_m, random epsilon_m)
a_i,q,c,t = -eta_c J_q,c,t g_i,c
```

The training `t/noise` draws are independent of the query trajectory. The
checkpoint responses are added as signed vectors. For an LDS subset `S`, the
score is

```text
Bundle(S, q) = mean_t || sum_{i in S} sum_c a_i,q,c,t ||_2^2.
```

The implementation uses one shared 4096-dimensional CountSketch for training
gradients and all 27 predicted-noise output-gradient rows. For LDS, membership
aggregation is performed in projected parameter space before the query matrix
multiplication. By linearity, this equals computing every projected pointwise
response and then adding the response vectors within each subset, without
materializing the multi-gigabyte pointwise tensor.

## Run q00-q09 on four GPUs

```bash
python 198_verify_bundle_tracin.py
python -u 201_launch_bundle_tracin_4gpu.py \
  --gpus 0,1,2,3 \
  --query-ids 0-9 \
  --batch-size 1280
```

The launcher is restartable at checkpoint-shard granularity. It saves:

```text
x3_lds_exp_50k/attribution/
  bundle_tracin_raw_mc10_independent_t_noise_projected4096_checkpoint_sum_timestamp_mean/
    qXX/bundle_vectors.npy   # [192, 100, 27]
    qXX/bundle_scores.npy    # [192]
    qXX/info.json
```

and evaluates both signs against all eight existing LDS targets. To run all
100 queries, use `--query-ids 0-99`; prompted and unprompted families run as
two sequential four-GPU phases.
