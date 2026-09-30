# Trajectory Bundle TracIn

This experiment retains a vector-valued TracIn response instead of reducing
each training point to an additive scalar.

For checkpoint `c`, training point `i`, query `q`, and trajectory timestamp
`t`, it estimates

```text
g_i,c = (1/4) sum_{e=1..4} grad_theta L_simple(i; t_train_i,c,e, epsilon_train_i,c,e)
a_i,q,c,t = -eta_c J_q,c,t g_i,c
```

The four `t/noise` pairs are the events actually realized for that datapoint
during the four epochs represented by checkpoint interval `c`. They come from
the exact training-event replay cache; they are not newly sampled MC10 draws
and are not forced to match the query trajectory. The checkpoint responses are
added as signed vectors. For an LDS subset `S`, the score is

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
python -u 24_prepare_forward_loss_alignment.py --skip-queries --skip-baseline
python 198_verify_bundle_tracin.py
python -u 201_launch_bundle_tracin_4gpu.py \
  --gpus 0,1,2,3 \
  --query-ids 0-9 \
  --batch-size 128
```

The preparation command is only needed when the exact replay cache is absent.
The worker flattens four replayed events per datapoint before projection, so a
datapoint batch of 128 creates an effective gradient batch of 512. Start at
128 on a 24-GB A5000; if utilization and memory allow, increase gradually.

The launcher is restartable at checkpoint-shard granularity. It saves:

```text
x3_lds_exp_50k/attribution/
  bundle_tracin_raw_replayed4event_projected4096_checkpoint_sum_timestamp_mean/
    qXX/bundle_vectors.npy   # [192, 100, 27]
    qXX/bundle_scores.npy    # [192]
    qXX/info.json
```

and evaluates both signs against all eight existing LDS targets. To run all
100 queries, use `--query-ids 0-99`; prompted and unprompted families run as
two sequential four-GPU phases.

## Full-model-centered post-hoc LDS

The observed trajectory-reference metrics compare each subset model with the
full-data model. After the main run, evaluate the corresponding complement
bundle without rerunning attribution:

```bash
python -u 202_eval_bundle_tracin_complement_lds.py --query-ids 0-9
```

It estimates `A_D` as `mean_S(A_S) / 0.25` and evaluates
`mean_t ||A_D - A_S||^2`. This is an approximation because the original run
did not save an exact independently accumulated full-data vector.
