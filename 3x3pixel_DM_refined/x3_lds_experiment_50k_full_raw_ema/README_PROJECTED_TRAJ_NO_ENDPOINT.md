# Projected Traj TracIn without the endpoint

This q00-q09 variant reproduces the first-order, raw-parameter,
next-checkpoint projected Traj TracIn bank while excluding trajectory index 99,
the final generated `x0` stored in `final_state.npy`. It retains trajectory
indices 0..98 and renormalizes the timestamp weight from `1/100` to `1/99`.

Linear, termwise-square, and timestamp-sum-square scores are saved under new
method namespaces, so existing 100-timestamp attribution is not overwritten.
The launcher uses four timestamp shards, merges them, and evaluates every LDS
target with both score signs.

```bash
python -u 134_launch_projected_traj_no_endpoint_10q_4gpu.py \
  --gpus 0,1,2,3 \
  --batch-size 128
```

LDS output:

```text
x3_lds_exp_50k/lds/projected_traj_no_endpoint_q00_q09.json
```
