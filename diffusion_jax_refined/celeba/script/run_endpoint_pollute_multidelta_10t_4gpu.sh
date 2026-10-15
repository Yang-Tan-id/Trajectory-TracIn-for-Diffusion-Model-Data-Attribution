#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
celeba_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$(command -v python)}"
query_file="${QUERY_FILE:-$celeba_root/queries_attr40_val25_x4.json}"
query_ids="${QUERY_IDS:-$(seq -s, 0 71)}"
experiment="${EXPERIMENT_TAG:-attr40_experiment1}"
gpu_ids_text="${GPU_IDS:-4,5,6,7}"
direction_count="${QUERY_DIRECTION_COUNT:-4}"
timesteps="${TIMESTEPS:-999,888,777,666,555,444,333,222,111,0}"
timestep_tag="${TIMESTEP_TAG:-}"
loss_side_aggregate_qvp="${LOSS_SIDE_AGGREGATE_QVP:-0}"
checkpoint_indices="${CHECKPOINT_INDICES:-0,5,10,16,21,26,32,37,42,48}"
if [[ "$loss_side_aggregate_qvp" == "1" ]]; then
  direction_count=1
fi
IFS=, read -r -a gpu_ids <<< "$gpu_ids_text"
shard_count="${#gpu_ids[@]}"
(( shard_count > 0 )) || { echo "GPU_IDS is empty" >&2; exit 1; }
[[ -f "$query_file" ]] || { echo "missing query file: $query_file" >&2; exit 1; }

run_tag="${RUN_TAG:-manual_$(date +%Y%m%d_%H%M%S)}"
result_root="$celeba_root/result/$experiment"
log_group="endpoint_pollute_multidelta${direction_count}_10t"
if [[ "$loss_side_aggregate_qvp" == "1" ]]; then
  log_group="endpoint_pollute_loss_aggregated_qvp_10x10"
fi
[[ -z "$timestep_tag" ]] || log_group="${log_group}_${timestep_tag}"
log_root="$result_root/logs/$log_group/$run_tag"
component_prefix="$log_root/components"
mkdir -p "$log_root"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DISABLE_JIT=0

echo "[definition] endpoint-pollute | 10 timestamps | ${direction_count} query feature(s)"
echo "[timesteps] $timesteps${timestep_tag:+ | tag=$timestep_tag}"
echo "[checkpoints] source pair indices=$checkpoint_indices"
echo "[definition] train gradient once; raw-gradient + AdamW-full; four normalization variants"
echo "[definition] reductions=linear,termwise_squared,per-timestamp-linear"
echo "[definition] per-direction scores + direction-mean score; no persistent train gradients"
echo "[queries] $query_ids"
echo "[gpus] $gpu_ids_text"
echo "[logs] $log_root"

common=(
  --query-file "$query_file"
  --query-ids "$query_ids"
  --experiment "$experiment"
  --train-seed "${TRAIN_SEED:-42}"
  --epochs "${JAX_EPOCHS:-200}"
  --timestamp-count 10
  --timesteps "$timesteps"
  --timestep-tag "$timestep_tag"
  --direction-count "$direction_count"
  --checkpoint-indices "$checkpoint_indices"
  --proj-dim "${TRAJ_TRACIN_PROJ_DIM:-4096}"
  --train-batch-size "${TRAJ_SCORE_BATCH_SIZE:-16}"
  --progress-every-batches "${PROGRESS_EVERY_BATCHES:-25}"
  --shard-count "$shard_count"
  --component-prefix "$component_prefix"
)
if [[ "$loss_side_aggregate_qvp" == "1" ]]; then
  common+=(--loss-side-aggregate-qvp)
fi

pids=()
for shard in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$shard]}"
  CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    "$python_bin" "$script_dir/run_endpoint_pollute_multidelta_10t.py" \
      "${common[@]}" --shard-index "$shard" \
      >"$log_root/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || {
  echo "GPU worker failed; inspect $log_root" >&2
  exit 1
}

JAX_PLATFORMS=cpu "$python_bin" \
  "$script_dir/run_endpoint_pollute_multidelta_10t.py" \
  "${common[@]}" --shard-index 0 --merge-only \
  >"$log_root/merge.log" 2>&1

echo "[done] multidelta endpoint-pollute scores complete"
echo "[logs] $log_root"
echo "[namespaces] $log_root/score_namespaces.txt"
