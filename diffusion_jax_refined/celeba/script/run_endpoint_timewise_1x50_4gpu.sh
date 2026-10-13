#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
celeba_root="$(cd "$script_dir/.." && pwd)"
repo="$(cd "$celeba_root/../.." && pwd)"
python_bin="${PYTHON_BIN:-$(command -v python)}"
query_file="${QUERY_FILE:-$celeba_root/queries_attr40_val25_x4.json}"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
experiment="${EXPERIMENT_TAG:-attr40_experiment1}"
gpu_ids_text="${GPU_IDS:-4,5,6,7}"
IFS=, read -r -a gpu_ids <<< "$gpu_ids_text"
shard_count="${#gpu_ids[@]}"
(( shard_count > 0 )) || { echo "GPU_IDS is empty" >&2; exit 1; }
[[ -f "$query_file" ]] || { echo "missing query file: $query_file" >&2; exit 1; }

run_tag="${RUN_TAG:-manual_$(date +%Y%m%d_%H%M%S)}"
result_root="$celeba_root/result/$experiment"
log_root="$result_root/logs/endpoint_pollute_timewise_endpoint_mean_1x50_shape3d_semantics/$run_tag"
component_prefix="$log_root/components"
query_part_root="${QUERY_PART_ROOT:-$result_root/query_bank_endpoint_timewise_1x50_shape3d_semantics_parts}"
mkdir -p "$log_root" "$query_part_root"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[definition] CelebA Attr40 endpoint-pollute timewise + endpoint-TracIn checkpoint-level mean50"
echo "[definition] pollute reductions=linear,termwise_squared,timestamp_sum_squared; endpoint reduction=linear"
echo "[definition] train transforms=raw_gradient,adamw_full; variants=raw,query_l2,train_l2,query_train_l2"
echo "[definition] score paths reproduce Shape3D endpoint-pollute/endpoint-TracIn semantics with timestamp count 50"
echo "[definition] pollute query=raw gradient / prediction-delta L2; four variants use projected norms"
echo "[definition] endpoint query noise=query-specific; endpoint train noise=per-example Shape3D seed rule"
echo "[definition] endpoint score variants normalize checkpoint-level projected mean50 features"
echo "[definition] endpoint train=grad(mean50 loss), then AdamW exactly once; no timestamp-paired contraction"
echo "[definition] separate per-loss full-L2 query bank is persisted for later paper ReTrac; it is not used by endpoint scores"
echo "[gpus] $gpu_ids_text"
echo "[logs] $log_root"
echo "[query parts] $query_part_root"

common=(
  --query-file "$query_file"
  --query-ids "$query_ids"
  --experiment "$experiment"
  --train-seed "${TRAIN_SEED:-42}"
  --epochs "${JAX_EPOCHS:-200}"
  --timestamp-count 50
  --proj-dim "${TRAJ_TRACIN_PROJ_DIM:-4096}"
  --train-batch-size "${TRAJ_SCORE_BATCH_SIZE:-16}"
  --endpoint-train-batch-size "${ENDPOINT_TRAJ_SCORE_BATCH_SIZE:-2}"
  --shard-count "$shard_count"
  --component-prefix "$component_prefix"
  --query-part-root "$query_part_root"
)

pids=()
for shard in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$shard]}"
  CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    "$python_bin" "$script_dir/run_endpoint_timewise_1x50.py" \
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

JAX_PLATFORMS=cpu "$python_bin" "$script_dir/run_endpoint_timewise_1x50.py" \
  "${common[@]}" --shard-index 0 --merge-only \
  >"$log_root/merge.log" 2>&1

echo "[done] scores and reusable query banks complete"
echo "[logs] $log_root"
