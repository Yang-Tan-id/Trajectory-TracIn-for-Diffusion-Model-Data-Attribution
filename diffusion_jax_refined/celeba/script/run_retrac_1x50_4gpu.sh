#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
celeba_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$(command -v python)}"
query_file="${QUERY_FILE:-$celeba_root/queries_attr40_val25_x4.json}"
query_ids="${QUERY_IDS:-$(seq -s, 0 71)}"
experiment="${EXPERIMENT_TAG:-attr40_experiment1}"
gpu_ids_text="${GPU_IDS:-0,5,6,7}"
IFS=, read -r -a gpu_ids <<< "$gpu_ids_text"
shard_count="${#gpu_ids[@]}"
(( shard_count > 0 )) || { echo "GPU_IDS is empty" >&2; exit 1; }

run_tag="${RUN_TAG:-celeba_retrac_1x50_$(date +%Y%m%d_%H%M%S)}"
result_root="$celeba_root/result/$experiment"
log_root="$result_root/logs/retrac_1x50_full_l2/$run_tag"
component_prefix="$log_root/components"
mkdir -p "$log_root"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DISABLE_JIT=0

echo "[definition] CelebA paper ReTrac 1x50; 49 checkpoint pairs"
echo "[definition] query: normalize each of 50 full-space loss gradients, project, then mean"
echo "[definition] train: normalize each full-space loss gradient before projection/contraction"
echo "[definition] transforms=raw_gradient,adamw_full; reduction=linear; no persistent train gradients"
echo "[queries] $query_ids"
echo "[gpus] $gpu_ids_text"
echo "[logs] $log_root"

common=(
  --query-file "$query_file"
  --query-ids "$query_ids"
  --experiment "$experiment"
  --train-seed "${TRAIN_SEED:-42}"
  --epochs "${JAX_EPOCHS:-200}"
  --proj-dim "${TRAJ_TRACIN_PROJ_DIM:-4096}"
  --train-batch-size "${TRAJ_SCORE_BATCH_SIZE:-16}"
  --progress-every-batches "${PROGRESS_EVERY_BATCHES:-25}"
  --shard-count "$shard_count"
  --component-prefix "$component_prefix"
)

pids=()
for shard in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$shard]}"
  CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    "$python_bin" "$script_dir/run_retrac_1x50.py" \
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

JAX_PLATFORMS=cpu "$python_bin" "$script_dir/run_retrac_1x50.py" \
  "${common[@]}" --shard-index 0 --merge-only \
  >"$log_root/merge.log" 2>&1

echo "[done] ReTrac raw-gradient and AdamW-full scores complete"
echo "[logs] $log_root"
