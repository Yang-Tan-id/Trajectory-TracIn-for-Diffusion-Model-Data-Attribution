#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
celeba_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$(command -v python)}"
query_file="${QUERY_FILE:-$celeba_root/queries_attr40_val25_x4.json}"
query_ids="${QUERY_IDS:-$(seq -s, 0 71)}"
experiment="${EXPERIMENT_TAG:-attr40_experiment1}"
gpu_ids_text="${GPU_IDS:-0,5,6,7}"
timesteps="${TIMESTEPS:-183,163,142,122,101,81,61,40,20,0}"
timestep_tag="${TIMESTEP_TAG:-endpoint_near10}"
checkpoint_indices="${CHECKPOINT_INDICES:-0,5,10,16,21,26,32,37,42,48}"
IFS=, read -r -a gpu_ids <<< "$gpu_ids_text"
shard_count="${#gpu_ids[@]}"
(( shard_count > 0 )) || { echo "GPU_IDS is empty" >&2; exit 1; }
[[ -f "$query_file" ]] || { echo "missing query file: $query_file" >&2; exit 1; }

run_tag="${RUN_TAG:-celeba_endpoint_tracin_query_modes_near10_ckpt10_$(date +%Y%m%d_%H%M%S)}"
result_root="$celeba_root/result/$experiment"
log_root="$result_root/logs/endpoint_tracin_query_modes_near10_checkpoint10/$run_tag"
component_prefix="$log_root/components"
mkdir -p "$log_root"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DISABLE_JIT=0

echo "[definition] Endpoint-TracIn; train=gradient(mean selected-timestep loss)"
echo "[definition] query modes: raw per-timestep gradient; full-space L2-normalized per-timestep gradient"
echo "[definition] reductions: linear, termwise_squared, timestamp_sum_squared, total_sum_squared"
echo "[definition] transforms: raw_gradient, adamw_full; train variants: raw, projected-L2"
echo "[definition] no persistent train gradients"
echo "[timesteps] $timesteps | tag=$timestep_tag"
echo "[checkpoints] source transition indices=$checkpoint_indices"
echo "[queries] $query_ids"
echo "[gpus] $gpu_ids_text"
echo "[logs] $log_root"

common=(
  --query-file "$query_file"
  --query-ids "$query_ids"
  --experiment "$experiment"
  --train-seed "${TRAIN_SEED:-42}"
  --epochs "${JAX_EPOCHS:-200}"
  --timesteps "$timesteps"
  --timestep-tag "$timestep_tag"
  --checkpoint-indices "$checkpoint_indices"
  --proj-dim "${TRAJ_TRACIN_PROJ_DIM:-4096}"
  --train-batch-size "${ENDPOINT_TRAJ_SCORE_BATCH_SIZE:-2}"
  --progress-every-batches "${PROGRESS_EVERY_BATCHES:-250}"
  --shard-count "$shard_count"
  --component-prefix "$component_prefix"
  --per-timestep-query-reductions
)

pids=()
for shard in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$shard]}"
  CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    "$python_bin" "$script_dir/run_endpoint_tracin_subset.py" \
      "${common[@]}" --shard-index "$shard" \
      >"$log_root/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for worker_pid in "${pids[@]}"; do
  wait "$worker_pid" || failed=1
done
(( failed == 0 )) || {
  echo "GPU worker failed; inspect $log_root" >&2
  exit 1
}

JAX_PLATFORMS=cpu "$python_bin" "$script_dir/run_endpoint_tracin_subset.py" \
  "${common[@]}" --shard-index 0 --merge-only \
  >"$log_root/merge.log" 2>&1

echo "[done] Endpoint-TracIn query-mode/reduction scores complete"
echo "[logs] $log_root"
echo "[namespaces] $log_root/score_namespaces.txt"
