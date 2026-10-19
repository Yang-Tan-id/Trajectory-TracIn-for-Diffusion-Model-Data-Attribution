#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
celeba_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
gpu_ids_text="${GPU_IDS:-0,5,6,7}"
query_ids_text="${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}"
experiment="${EXPERIMENT:-attr40_experiment1}"
query_file="${QUERY_FILE:-$celeba_root/queries_attr40_val25_x4.json}"
run_tag="${RUN_TAG:-celeba_endpoint_full10k_query_l2_top1000_$(date +%Y%m%d_%H%M%S)}"
worker="$script_dir/run_endpoint_top500_removal.py"
log_root="$celeba_root/result/$experiment/logs/endpoint_full10k_query_l2_top1000/$run_tag"
topk=1000
methods=(
  endpoint_pollute_adamw_timestamp_query_l2_full10k_pos
  endpoint_tracin_adamw_linear_query_l2_full10k_neg
)

IFS=',' read -r -a gpu_ids <<< "$gpu_ids_text"
IFS=',' read -r -a query_ids <<< "$query_ids_text"
[[ ${#gpu_ids[@]} -eq 4 ]] || { echo "ERROR: exactly four GPUs required" >&2; exit 2; }
[[ ${#query_ids[@]} -eq 10 ]] || { echo "ERROR: exactly ten queries required" >&2; exit 2; }
mkdir -p "$log_root"

echo "[definition] full-10000 candidate top-1000 removal"
echo "[definition] Pollute Query-L2 descending; TracIn Query-L2 ascending"
echo "[gpus] $gpu_ids_text"
echo "[logs] $log_root"

tasks=()
for method in "${methods[@]}"; do
  for query_id in "${query_ids[@]}"; do
    tasks+=("$method:$query_id")
  done
done

for task in "${tasks[@]}"; do
  method="${task%%:*}"
  query_id="${task##*:}"
  JAX_PLATFORMS=cpu "$python_bin" "$worker" \
    --method "$method" --query-id "$query_id" --query-file "$query_file" \
    --experiment "$experiment" --topk "$topk" --stage validate
done | tee "$log_root/preflight.log"

pids=()
for slot in 0 1 2 3; do
  gpu_id="${gpu_ids[$slot]}"
  (
    for ((task_index=slot; task_index<${#tasks[@]}; task_index+=4)); do
      task="${tasks[$task_index]}"
      method="${task%%:*}"
      query_id="${task##*:}"
      echo "[task start] slot=$slot gpu=$gpu_id method=$method Q$query_id"
      CUDA_VISIBLE_DEVICES="$gpu_id" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
      "$python_bin" "$worker" \
        --method "$method" --query-id "$query_id" --query-file "$query_file" \
        --experiment "$experiment" --topk "$topk" --stage all
      echo "[task done] slot=$slot gpu=$gpu_id method=$method Q$query_id"
    done
  ) >"$log_root/gpu_${gpu_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "[failed] inspect $log_root/gpu_*.log" >&2; exit 1; }
echo "[done] all 20 full-10000 top-1000 removal runs complete"
