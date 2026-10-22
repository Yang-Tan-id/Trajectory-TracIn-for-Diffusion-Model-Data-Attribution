#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
celeba_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
gpu_ids_text="${GPU_IDS:-0,5,6,7}"
query_ids_text="${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}"
experiment="${EXPERIMENT:-attr40_experiment1}"
query_file="${QUERY_FILE:-$celeba_root/queries_attr40_val25_x4.json}"
run_tag="${RUN_TAG:-celeba_retrac_full10k_top1000_q0_9_$(date +%Y%m%d_%H%M%S)}"
worker="$script_dir/run_endpoint_top500_removal.py"
log_root="$celeba_root/result/$experiment/logs/retrac_full10k_top1000_q0_9/$run_tag"
method="retrac_adamw_query_train_l2_full10k_neg"
topk=1000

IFS=',' read -r -a gpu_ids <<< "$gpu_ids_text"
IFS=',' read -r -a query_ids <<< "$query_ids_text"
(( ${#gpu_ids[@]} > 0 )) || { echo "ERROR: GPU_IDS is empty" >&2; exit 2; }
[[ ${#query_ids[@]} -eq 10 ]] || {
  echo "ERROR: expected ten queries; got QUERY_IDS=$query_ids_text" >&2
  exit 2
}
[[ -x "$python_bin" ]] || { echo "ERROR: Python not executable: $python_bin" >&2; exit 2; }
[[ -f "$query_file" ]] || { echo "ERROR: query file missing: $query_file" >&2; exit 2; }
mkdir -p "$log_root"

echo "[definition] paper-ReTrac AdamW-full replay4-batch1 query+train-L2"
echo "[definition] full-10000 candidate bank; stored score ascending"
echo "[definition] remove top-1000 = 10% of the complete training set"
echo "[queries] $query_ids_text"
echo "[gpus] $gpu_ids_text"
echo "[logs] $log_root"

for query_id in "${query_ids[@]}"; do
  JAX_PLATFORMS=cpu "$python_bin" "$worker" \
    --method "$method" \
    --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "$experiment" \
    --topk "$topk" \
    --stage validate
done | tee "$log_root/preflight.log"

pids=()
for slot in "${!gpu_ids[@]}"; do
  gpu_id="${gpu_ids[$slot]}"
  (
    for ((query_slot=slot; query_slot<${#query_ids[@]}; query_slot+=${#gpu_ids[@]})); do
      query_id="${query_ids[$query_slot]}"
      echo "[task start] slot=$slot gpu=$gpu_id method=$method Q$query_id"
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      JAX_PLATFORMS=cuda \
      JAX_NUM_DEVICES=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      TF_GPU_ALLOCATOR=cuda_malloc_async \
      "$python_bin" "$worker" \
        --method "$method" \
        --query-id "$query_id" \
        --query-file "$query_file" \
        --experiment "$experiment" \
        --topk "$topk" \
        --stage all
      echo "[task done] slot=$slot gpu=$gpu_id method=$method Q$query_id"
    done
  ) >"$log_root/gpu_${gpu_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || {
  echo "[failed] at least one worker failed; inspect $log_root/gpu_*.log" >&2
  exit 1
}

echo "[done] all 10 ReTrac full-10000 top-1000 removal runs complete"
echo "[results] $celeba_root/result/$experiment/endpoint_top1000_removal_q0_9/$method"
echo "[logs] $log_root"
