#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
celeba_root="$(cd "$script_dir/.." && pwd)"

python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
gpu_ids_text="${GPU_IDS:-0,5,6,7}"
query_ids_text="${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}"
experiment="${EXPERIMENT:-attr40_experiment1}"
run_tag="${RUN_TAG:-celeba_endpoint_top1000_query_l2_q0_9_$(date +%Y%m%d_%H%M%S)}"
query_file="${QUERY_FILE:-$celeba_root/queries_attr40_val25_x4.json}"
worker="$script_dir/run_endpoint_top500_removal.py"
topk=1000
log_root="$celeba_root/result/$experiment/logs/endpoint_top1000_query_l2_q0_9/$run_tag"

methods=(
  endpoint_pollute_adamw_timestamp_query_l2_pos
  endpoint_tracin_adamw_linear_query_l2_neg
)

IFS=',' read -r -a gpu_ids <<< "$gpu_ids_text"
IFS=',' read -r -a query_ids <<< "$query_ids_text"
if [[ ${#gpu_ids[@]} -ne 4 ]]; then
  echo "ERROR: exactly four GPU IDs are required; got GPU_IDS=$gpu_ids_text" >&2
  exit 2
fi
if [[ ${#query_ids[@]} -ne 10 ]]; then
  echo "ERROR: exactly ten query IDs are required; got QUERY_IDS=$query_ids_text" >&2
  exit 2
fi
[[ -x "$python_bin" ]] || { echo "ERROR: Python not executable: $python_bin" >&2; exit 2; }
[[ -f "$query_file" ]] || { echo "ERROR: query file missing: $query_file" >&2; exit 2; }

mkdir -p "$log_root"
echo "[definition] CelebA Q0-Q9 top-1000 removal = 10% of full training set"
echo "[definition] methods=Endpoint-Pollute query-L2, Endpoint-TracIn query-L2"
echo "[definition] Pollute stored scores descending; Endpoint-TracIn stored scores ascending"
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
    --method "$method" \
    --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "$experiment" \
    --topk "$topk" \
    --stage validate
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
  ) > "$log_root/gpu_${gpu_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "[failed] at least one worker failed; inspect $log_root/gpu_*.log" >&2
  exit "$status"
fi

echo "[done] all 20 query-L2 top-1000 removal retrains/evaluations complete"
echo "[results] $celeba_root/result/$experiment/endpoint_top1000_removal_q0_9"
echo "[logs] $log_root"
