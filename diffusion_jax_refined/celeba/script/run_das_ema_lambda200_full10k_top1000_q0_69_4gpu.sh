#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
celeba_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
gpu_ids_text="${GPU_IDS:-0,1,2,3}"
query_ids_text="${QUERY_IDS:-$(seq -s, 0 69)}"
experiment="${EXPERIMENT:-attr40_experiment1}"
query_file="${QUERY_FILE:-$celeba_root/queries_attr40_val25_x4.json}"
run_tag="${RUN_TAG:-celeba_das_ema_l200_full10k_top1000_$(date +%Y%m%d_%H%M%S)}"
worker="$script_dir/run_das_ema_top1000_removal.py"
log_root="$celeba_root/result/$experiment/logs/das_ema_l200_full10k_top1000/$run_tag"
method="das_ema_lambda200_full10k_pos"
output_group="das_top1000_removal_q0_69"
topk=1000

IFS=',' read -r -a gpu_ids <<< "$gpu_ids_text"
IFS=',' read -r -a query_ids <<< "$query_ids_text"
[[ ${#gpu_ids[@]} -eq 4 ]] || { echo "ERROR: exactly four GPUs required" >&2; exit 2; }
[[ ${#query_ids[@]} -gt 0 ]] || { echo "ERROR: QUERY_IDS is empty" >&2; exit 2; }
mkdir -p "$log_root"

echo "[definition] CelebA EMA-DAS 50x1 lambda=200, full-10000 Top-1000 removal"
echo "[selection] one global lambda selected by mean Q0-Q99 simple_loss_50x1 LDS"
echo "[queries] $query_ids_text"
echo "[gpus] $gpu_ids_text"
echo "[logs] $log_root"

for query_id in "${query_ids[@]}"; do
  JAX_PLATFORMS=cpu "$python_bin" "$worker" \
    --method "$method" --query-id "$query_id" --query-file "$query_file" \
    --experiment "$experiment" --topk "$topk" --output-group "$output_group" \
    --stage validate
done | tee "$log_root/preflight.log"

pids=()
for slot in 0 1 2 3; do
  gpu_id="${gpu_ids[$slot]}"
  (
    for ((index=slot; index<${#query_ids[@]}; index+=4)); do
      query_id="${query_ids[$index]}"
      echo "[task start] slot=$slot gpu=$gpu_id Q$query_id"
      CUDA_VISIBLE_DEVICES="$gpu_id" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
      "$python_bin" "$worker" \
        --method "$method" --query-id "$query_id" --query-file "$query_file" \
        --experiment "$experiment" --topk "$topk" --output-group "$output_group" \
        --stage all
      echo "[task done] slot=$slot gpu=$gpu_id Q$query_id"
    done
  ) >"$log_root/gpu_${gpu_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "[failed] inspect $log_root/gpu_*.log" >&2; exit 1; }
echo "[done] ${#query_ids[@]} DAS Top-1000 removal runs complete"
