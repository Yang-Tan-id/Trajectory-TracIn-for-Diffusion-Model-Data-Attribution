#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$HOME/diffusion_model/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution}"
celeba="$repo/diffusion_jax_refined/celeba"
py="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
script="$celeba/script/evaluate_top1000_q0_69_generate.py"
query_file="$celeba/queries_attr40_val25_x4.json"
gpu_text="${GPU_IDS:-0,5,6,7}"
IFS=',' read -r -a gpus <<< "$gpu_text"
[[ ${#gpus[@]} -gt 0 ]] || { echo "ERROR: GPU_IDS is empty" >&2; exit 2; }

output_dir="$celeba/result/attr40_experiment1/analysis/endpoint_top1000_q0_69_five_method_metrics"
log_dir="$output_dir/logs"
mkdir -p "$log_dir"

"$py" -m py_compile "$script"

pids=()
for slot in "${!gpus[@]}"; do
  ids=()
  for ((query_id=slot; query_id<70; query_id+=${#gpus[@]})); do
    ids+=("$query_id")
  done
  query_ids="$(IFS=,; echo "${ids[*]}")"
  gpu="${gpus[$slot]}"
  log="$log_dir/slot_${slot}_gpu_${gpu}.log"
  echo "[launch] slot=$slot gpu=$gpu queries=$query_ids log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$py" "$script" \
    --query-file "$query_file" \
    --query-ids "$query_ids" \
    --trajectory-steps 10 \
    --output-dir "$output_dir" \
    >"$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "ERROR: one or more generation slots failed; inspect $log_dir" >&2
  exit "$status"
fi

count="$(find "$output_dir" -type f -name METRICS_GENERATION_COMPLETE | wc -l | tr -d ' ')"
[[ "$count" == "70" ]] || { echo "ERROR: expected 70 complete queries, found $count" >&2; exit 3; }
echo "[done] all 70 queries generated: $output_dir"
