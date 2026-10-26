#!/usr/bin/env bash
set -euo pipefail

slot="${1:?slot is required}"
slot_count="${2:?slot count is required}"
task_file="${3:?task file is required}"

repo="${REPO_ROOT:?REPO_ROOT is required}"
artbench="$repo/stable_diffusion_artbench_pytorch"
output_root="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
mapfile -t tasks < "$task_file"
[[ ${#tasks[@]} -eq 260 ]] || {
  echo "ERROR: expected 260 tasks, found ${#tasks[@]}" >&2
  exit 2
}

for ((task_index=slot; task_index<${#tasks[@]}; task_index+=slot_count)); do
  IFS=$'\t' read -r method query_index <<< "${tasks[$task_index]}"
  echo "[task start] index=$task_index/260 slot=$slot method=$method Q$query_index"
  python -m sd_artbench.topk_removal \
    --method "$method" \
    --query-index "$query_index" \
    --top-k 1000 \
    --random-seed 2026 \
    --output-root "$output_root" \
    --stage all
  echo "[task done] index=$task_index/260 slot=$slot method=$method Q$query_index"
done

echo "[slot complete] slot=$slot/$slot_count"
