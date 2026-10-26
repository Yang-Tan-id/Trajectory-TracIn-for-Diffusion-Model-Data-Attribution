#!/usr/bin/env bash
set -euo pipefail

slot="${1:?slot is required}"
slot_count="${2:?slot count is required}"
task_file="${3:?task file is required}"

repo="${REPO_ROOT:?REPO_ROOT is required}"
output_root="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
mapfile -t tasks < "$task_file"
(( ${#tasks[@]} > 0 )) || { echo "ERROR: empty task file" >&2; exit 2; }

for ((task_index=slot; task_index<${#tasks[@]}; task_index+=slot_count)); do
  IFS=$'\t' read -r method query_index <<< "${tasks[$task_index]}"
  [[ "$method" == "random_full5000_seeded" ]] || {
    echo "ERROR: non-random task in random worker: $method" >&2; exit 2;
  }
  echo "[task start] index=$task_index/${#tasks[@]} slot=$slot method=$method Q$query_index"
  python -m sd_artbench.topk_removal \
    --method "$method" \
    --query-index "$query_index" \
    --top-k 1000 \
    --random-seed 2026 \
    --output-root "$output_root" \
    --stage all
  echo "[task done] index=$task_index/${#tasks[@]} slot=$slot method=$method Q$query_index"
done

echo "[slot complete] slot=$slot/$slot_count"
