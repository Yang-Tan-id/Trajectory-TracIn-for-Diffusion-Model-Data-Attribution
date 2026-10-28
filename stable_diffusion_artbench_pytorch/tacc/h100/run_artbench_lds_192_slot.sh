#!/usr/bin/env bash
set -euo pipefail

slot="${1:?slot is required}"
slot_count="${2:?slot count is required}"
task_file="${3:?task file is required}"

repo="${REPO_ROOT:?REPO_ROOT is required}"
artbench="$repo/stable_diffusion_artbench_pytorch"
output_root="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
mapfile -t tasks < "$task_file"
[[ ${#tasks[@]} -eq 192 ]] || {
  echo "ERROR: expected 192 tasks, found ${#tasks[@]}" >&2
  exit 2
}

for ((task_index=slot; task_index<${#tasks[@]}; task_index+=slot_count)); do
  IFS=$'\t' read -r subset_bank subset_id <<< "${tasks[$task_index]}"
  echo "[task start] index=$task_index/192 slot=$slot subset_seed=$subset_bank subset=$subset_id"
  python -m sd_artbench.artbench_lds_192_plan \
    --mode train \
    --output-root "$output_root" \
    --subset-bank "$subset_bank" \
    --subset-id "$subset_id"
  echo "[task done] index=$task_index/192 slot=$slot subset_seed=$subset_bank subset=$subset_id"
done

echo "[slot complete] slot=$slot/$slot_count"
