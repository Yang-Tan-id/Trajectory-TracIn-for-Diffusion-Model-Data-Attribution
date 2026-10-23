#!/usr/bin/env bash
set -euo pipefail

slot="${1:?slot is required}"
slot_count="${2:?slot count is required}"
task_file="${3:?task file is required}"

repo="${REPO_ROOT:?REPO_ROOT is required}"
celeba="$repo/diffusion_jax_refined/celeba"
worker="$celeba/script/run_endpoint_top500_removal.py"
query_file="$celeba/queries_attr40_val25_x4.json"

mapfile -t tasks < "$task_file"
[[ ${#tasks[@]} -eq 250 ]] || {
  echo "ERROR: expected 250 tasks, found ${#tasks[@]}" >&2
  exit 2
}

for ((task_index=slot; task_index<${#tasks[@]}; task_index+=slot_count)); do
  IFS=$'\t' read -r method query_id <<< "${tasks[$task_index]}"
  echo "[task start] index=$task_index/250 slot=$slot method=$method Q$query_id"
  python "$worker" \
    --method "$method" \
    --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "${EXPERIMENT_TAG:-attr40_experiment1}" \
    --train-seed "${TRAIN_SEED:-42}" \
    --epochs "${JAX_EPOCHS:-200}" \
    --topk 1000 \
    --output-group endpoint_top1000_removal_q0_69 \
    --stage all
  echo "[task done] index=$task_index/250 slot=$slot method=$method Q$query_id"
done
