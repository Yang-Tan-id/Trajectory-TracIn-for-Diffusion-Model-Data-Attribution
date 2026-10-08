#!/usr/bin/env bash
set -euo pipefail

slot="${1:?slot index is required}"
slot_count="${2:?slot count is required}"
repo="${REPO_ROOT:?REPO_ROOT is required}"
shapes="$repo/diffusion_jax_refined/3dshapes"
worker="$shapes/script/run_retrac_endpoint_topk_removal.py"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"

methods=(
  retrac_adamw_both_l2_neg
  endpoint_pollute_adamw_timestamp_train_l2
)
topks=(400 1000)
task_count=240

for ((task=slot; task<task_count; task+=slot_count)); do
  query_id="$((task % 60))"
  topk_id="$(((task / 60) % 2))"
  method_id="$((task / 120))"
  method="${methods[$method_id]}"
  topk="${topks[$topk_id]}"
  echo "[task $task/$task_count] slot=$slot method=$method topk=$topk query=Q$query_id"

  python "$worker" \
    --method "$method" \
    --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "${EXPERIMENT_TAG:-experiment1}" \
    --train-seed "${TRAIN_SEED:-42}" \
    --epochs "${JAX_EPOCHS:-200}" \
    --topk "$topk" \
    --stage train

  # A fresh process releases the training executable before trajectory evaluation.
  python "$worker" \
    --method "$method" \
    --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "${EXPERIMENT_TAG:-experiment1}" \
    --train-seed "${TRAIN_SEED:-42}" \
    --epochs "${JAX_EPOCHS:-200}" \
    --topk "$topk" \
    --stage eval
done

echo "[slot done] $slot/$slot_count"
