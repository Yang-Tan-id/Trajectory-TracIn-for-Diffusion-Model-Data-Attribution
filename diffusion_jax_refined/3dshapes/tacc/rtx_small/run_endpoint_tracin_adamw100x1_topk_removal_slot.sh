#!/usr/bin/env bash
set -euo pipefail

mode="${1:?mode is required}"
slot="${2:-0}"
slot_count="${3:-1}"
repo="${REPO_ROOT:?REPO_ROOT is required}"
shapes="$repo/diffusion_jax_refined/3dshapes"
worker="$shapes/script/run_retrac_endpoint_topk_removal.py"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
method="endpoint_tracin_adamw_full_100x1_raw_neg"
topks=(400 1000)
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"
[[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 1; }

if [[ "$mode" == "validate" ]]; then
  for query_id in $(seq 0 59); do
    "$python_bin" "$worker" \
      --method "$method" --query-id "$query_id" \
      --query-file "$query_file" \
      --experiment "${EXPERIMENT_TAG:-experiment1}" \
      --train-seed "${TRAIN_SEED:-42}" --epochs "${JAX_EPOCHS:-200}" \
      --topk 1000 --stage validate >/dev/null
  done
  echo "[preflight] all 60 Endpoint-TracIn raw x -1 score artifacts valid"
  exit 0
fi

[[ "$mode" == "run" ]] || { echo "invalid mode: $mode" >&2; exit 1; }
task_count=120
for ((task=slot; task<task_count; task+=slot_count)); do
  query_id="$((task % 60))"
  topk="${topks[$((task / 60))]}"
  echo "[task $task/$task_count] slot=$slot method=$method topk=$topk query=Q$query_id"

  "$python_bin" "$worker" \
    --method "$method" --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "${EXPERIMENT_TAG:-experiment1}" \
    --train-seed "${TRAIN_SEED:-42}" --epochs "${JAX_EPOCHS:-200}" \
    --topk "$topk" --stage train

  "$python_bin" "$worker" \
    --method "$method" --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "${EXPERIMENT_TAG:-experiment1}" \
    --train-seed "${TRAIN_SEED:-42}" --epochs "${JAX_EPOCHS:-200}" \
    --topk "$topk" --stage eval
done
echo "[slot done] $slot/$slot_count"
