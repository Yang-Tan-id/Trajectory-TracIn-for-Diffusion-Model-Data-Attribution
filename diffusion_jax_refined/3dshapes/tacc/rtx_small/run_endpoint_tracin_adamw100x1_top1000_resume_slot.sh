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
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"

[[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 1; }

run_worker() {
  local query_id="$1"
  local stage="$2"
  "$python_bin" "$worker" \
    --method "$method" \
    --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "${EXPERIMENT_TAG:-experiment1}" \
    --train-seed "${TRAIN_SEED:-42}" \
    --epochs "${JAX_EPOCHS:-200}" \
    --topk 1000 \
    --stage "$stage"
}

if [[ "$mode" == "validate" ]]; then
  for query_id in $(seq 0 59); do
    run_worker "$query_id" validate >/dev/null
  done
  echo "[preflight] all 60 Endpoint-TracIn top-1000 score artifacts valid"
  exit 0
fi

[[ "$mode" == "run" ]] || { echo "invalid mode: $mode" >&2; exit 1; }

for ((query_id=slot; query_id<60; query_id+=slot_count)); do
  echo "[task] slot=$slot/$slot_count method=$method topk=1000 query=Q$query_id"
  run_worker "$query_id" train
  run_worker "$query_id" eval
done

echo "[slot done] $slot/$slot_count"
