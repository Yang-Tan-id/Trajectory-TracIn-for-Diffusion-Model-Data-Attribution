#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SAMPLE_MODEL_MODE="${SAMPLE_MODEL_MODE:-unprompted_solo}"
LDS_MODEL_TRAIN_SEED="${LDS_MODEL_TRAIN_SEED:-${LDS_TRAIN_SEED:-${TRAIN_SEED:-42}}}"

ARGS=(
  --unprompted
  --sample-model-mode "${SAMPLE_MODEL_MODE}"
  --model-train-seed "${LDS_MODEL_TRAIN_SEED}"
  --m "${LDS_M:-${LDS_NUM_SUBSETS:-100}}"
  --sample-random-seed "${LDS_SAMPLE_RANDOM_SEED:-${LDS_SUBSET_SEED:-0}}"
)

if [[ -n "${LDS_DATASET_PERCENTAGE:-${LDS_DATASET_PERCENT:-}}" ]]; then
  ARGS+=(--dataset-percentage "${LDS_DATASET_PERCENTAGE:-${LDS_DATASET_PERCENT:-}}")
  unset LDS_K LDS_SUBSET_SIZE
elif [[ -n "${LDS_K:-${LDS_SUBSET_SIZE:-}}" ]]; then
  ARGS+=(--k "${LDS_K:-${LDS_SUBSET_SIZE:-}}")
fi

"${PYTHON_BIN}" "${ROOT}/lds/run_training.py" "${ARGS[@]}" "$@"
