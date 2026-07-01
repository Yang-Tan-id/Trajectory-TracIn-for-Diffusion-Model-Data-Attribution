#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_INDEX="${1:-0}"
END_INDEX="${2:-0}"
TRAIN_SEEDS="${TRAIN_SEEDS:-${TRAIN_SEED:-42}}"
UNPROMPTED_USE_SUBSET="${UNPROMPTED_USE_SUBSET:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
export GPU_IDS="${GPU_IDS:-${CUDA_DEVICE}}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"

if [[ "${UNPROMPTED_USE_SUBSET}" == "1" ]]; then
  echo "Running unprompted subset training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} for subsets ${START_INDEX}..${END_INDEX}, seeds ${TRAIN_SEEDS}"
else
  echo "Running unprompted full-dataset training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, seeds ${TRAIN_SEEDS}"
fi

for TRAIN_SEED_VALUE in ${TRAIN_SEEDS}; do
  export TRAIN_SEED="${TRAIN_SEED_VALUE}"
  if [[ "${UNPROMPTED_USE_SUBSET}" == "1" ]]; then
    for SUBSET_INDEX_VALUE in $(seq "${START_INDEX}" "${END_INDEX}"); do
      export SUBSET_INDEX="${SUBSET_INDEX_VALUE}"
      python "${ROOT}/../common/unprompted_training_runner.py" "${ROOT}/dataset_config.py"
    done
  else
    python "${ROOT}/../common/unprompted_training_runner.py" "${ROOT}/dataset_config.py"
  fi
done
