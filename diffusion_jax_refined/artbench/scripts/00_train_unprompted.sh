#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_INDEX="${1:-0}"
END_INDEX="${2:-0}"
ALGORITHMS="${ALGORITHMS:-das traj_tracin dtrak end_tracin journey_trak}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
export GPU_IDS="${GPU_IDS:-${CUDA_DEVICE}}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"

echo "Running training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} for subsets ${START_INDEX}..${END_INDEX}"

for ALGORITHM in ${ALGORITHMS}; do
  echo "Running unprompted training: ${ALGORITHM}"
  for TRAIN_SEED_VALUE in 0 1 2; do
    export TRAIN_SEED="${TRAIN_SEED_VALUE}"
    for SUBSET_INDEX_VALUE in $(seq "${START_INDEX}" "${END_INDEX}"); do
      export SUBSET_INDEX="${SUBSET_INDEX_VALUE}"
      (cd "${ROOT}/data_attribution/${ALGORITHM}" && python run_training.py)
    done
  done
done
