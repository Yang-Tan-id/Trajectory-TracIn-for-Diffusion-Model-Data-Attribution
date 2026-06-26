#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALGORITHMS="${ALGORITHMS:-das traj_tracin dtrak end_tracin journey_trak}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
ATTRIBUTION_RANGES="${ATTRIBUTION_RANGES:-${SCORE_INDEX_RANGES:-}}"

echo "Running unprompted attribution on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for ALGORITHM in ${ALGORITHMS}; do
  if [[ -n "${ATTRIBUTION_RANGES}" ]]; then
    RANGE_LIST="${ATTRIBUTION_RANGES//,/ }"
    for RANGE_VALUE in ${RANGE_LIST}; do
      echo "Running unprompted attribution: ${ALGORITHM} range=${RANGE_VALUE}"
      SCORE_INDEX_RANGES="${RANGE_VALUE}" python "${ROOT}/../common/unprompted_diffusers_attribution.py" "${ROOT}/dataset_config.py" --algorithm "${ALGORITHM}"
    done
  else
    echo "Running unprompted attribution: ${ALGORITHM}"
    python "${ROOT}/../common/unprompted_diffusers_attribution.py" "${ROOT}/dataset_config.py" --algorithm "${ALGORITHM}"
  fi
done
