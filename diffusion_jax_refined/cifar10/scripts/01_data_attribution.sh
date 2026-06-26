#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALGORITHMS="${ALGORITHMS:-das traj_tracin dtrak end_tracin journey_trak}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
ATTRIBUTION_RANGES="${ATTRIBUTION_RANGES:-${SCORE_INDEX_RANGES:-}}"

echo "Running attribution on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for ALGORITHM in ${ALGORITHMS}; do
  if [[ "${ALGORITHM}" == "traj_tracin" && -n "${ATTRIBUTION_RANGES}" ]]; then
    RANGE_LIST="${ATTRIBUTION_RANGES//,/ }"
    for RANGE_VALUE in ${RANGE_LIST}; do
      echo "Running attribution: ${ALGORITHM} range=${RANGE_VALUE}"
      (cd "${ROOT}/data_attribution/${ALGORITHM}" && SCORE_INDEX_RANGES="${RANGE_VALUE}" python run_attribution.py)
    done
  else
    echo "Running attribution: ${ALGORITHM}"
    (cd "${ROOT}/data_attribution/${ALGORITHM}" && python run_attribution.py)
  fi
done
