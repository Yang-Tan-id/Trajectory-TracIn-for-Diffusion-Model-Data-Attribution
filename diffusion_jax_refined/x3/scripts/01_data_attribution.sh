#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALGORITHMS="${ALGORITHMS:-das traj_tracin dtrak end_tracin journey_trak}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
ATTRIBUTION_RANGES="${ATTRIBUTION_RANGES:-${SCORE_INDEX_RANGES:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Running staged attribution on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "03_score.py is a pure artifact combiner; it will not rerun the monolithic attribution engine."

for ALGORITHM in ${ALGORITHMS}; do
  if [[ "${ALGORITHM}" == "traj_tracin" && -n "${ATTRIBUTION_RANGES}" ]]; then
    RANGE_LIST="${ATTRIBUTION_RANGES//,/ }"
    for RANGE_VALUE in ${RANGE_LIST}; do
      echo "Running query-gradient + pure score combine: ${ALGORITHM} range=${RANGE_VALUE}"
      (cd "${ROOT}/data_attribution/${ALGORITHM}" && SCORE_INDEX_RANGES="${RANGE_VALUE}" "${PYTHON_BIN}" 02_query_gradient.py && SCORE_INDEX_RANGES="${RANGE_VALUE}" "${PYTHON_BIN}" 03_score.py)
    done
  else
    echo "Running query-gradient + pure score combine: ${ALGORITHM}"
    (cd "${ROOT}/data_attribution/${ALGORITHM}" && "${PYTHON_BIN}" 02_query_gradient.py && "${PYTHON_BIN}" 03_score.py)
  fi
done
