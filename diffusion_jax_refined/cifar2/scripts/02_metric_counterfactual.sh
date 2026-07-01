#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALGORITHMS="${ALGORITHMS:-das traj_tracin dtrak end_tracin journey_trak}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"

echo "Running counterfactual metrics on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for ALGORITHM_VALUE in ${ALGORITHMS}; do
  echo "Running counterfactual metric: ${ALGORITHM_VALUE}"
  ALGORITHM="${ALGORITHM_VALUE}" python "${ROOT}/counterfactual/run_counterfactual.py"
done
