#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LDS_MODEL_DIRS:?Set LDS_MODEL_DIRS to one or more comma-separated lds_model folders}"
ALGORITHMS="${ALGORITHMS:-das traj_tracin dtrak end_tracin journey_trak}"

for ALGORITHM_VALUE in ${ALGORITHMS}; do
  python "${ROOT}/lds/run_eval.py" \
    --algorithm "${ALGORITHM_VALUE}" \
    --lds-model-dirs "${LDS_MODEL_DIRS}" \
    "$@"
done
