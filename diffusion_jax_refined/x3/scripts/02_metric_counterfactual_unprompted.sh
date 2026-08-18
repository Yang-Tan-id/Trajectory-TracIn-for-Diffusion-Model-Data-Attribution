#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALGORITHMS="${ALGORITHMS:-das traj_tracin dtrak end_tracin journey_trak}"

for ALGORITHM in ${ALGORITHMS}; do
  echo "Running unprompted counterfactual eval: ${ALGORITHM}"
  python "${ROOT}/../common/unprompted_counterfactual_eval.py" "${ROOT}/dataset_config.py" --algorithm "${ALGORITHM}" --topk "${TOPK:-5000}"
done
