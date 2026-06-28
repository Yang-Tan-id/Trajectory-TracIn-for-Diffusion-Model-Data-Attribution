#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALGORITHMS="${ALGORITHMS:-das traj_tracin dtrak end_tracin journey_trak}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"

for ALGORITHM in ${ALGORITHMS}; do
  if [[ "${LDS_PROXY:-0}" == "1" ]]; then
    echo "Running unprompted LDS proxy (smoke test): ${ALGORITHM}"
    python "${ROOT}/../common/unprompted_lds_eval.py" "${ROOT}/dataset_config.py" --algorithm "${ALGORITHM}" --m "${LDS_M:-100}" --subset-size "${LDS_SUBSET_SIZE:-5000}" --subset-seed "${LDS_SUBSET_SEED:-0}"
  else
    echo "Running full unprompted LDS with subset retraining: ${ALGORITHM}"
    ALGORITHM="${ALGORITHM}" python "${ROOT}/../common/command_runner.py" "${ROOT}/lds/CONFIG_unprompted.py" lds
  fi
done
