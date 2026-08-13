#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_MODE="${TRAIN_MODE:-prompted_multi}"
TRAIN_MODES="${TRAIN_MODES:-${TRAIN_MODE}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_IDS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
export JAX_NUM_DEVICES="${JAX_NUM_DEVICES:-$(printf '%s' "${GPU_IDS}" | awk -F',' '{print NF}')}"

unset PYTHONPATH
if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-${SCRATCH}/conda-envs/trajectory-tracin}"
  if [[ -f "${SCRATCH}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh under \${SCRATCH}/miniforge3 or \${HOME}/miniforge3." >&2
    echo "Set ENV_SETUP=/path/to/setup.sh or CONDA_ENV_PATH=/path/to/env." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_PATH}"
fi

echo "python=$(command -v python)"
echo "h100 script_0: train_modes=${TRAIN_MODES}, gpus=${CUDA_VISIBLE_DEVICES}, experiment=${EXPERIMENT_TAG:-experiment1}"
exec bash "${DATASET_ROOT}/scripts/script_0.sh" ${TRAIN_MODES//,/ }
