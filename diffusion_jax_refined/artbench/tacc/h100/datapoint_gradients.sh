#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ALGORITHMS="${ALGORITHMS:-${ALGO:-${ALGORITHM:-das}}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export ALGORITHMS
export GPU_IDS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
export JAX_NUM_DEVICES="${JAX_NUM_DEVICES:-$(printf '%s' "${GPU_IDS}" | awk -F',' '{print NF}')}"

echo "h100 datapoint gradients: algorithms=${ALGORITHMS}, train_modes=${TRAIN_MODES:-${TRAIN_MODE:-none}}, gradient_modes=${DATAPOINT_GRADIENT_MODES:-${GRADIENT_MODES:-prompted}}, gpus=${CUDA_VISIBLE_DEVICES}"
exec bash "${DATASET_ROOT}/scripts/01_datapoint_gradients.sh"
