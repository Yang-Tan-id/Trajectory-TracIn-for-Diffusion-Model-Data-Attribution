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

echo "vista script_0: train_modes=${TRAIN_MODES}, gpus=${CUDA_VISIBLE_DEVICES}, experiment=${EXPERIMENT_TAG:-experiment1}"
exec bash "${DATASET_ROOT}/scripts/script_0.sh" ${TRAIN_MODES//,/ }
