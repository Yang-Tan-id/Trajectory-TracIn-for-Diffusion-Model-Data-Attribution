#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ALGORITHMS="${ALGORITHMS:-${ALGO:-${ALGORITHM:-das}}}"
GPU_IDS="${GPU_IDS:-0}"
export ALGORITHMS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"

echo "h100 sample+query-gradient: experiment=${EXPERIMENT_TAG:-experiment1}, model_mode=${SAMPLE_MODEL_MODE:-prompted_solo}, seeds=${SAMPLE_SEEDS:-${INITIAL_SEED:-0}}, algorithms=${ALGORITHMS}, gpus=${CUDA_VISIBLE_DEVICES}"
exec bash "${DATASET_ROOT}/scripts/02_sample_query_gradient.sh"
