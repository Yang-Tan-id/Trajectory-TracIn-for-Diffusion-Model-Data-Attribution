#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"

echo "vista sample: experiment=${EXPERIMENT_TAG:-experiment1}, model_mode=${SAMPLE_MODEL_MODE:-prompted_solo}, seeds=${SAMPLE_SEEDS:-0}, gpus=${CUDA_VISIBLE_DEVICES}"
exec bash "${DATASET_ROOT}/scripts/00_sample_for_attribution.sh"
