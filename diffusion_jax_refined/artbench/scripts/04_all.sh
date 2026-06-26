#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_INDEX="${1:-0}"
END_INDEX="${2:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
export GPU_IDS="${GPU_IDS:-${CUDA_DEVICE}}"

echo "Running full pipeline on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

bash "${ROOT}/scripts/00_train.sh" "${START_INDEX}" "${END_INDEX}"
bash "${ROOT}/scripts/00_sample.sh"
bash "${ROOT}/scripts/01_data_attribution.sh"
bash "${ROOT}/scripts/02_metric_counterfactual.sh"
bash "${ROOT}/scripts/03_metric_lds.sh"
