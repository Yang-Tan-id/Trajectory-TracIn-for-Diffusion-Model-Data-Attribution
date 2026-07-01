#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"

echo "Running unconditional JAX sampling on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
UNPROMPTED=1 python "${ROOT}/sampling/run_sampling.py"
