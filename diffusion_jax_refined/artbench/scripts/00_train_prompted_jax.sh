#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALGORITHMS="${ALGORITHMS:-shared}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"

echo "Running ArtBench prompted JAX latent training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Default diffusion checkpoint dir is result/<experiment>/model/prompted_jax."
echo "Use JAX_PER_ALGORITHM=1 to save under result/<experiment>/model/<algorithm>/prompted_jax."

for ALGORITHM in ${ALGORITHMS}; do
  echo "Running ArtBench prompted JAX training: ${ALGORITHM}"
  python "${ROOT}/../common/prompted_jax_training.py" "${ROOT}/dataset_config.py" --algorithm "${ALGORITHM}"
done
