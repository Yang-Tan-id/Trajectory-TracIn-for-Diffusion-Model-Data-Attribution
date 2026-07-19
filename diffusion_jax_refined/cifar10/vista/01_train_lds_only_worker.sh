#!/usr/bin/env bash
set -euo pipefail

: "${CIFAR10_ROOT:?CIFAR10_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${BASELINE_CKPT:?BASELINE_CKPT is required}"
: "${EXPERIMENT_TAG:?EXPERIMENT_TAG is required}"
: "${LDS_M:?LDS_M is required}"
: "${LDS_K:?LDS_K is required}"
: "${LDS_EPOCHS:?LDS_EPOCHS is required}"

RANK="${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-0}}"
SEED_VAR="LDS_SEED_$((RANK + 1))"
SEED="${!SEED_VAR:-$((RANK + 1))}"

export CUDA_VISIBLE_DEVICES=0
cd "${CIFAR10_ROOT}"

echo "[rank ${RANK}] running LDS seed ${SEED}"
env \
  EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
  LDS_M="${LDS_M}" \
  LDS_K="${LDS_K}" \
  LDS_SAMPLE_RANDOM_SEED="${SEED}" \
  LDS_EPOCHS="${LDS_EPOCHS}" \
  LDS_DEVICE=gpu \
  LDS_NUM_DEVICES=1 \
  bash scripts/03_lds_training.sh >"${LOG_ROOT}/lds_seed_${SEED}.log" 2>&1

echo "[rank ${RANK}] LDS seed ${SEED} completed"
