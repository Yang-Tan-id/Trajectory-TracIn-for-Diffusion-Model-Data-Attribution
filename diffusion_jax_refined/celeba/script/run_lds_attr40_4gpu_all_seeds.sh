#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELEBA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CELEBA_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-attr40_experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export LDS_EPOCHS="${LDS_EPOCHS:-200}"
export LDS_SAVE_EVERY_EPOCHS="${LDS_SAVE_EVERY_EPOCHS:-200}"
export LDS_KEEP_LAST_K="${LDS_KEEP_LAST_K:-1}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export LDS_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

GPU_IDS="${LDS_WORKER_GPUS:-0,1,2,3}"
SUBSET_SEEDS="${LDS_SUBSET_SEEDS:-0 1 2}"
epoch_tag="$(printf '%04d' "${JAX_EPOCHS}")"
base_checkpoint="result/${EXPERIMENT_TAG}/model/prompted_jax/seed_${TRAIN_SEED}_epoch_${epoch_tag}.ckpt"
attribution_indices="../dataset/celeba/10000_attr40/attribution_5k_indices.npy"

[[ -f "${base_checkpoint}" ]] || {
  echo "Missing conditional base checkpoint: ${CELEBA_ROOT}/${base_checkpoint}" >&2
  exit 1
}
[[ -f "${attribution_indices}" ]] || {
  echo "Missing Attr40 attribution universe: ${CELEBA_ROOT}/${attribution_indices}" >&2
  exit 1
}

echo "CelebA-Attr40 LDS: GPUs=${GPU_IDS}; subset seeds=${SUBSET_SEEDS}; m=64; k=2500"
echo "Each model inherits the conditional 40-attribute base training config."
nvidia-smi

for subset_seed in ${SUBSET_SEEDS}; do
  echo "[$(date -Is)] starting Attr40 LDS subset seed ${subset_seed}"
  "${PYTHON_BIN}" lds/run_training_attr40_multi_gpu.py \
    --gpus "${GPU_IDS}" \
    --subset-seed "${subset_seed}" \
    --m 64 \
    --k 2500
  echo "[$(date -Is)] completed Attr40 LDS subset seed ${subset_seed}"
done

echo "[$(date -Is)] completed all 192 conditional CelebA-Attr40 LDS models"
