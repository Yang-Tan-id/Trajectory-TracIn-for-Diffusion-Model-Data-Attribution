#!/usr/bin/env bash
#SBATCH -J celeba-base-rtx
#SBATCH -o celeba-base-rtx-%j.out
#SBATCH -e celeba-base-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CELEBA_ROOT="${REPO_ROOT}/diffusion_jax_refined/celeba"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-16}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

DATA_ROOT="${REPO_ROOT}/diffusion_jax_refined/dataset/celeba/10000"
DATASET_PATH="${DATA_ROOT}/dataset.npz"
IMAGES_ZIP="${CELEBA_IMAGES_ZIP:-${REPO_ROOT}/diffusion_jax_refined/dataset/celeba/img_align_celeba.zip}"
PARTITION="${CELEBA_PARTITION:-${REPO_ROOT}/diffusion_jax_refined/dataset/celeba/list_eval_partition.txt}"

cd "${CELEBA_ROOT}"
if [[ ! -f "${DATASET_PATH}" ]]; then
  [[ -f "${IMAGES_ZIP}" ]] || { echo "Missing CelebA archive: ${IMAGES_ZIP}" >&2; exit 1; }
  [[ -f "${PARTITION}" ]] || { echo "Missing CelebA partition: ${PARTITION}" >&2; exit 1; }
  "${PYTHON_BIN}" script/prepare_celeba.py \
    --images-zip "${IMAGES_ZIP}" \
    --partition "${PARTITION}" \
    --out-root "${DATA_ROOT}"
fi

echo "CelebA base training: seed=${TRAIN_SEED}, epochs=${JAX_EPOCHS}, batch=${JAX_BATCH_SIZE}"
nvidia-smi
"${PYTHON_BIN}" training/run_training.py
