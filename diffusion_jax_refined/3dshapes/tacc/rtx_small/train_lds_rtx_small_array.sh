#!/usr/bin/env bash
#SBATCH -J 3d-lds-rtx
#SBATCH -o 3d-lds-rtx-%A_%a.out
#SBATCH -e 3d-lds-rtx-%A_%a.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH --array=0-2%1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

if [[ ! -f "${SHAPES_ROOT}/lds/run_training_multi_gpu.py" ]]; then
  echo "Could not locate the 3D Shapes LDS launcher below REPO_ROOT=${REPO_ROOT}" >&2
  exit 1
fi

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

cd "${SHAPES_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export LDS_EPOCHS="${LDS_EPOCHS:-200}"
export LDS_SAVE_EVERY_EPOCHS="${LDS_SAVE_EVERY_EPOCHS:-200}"
export LDS_KEEP_LAST_K="${LDS_KEEP_LAST_K:-1}"
export JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-16}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export LDS_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

echo "3D Shapes LDS training on TACC RTX-small"
echo "repo=${REPO_ROOT}; experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}"
echo "subset_seed=${SLURM_ARRAY_TASK_ID}; independent GPUs=0,1"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
nvidia-smi

"${PYTHON_BIN}" lds/run_training_multi_gpu.py \
  --gpus 0,1 \
  --subset-seed "${SLURM_ARRAY_TASK_ID}" \
  --m 64 \
  --k 2500
