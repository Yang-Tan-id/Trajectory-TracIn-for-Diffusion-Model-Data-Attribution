#!/usr/bin/env bash
#SBATCH -J 3d-base-rtx
#SBATCH -o 3d-base-rtx-%j.out
#SBATCH -e 3d-base-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

if [[ ! -f "${SHAPES_ROOT}/script/run_3dshapes_experiment.py" ]]; then
  echo "Could not locate the 3D Shapes driver below REPO_ROOT=${REPO_ROOT}" >&2
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
export JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-16}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

echo "3D Shapes base training on TACC RTX-small"
echo "repo=${REPO_ROOT}; experiment=${EXPERIMENT_TAG:-experiment1}; seed=${TRAIN_SEED:-42}"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
nvidia-smi

"${PYTHON_BIN}" script/run_3dshapes_experiment.py \
  --execute \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --epochs "${JAX_EPOCHS:-200}" \
  --base-gpu 0 \
  --skip-prepare \
  --skip-lds-train \
  --skip-sampling \
  --skip-attribution \
  --skip-eval
