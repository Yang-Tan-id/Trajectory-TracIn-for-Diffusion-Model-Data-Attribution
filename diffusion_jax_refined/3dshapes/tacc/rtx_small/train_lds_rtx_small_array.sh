#!/usr/bin/env bash
#SBATCH -J 3d-lds-rtx
#SBATCH -o 3d-lds-rtx-%j.out
#SBATCH -e 3d-lds-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SUBSET_SEED="${LDS_SUBSET_SEED:?Set LDS_SUBSET_SEED to 0, 1, or 2}"
if [[ ! "${SUBSET_SEED}" =~ ^[012]$ ]]; then
  echo "LDS_SUBSET_SEED must be 0, 1, or 2; got ${SUBSET_SEED}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/../../lds/run_training_multi_gpu.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../../../.." && pwd)"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/3dshapes/lds/run_training_multi_gpu.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
  fi
fi
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
TACC_SCRIPT_DIR="${SHAPES_ROOT}/tacc/rtx_small"

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
echo "subset_seed=${SUBSET_SEED}; independent GPUs=0,1"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
nvidia-smi

"${PYTHON_BIN}" lds/run_training_multi_gpu.py \
  --gpus 0,1 \
  --subset-seed "${SUBSET_SEED}" \
  --m 64 \
  --k 2500

if (( SUBSET_SEED < 2 )) && [[ "${AUTO_SUBMIT_LDS:-1}" == "1" ]]; then
  next_seed="$((SUBSET_SEED + 1))"
  account_args=()
  if [[ -n "${TACC_ACCOUNT:-${ACCOUNT:-}}" ]]; then
    account_args=(-A "${TACC_ACCOUNT:-${ACCOUNT}}")
  fi
  next_job="$(
    sbatch --parsable "${account_args[@]}" \
      --export=ALL,LDS_SUBSET_SEED="${next_seed}" \
      "${TACC_SCRIPT_DIR}/train_lds_rtx_small_array.sh"
  )"
  echo "LDS subset seed ${SUBSET_SEED} complete; submitted seed ${next_seed} as job ${next_job}"
else
  echo "LDS subset seed ${SUBSET_SEED} complete; training pipeline finished"
fi
