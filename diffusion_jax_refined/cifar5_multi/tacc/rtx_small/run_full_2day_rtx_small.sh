#!/usr/bin/env bash
#SBATCH -J cifar5-full-rtx
#SBATCH -o cifar5-full-rtx-%j.out
#SBATCH -e cifar5-full-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CIFAR5_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar5_multi"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh. Set ENV_SETUP=/path/to/env_setup.sh if needed." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_NAME:-experiment_dm}"
fi

cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-cifar5_multi_exp1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export LDS_EPOCHS="${LDS_EPOCHS:-200}"
export JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-16}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export LDS_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

export DAS_PROJ_DIM="${DAS_PROJ_DIM:-4096}"
export DAS_DAMPING_SWEEP="${DAS_DAMPING_SWEEP:-1}"
export TRAJ_TRACIN_PROJ_DIM="${TRAJ_TRACIN_PROJ_DIM:-4096}"
export PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
export PROJECTED_DIMS="${PROJECTED_DIMS:-4096}"
export TRACIN_USE_SHARED_TRAIN_GRADIENT="${TRACIN_USE_SHARED_TRAIN_GRADIENT:-1}"

echo "CIFAR5 multi full RTX-small job"
echo "repo=${REPO_ROOT}"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}"
echo "nodes=${SLURM_JOB_NUM_NODES:-1}; ntasks=${SLURM_NTASKS:-2}; batch=${JAX_BATCH_SIZE}; bf16=${JAX_BFLOAT16}"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

"${PYTHON_BIN}" "${CIFAR5_ROOT}/script/run_cifar5_multi_experiment.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --size "${CIFAR5_MULTI_SIZE:-10000}" \
  --data-seed "${DATA_SEED:-0}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --lds-epochs "${LDS_EPOCHS}" \
  --lds-m "${LDS_M:-64}" \
  --lds-percentage "${LDS_DATASET_PERCENTAGE:-25}" \
  --lds-subset-seeds "${LDS_SUBSET_SEEDS:-0,1,2}" \
  --gpus "${GPU_IDS:-0,1}" \
  --slots "${GPU_SLOTS:-2}" \
  --gpu-per-node 2 \
  --cpus-per-worker "${CPUS_PER_WORKER:-8}" \
  --slot-backend "${TACC_SLOT_BACKEND:-local}" \
  --attribution-algorithms "${ATTRIBUTION_ALGORITHMS:-das,traj_tracin}" \
  ${SKIP_GENERATE:+--skip-generate} \
  ${EXTRA_CIFAR5_ARGS:-}
