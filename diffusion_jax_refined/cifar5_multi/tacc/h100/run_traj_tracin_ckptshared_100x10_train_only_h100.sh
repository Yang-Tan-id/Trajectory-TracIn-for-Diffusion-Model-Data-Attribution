#!/usr/bin/env bash
#SBATCH -J cifar5-traj-train-100x10-h100
#SBATCH -o cifar5-traj-train-100x10-h100-%j.out
#SBATCH -e cifar5-traj-train-100x10-h100-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_attribution_distributed.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
  fi
fi
CIFAR5_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar5_multi"
if [[ ! -f "${CIFAR5_ROOT}/script/run_cifar5_multi_attribution_distributed.py" ]]; then
  echo "Could not locate CIFAR5 attribution driver at ${CIFAR5_ROOT}/script/run_cifar5_multi_attribution_distributed.py" >&2
  exit 1
fi

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
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export LDS_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

export TRAJ_TRACIN_PROJ_DIM="${TRAJ_TRACIN_PROJ_DIM:-4096}"
export PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
export PROJECTED_DIMS="${PROJECTED_DIMS:-4096}"
export TRACIN_USE_SHARED_TRAIN_GRADIENT="${TRACIN_USE_SHARED_TRAIN_GRADIENT:-1}"
export TRAJ_NUM_SNAPSHOTS="${TRAJ_NUM_SNAPSHOTS:-100}"
export TRAJ_TRAIN_MC_SAMPLES="${TRAJ_TRAIN_MC_SAMPLES:-10}"
export ATTRIBUTION_ARTIFACT_NAMESPACE="${ATTRIBUTION_ARTIFACT_NAMESPACE:-h100_traj_ckptshared_100x10_train}"

echo "CIFAR5 multi traj_tracin train-gradient-only H100 job"
echo "repo=${REPO_ROOT}"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; slots=${GPU_SLOTS:-16}"
echo "traj_namespace=${ATTRIBUTION_ARTIFACT_NAMESPACE}"
echo "traj_num_snapshots=${TRAJ_NUM_SNAPSHOTS}; traj_train_mc_samples=${TRAJ_TRAIN_MC_SAMPLES}"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

"${PYTHON_BIN}" "${CIFAR5_ROOT}/script/run_cifar5_multi_attribution_distributed.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --size "${CIFAR5_MULTI_SIZE:-10000}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --artifact-namespace "${ATTRIBUTION_ARTIFACT_NAMESPACE}" \
  --gpus "${GPU_IDS:-0,1,2,3}" \
  --slots "${GPU_SLOTS:-16}" \
  --gpu-per-node 4 \
  --cpus-per-worker "${CPUS_PER_WORKER:-24}" \
  --slot-backend "${TACC_SLOT_BACKEND:-ibrun}" \
  --skip-das \
  --only-train-gradient \
  --max-parallel "${MAX_PARALLEL:-16}" \
  ${EXTRA_CIFAR5_ATTR_ARGS:-}
