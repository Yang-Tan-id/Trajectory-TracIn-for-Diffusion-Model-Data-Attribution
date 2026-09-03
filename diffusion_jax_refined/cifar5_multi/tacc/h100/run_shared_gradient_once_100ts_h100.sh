#!/usr/bin/env bash
#SBATCH -J cifar5-sgonce-h100
#SBATCH -o cifar5-sgonce-h100-%j.out
#SBATCH -e cifar5-sgonce-h100-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_shared_gradient_once.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
  fi
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
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_SLOTS="${GPU_SLOTS:-16}"
export GPU_PER_NODE="${GPU_PER_NODE:-4}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-24}"
export MAX_PARALLEL="${MAX_PARALLEL:-16}"
export TACC_SLOT_BACKEND="${TACC_SLOT_BACKEND:-ibrun}"
export ATTRIBUTION_ARTIFACT_NAMESPACE="${ATTRIBUTION_ARTIFACT_NAMESPACE:-raw_nextckpt_h100_shared_gradient_once_100ts}"

echo "CIFAR5 shared_gradient_once H100 job"
echo "repo=${REPO_ROOT}"
echo "namespace=${ATTRIBUTION_ARTIFACT_NAMESPACE}"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

"${PYTHON_BIN}" diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_shared_gradient_once.py \
  --execute \
  --namespace "${ATTRIBUTION_ARTIFACT_NAMESPACE}" \
  --gpus "${GPU_IDS}" \
  --slots "${GPU_SLOTS}" \
  --gpu-per-node "${GPU_PER_NODE}" \
  --cpus-per-worker "${CPUS_PER_WORKER}" \
  --slot-backend "${TACC_SLOT_BACKEND}" \
  --max-parallel "${MAX_PARALLEL}" \
  ${EXTRA_CIFAR5_SHARED_GRADIENT_ONCE_ARGS:-}
