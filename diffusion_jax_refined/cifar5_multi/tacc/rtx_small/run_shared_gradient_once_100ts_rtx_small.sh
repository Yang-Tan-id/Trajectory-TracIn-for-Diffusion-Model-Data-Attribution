#!/usr/bin/env bash
#SBATCH -J cifar5-sgonce-rtx
#SBATCH -o cifar5-sgonce-rtx-%j.out
#SBATCH -e cifar5-sgonce-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
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
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export GPU_IDS="${GPU_IDS:-0,1}"
export GPU_SLOTS="${GPU_SLOTS:-2}"
export GPU_PER_NODE="${GPU_PER_NODE:-2}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-2}"
export TACC_SLOT_BACKEND="${TACC_SLOT_BACKEND:-local}"
export ATTRIBUTION_ARTIFACT_NAMESPACE="${ATTRIBUTION_ARTIFACT_NAMESPACE:-raw_nextckpt_rtx_shared_gradient_once_100ts}"

echo "CIFAR5 shared_gradient_once RTX-small job"
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
