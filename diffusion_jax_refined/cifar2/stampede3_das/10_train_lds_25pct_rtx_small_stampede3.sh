#!/usr/bin/env bash
#SBATCH -J cifar2-s3-lds25-rtx
#SBATCH -o cifar2-s3-lds25-rtx-%j.out
#SBATCH -e cifar2-s3-lds25-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment_67}"
export TRAIN_SEED="${TRAIN_SEED:-67}"

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/stampede3_das" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/_stampede3_das_lib.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_stampede3_das_lib.sh
source "${SCRIPT_DIR}/_stampede3_das_lib.sh"
stampede3_das_init

export STAMPEDE3_SLOT_BACKEND="${STAMPEDE3_SLOT_BACKEND:-local}"
export GPU_PER_NODE="${GPU_PER_NODE:-2}"
export LDS_TRAIN_MAX_PARALLEL="${LDS_TRAIN_MAX_PARALLEL:-2}"
export LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-25}"
export LDS_K="${LDS_K:-2500}"
export LDS_M="${LDS_M:-64}"
export LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 0 7)}"

echo "Stampede3 RTX-small LDS 25% training"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; lds_m=${LDS_M}; lds_k=${LDS_K}; pct=${LDS_DATASET_PERCENTAGE}; seeds=${LDS_SEEDS}"
echo "gpu_per_node=${GPU_PER_NODE}; max_parallel=${LDS_TRAIN_MAX_PARALLEL}"

bash "${SCRIPT_DIR}/01_train_lds_models_stampede3.sh"
