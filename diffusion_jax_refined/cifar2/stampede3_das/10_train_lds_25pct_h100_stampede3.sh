#!/usr/bin/env bash
#SBATCH -J cifar2-s3-lds25
#SBATCH -o cifar2-s3-lds25-%j.out
#SBATCH -e cifar2-s3-lds25-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 12:00:00

set -euo pipefail

export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_67}"
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

export LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-25}"
export LDS_K="${LDS_K:-2500}"
export LDS_M="${LDS_M:-64}"
export LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 0 7)}"

echo "Stampede3 H100 LDS 25% training"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; lds_m=${LDS_M}; lds_k=${LDS_K}; pct=${LDS_DATASET_PERCENTAGE}; seeds=${LDS_SEEDS}"

bash "${SCRIPT_DIR}/01_train_lds_models_stampede3.sh"
