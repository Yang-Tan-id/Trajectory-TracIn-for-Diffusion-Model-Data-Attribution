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
resolve_repo_root() {
  local start candidate
  for start in "${REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}"; do
    [[ -n "${start}" && -d "${start}" ]] || continue
    candidate="$(cd "${start}" && pwd)"
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_3dshapes_experiment.py" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done
  done
  return 1
}
REPO_ROOT="$(resolve_repo_root)" || {
  echo "Could not locate the repository from REPO_ROOT, SLURM_SUBMIT_DIR, or script path" >&2
  exit 1
}
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
TACC_SCRIPT_DIR="${SHAPES_ROOT}/tacc/rtx_small"

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

RAW_DATA_PATH="${THREEDSHAPES_H5:-${REPO_ROOT}/diffusion_jax_refined/dataset/3dshapes/3dshapes.h5}"
PREPARED_DATA_PATH="${REPO_ROOT}/diffusion_jax_refined/dataset/3dshapes/20000/dataset.npz"
if [[ ! -f "${PREPARED_DATA_PATH}" ]]; then
  if [[ ! -f "${RAW_DATA_PATH}" ]]; then
    echo "Missing raw dataset: ${RAW_DATA_PATH}" >&2
    exit 1
  fi
  echo "Prepared dataset not found; preparing from ${RAW_DATA_PATH}"
  "${PYTHON_BIN}" script/prepare_3dshapes.py --input "${RAW_DATA_PATH}"
else
  echo "Using prepared dataset: ${PREPARED_DATA_PATH}"
fi

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

# rtx-small allows only two submitted jobs per user. Submit only the next
# stage here, after base training has succeeded, instead of submitting a
# three-element LDS array up front.
if [[ "${AUTO_SUBMIT_LDS:-0}" == "1" ]]; then
  account_args=()
  if [[ -n "${TACC_ACCOUNT:-${ACCOUNT:-}}" ]]; then
    account_args=(-A "${TACC_ACCOUNT:-${ACCOUNT}}")
  fi
  lds_job="$(
    sbatch --parsable "${account_args[@]}" \
      --export=ALL,LDS_SUBSET_SEED=0 \
      "${TACC_SCRIPT_DIR}/train_lds_rtx_small_array.sh"
  )"
  echo "Base training complete; submitted LDS subset seed 0 as job ${lds_job}"
else
  echo "Base training complete. Submit LDS subset seed 0 from a TACC login node."
fi
