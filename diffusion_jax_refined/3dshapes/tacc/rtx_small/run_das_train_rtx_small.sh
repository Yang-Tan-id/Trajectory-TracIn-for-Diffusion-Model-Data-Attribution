#!/usr/bin/env bash
#SBATCH -J 3d-das-train
#SBATCH -o 3d-das-train-%j.out
#SBATCH -e 3d-das-train-%j.err
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
      if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/data_attribution/das/01_train_datapoint_gradient.py" ]]; then
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

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

cd "${SHAPES_ROOT}/data_attribution/das"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${DAS_GPU:-0}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export DATAPOINT_MODEL_MODE=prompted_solo
export SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo
export DAS_NUM_MC_NOISE="${DAS_NUM_MC_NOISE:-1}"
export DAS_PROJ_DIM="${DAS_PROJ_DIM:-4096}"
export DAS_DAMPING_SWEEP=1
export DAS_SCORE_DENOMINATOR_CACHE=1
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export PYTHONUNBUFFERED=1

echo "3D Shapes DAS train-gradient on TACC RTX-small"
echo "repo=${REPO_ROOT}; experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}"
echo "gpu=${CUDA_VISIBLE_DEVICES}; timestamps=100; mc_per_timestamp=${DAS_NUM_MC_NOISE}; proj_dim=${DAS_PROJ_DIM}"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
nvidia-smi

"${PYTHON_BIN}" 01_train_datapoint_gradient.py
