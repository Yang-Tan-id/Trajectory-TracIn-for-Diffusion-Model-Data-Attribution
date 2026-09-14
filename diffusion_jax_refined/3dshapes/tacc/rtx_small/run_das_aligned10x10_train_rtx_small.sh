#!/usr/bin/env bash
#SBATCH -J 3d-das-10x10t
#SBATCH -o 3d-das-10x10t-%j.out
#SBATCH -e 3d-das-10x10t-%j.err
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
  echo "Could not locate the repository" >&2
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

export PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${DAS_GPU:-0}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export DATAPOINT_MODEL_MODE=prompted_solo
export SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo
export DAS_TIMESTEPS="${DAS_TIMESTEPS:-0,111,222,333,444,555,666,777,888,999}"
export DAS_NUM_MC_NOISE=10
export DAS_AGGREGATE_MC_GRADIENT=1
export DAS_PROJ_DIM="${DAS_PROJ_DIM:-4096}"
export DAS_DAMPING_SWEEP=1
export DAS_SCORE_DENOMINATOR_CACHE=1
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export JAX_PLATFORMS=cuda
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export PYTHONUNBUFFERED=1

ARTIFACT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/das_aligned10x10"
export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="${ARTIFACT_DIR}/train_datapoint_gradient_artifact.npz"
export DAS_GLOBAL_GRAM_ARTIFACT_PATH="${TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH}"

echo "3D Shapes DAS: 10 timestamps, each storing one gradient averaged over 10 MC noises"
echo "timestamps=${DAS_TIMESTEPS}; mc_per_timestamp=${DAS_NUM_MC_NOISE}; artifact=${TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH}"
nvidia-smi

cd "${SHAPES_ROOT}/data_attribution/das"
"${PYTHON_BIN}" 01_train_datapoint_gradient.py
