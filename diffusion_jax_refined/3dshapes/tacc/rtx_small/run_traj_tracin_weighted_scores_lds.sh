#!/usr/bin/env bash
#SBATCH -J 3d-traj-wlds
#SBATCH -o 3d-traj-wlds-%j.out
#SBATCH -e 3d-traj-wlds-%j.err
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
      if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_traj_tracin_weighted_scores.py" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done
  done
  return 1
}

REPO_ROOT="$(resolve_repo_root)" || {
  echo "Could not locate the weighted Traj TracIn score driver" >&2
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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

cd "${SHAPES_ROOT}"
echo "[1/2] Re-score aligned gradients with constant-LR and cosine-LR+DDIM weighting"
"${PYTHON_BIN}" script/run_traj_tracin_weighted_scores.py \
  --execute \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --query-ids "${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}" \
  --gpus "${WEIGHTED_SCORE_GPUS:-0,1}" \
  --python-bin "${PYTHON_BIN}"

echo "[2/2] Evaluate both new score schemes against the shared true-f cache"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" script/run_traj_tracin_lds_cached.py \
  --execute \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --query-ids "${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}" \
  --score-schemes "constant_lr_uniform,cosine_lr_ddim_step_squared" \
  --python-bin "${PYTHON_BIN}"
