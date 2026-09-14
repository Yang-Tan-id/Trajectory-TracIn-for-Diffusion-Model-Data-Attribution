#!/usr/bin/env bash
#SBATCH -J 3d-das-n10q
#SBATCH -o 3d-das-n10q-%j.out
#SBATCH -e 3d-das-n10q-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
candidate="$(cd "${candidate}" && pwd)"
while [[ "${candidate}" != "/" ]]; do
  [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_das_queries_and_scores.py" ]] && break
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
[[ -f "${REPO_ROOT}/diffusion_jax_refined/3dshapes/script/run_das_queries_and_scores.py" ]] || {
  echo "Could not locate repository" >&2
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
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

cd "${SHAPES_ROOT}"
"${PYTHON_BIN}" script/run_das_queries_and_scores.py \
  --execute \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --query-ids "${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}" \
  --gpus "${DAS_QUERY_SCORE_GPUS:-0,1}" \
  --artifact-namespace mc_normalized10x10 \
  --timesteps "${DAS_TIMESTEPS:-0,111,222,333,444,555,666,777,888,999}" \
  --num-mc-noise 10 \
  --aggregate-mc-normalized \
  --python-bin "${PYTHON_BIN}"
