#!/usr/bin/env bash
#SBATCH -J 3d-das-lin100
#SBATCH -o 3d-das-lin100-%j.out
#SBATCH -e 3d-das-lin100-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_das_queries_and_scores.py" ]]; do
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
echo "[phase 1/3] reuse standard DAS 100x1 artifacts; compute unsquared linear scores"
"${PYTHON_BIN}" script/run_das_queries_and_scores.py \
  --execute \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --query-ids "${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}" \
  --gpus "${DAS_QUERY_SCORE_GPUS:-0,1}" \
  --skip-query-gradient \
  --score-output-namespace linear100x1 \
  --score-contraction linear \
  --python-bin "${PYTHON_BIN}"

echo "[phase 2/3] cached LDS with signed linear prediction"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" script/run_das_lds_cached.py \
  --execute \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --query-ids "${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}" \
  --artifact-namespace linear100x1 \
  --prediction-sign 1

echo "[phase 3/3] print shared-lambda counterfactual results"
"${PYTHON_BIN}" script/print_das_shared_lambda.py \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --query-ids "${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}" \
  --artifact-namespace linear100x1 \
  --prediction-sign p1

echo "[done] DAS linear 100x1 scores and LDS complete"
