#!/usr/bin/env bash
#SBATCH -J 3d-pn12-lindir
#SBATCH -o 3d-pn12-lindir-%j.out
#SBATCH -e 3d-pn12-lindir-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_predicted_noise_linear_bank_directions.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
ANALYZER="${SHAPES_ROOT}/script/analyze_predicted_noise_linear_bank_directions.py"
[[ -f "${ANALYZER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
QUERY_IDS="${QUERY_IDS:-0,1,2,3,4,5,6,7,8,9}"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/predicted_noise_linear_bank_direction_analysis/run_${SLURM_JOB_ID}"

python "${ANALYZER}" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids "${QUERY_IDS}" \
  --out-dir "${OUT_DIR}"

echo "[done] linear old12/fresh12 direction analysis complete"
