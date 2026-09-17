#!/usr/bin/env bash
#SBATCH -J 3d-q8-ckgeom
#SBATCH -o 3d-q8-ckgeom-%j.out
#SBATCH -e 3d-q8-ckgeom-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q8_checkpoint_sign_predicted_noise_geometry.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
[[ -f "${SHAPES_ROOT}/script/analyze_q8_checkpoint_sign_predicted_noise_geometry.py" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-3507876}"
CROSSFIT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_checkpoint_sign_crossfit_bad_queries/run_${SOURCE_RUN_ID}"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q8_checkpoint_sign_predicted_noise_geometry/run_${SLURM_JOB_ID}"

python "${SHAPES_ROOT}/script/analyze_q8_checkpoint_sign_predicted_noise_geometry.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-id 8 \
  --variant query_train_l2 \
  --method five_bins \
  --checkpoint-crossfit-dir "${CROSSFIT_DIR}" \
  --out-dir "${OUT_DIR}"

echo "[done] Q8 checkpoint-sign predicted-noise geometry: ${OUT_DIR}"
