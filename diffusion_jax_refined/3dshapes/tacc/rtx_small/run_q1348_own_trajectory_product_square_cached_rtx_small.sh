#!/usr/bin/env bash
#SBATCH -J 3d-q1348-sq
#SBATCH -o 3d-q1348-sq-%j.out
#SBATCH -e 3d-q1348-sq-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q1348_own_trajectory_product_square.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q1348_own_trajectory_product_square/run_${SLURM_JOB_ID}"

CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python \
  "${SHAPES_ROOT}/script/analyze_q1348_own_trajectory_product_square.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --query-ids 1,3,4,8 \
  --original-query-namespace loss_direction_original_f_checkpoint_own_trajectory \
  --out-dir "${OUT_DIR}"

echo "[done] Q1,Q3,Q4,Q8 own-trajectory product-square LDS: ${OUT_DIR}"
