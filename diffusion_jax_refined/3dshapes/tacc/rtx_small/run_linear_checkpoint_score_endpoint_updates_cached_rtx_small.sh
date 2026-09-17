#!/usr/bin/env bash
#SBATCH -J 3d-lins-step
#SBATCH -o 3d-lins-step-%j.out
#SBATCH -e 3d-lins-step-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_linear_checkpoint_score_endpoint_updates.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export SOURCE_SCORE_RUN_ID="${SOURCE_SCORE_RUN_ID:-3509662}"
export SCORE_VARIANT="${SCORE_VARIANT:-query_l2}"

SCORE_RESULTS="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/all_query_own_trajectory_linear_square_root/run_${SOURCE_SCORE_RUN_ID}/linear_square_root_results.csv"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/linear_${SCORE_VARIANT}_checkpoint_score_endpoint_updates/run_${SLURM_JOB_ID}"

CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python \
  "${SHAPES_ROOT}/script/analyze_linear_checkpoint_score_endpoint_updates.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --score-results "${SCORE_RESULTS}" \
  --sign-variant "${SCORE_VARIANT}" \
  --out-dir "${OUT_DIR}"

echo "[done] linear checkpoint score/endpoint updates: ${OUT_DIR}"
