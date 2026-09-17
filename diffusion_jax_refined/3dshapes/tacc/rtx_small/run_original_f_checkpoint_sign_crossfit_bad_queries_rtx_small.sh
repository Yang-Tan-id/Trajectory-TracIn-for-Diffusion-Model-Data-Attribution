#!/usr/bin/env bash
#SBATCH -J 3d-orgf-ckcv
#SBATCH -o 3d-orgf-ckcv-%j.out
#SBATCH -e 3d-orgf-ckcv-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_original_f_checkpoint_sign_crossfit.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
[[ -f "${SHAPES_ROOT}/script/analyze_original_f_checkpoint_sign_crossfit.py" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"

OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_checkpoint_sign_crossfit_bad_queries/run_${SLURM_JOB_ID}"
echo "[cached] build 49 checkpoint components, then structured and individual-sign crossfit"
CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python \
  "${SHAPES_ROOT}/script/analyze_original_f_checkpoint_sign_crossfit.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids 1,3,4,8 \
  --repeats 20 \
  --random-seed 20260916 \
  --out-dir "${OUT_DIR}"

echo "[done] original-f checkpoint-sign crossfit complete: ${OUT_DIR}"
