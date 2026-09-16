#!/usr/bin/env bash
#SBATCH -J 3d-orgf-tscv
#SBATCH -o 3d-orgf-tscv-%j.out
#SBATCH -e 3d-orgf-tscv-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_original_f_timestamp_sign_crossfit.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
[[ -f "${SHAPES_ROOT}/script/analyze_original_f_timestamp_sign_crossfit.py" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"

PART_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin_loss_direction_residual_rms/train_datapoint_gradient_artifact.npz.parts"
if [[ -d "${PART_DIR}" ]]; then
  part_count="$(find "${PART_DIR}" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
else
  part_count=0
fi
if [[ "${part_count}" != "50" ]]; then
  echo "[phase 1/2] restore forward-only residual-RMS parts (${part_count}/50 present)"
  bash "${SHAPES_ROOT}/tacc/rtx_small/run_traj_tracin_loss_direction_residual_rms_train_rtx_small.sh"
else
  echo "[phase 1/2] reuse 50 residual-RMS parts"
fi

OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_timestamp_sign_crossfit_bad_queries/run_${SLURM_JOB_ID}"
echo "[phase 2/2] Q=1,3,4,8; four normalization variants; repeated two-fold crossfit"
CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python \
  "${SHAPES_ROOT}/script/analyze_original_f_timestamp_sign_crossfit.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids 1,3,4,8 \
  --repeats 20 \
  --random-seed 20260916 \
  --out-dir "${OUT_DIR}"

echo "[done] original-f timestamp-sign crossfit complete: ${OUT_DIR}"
