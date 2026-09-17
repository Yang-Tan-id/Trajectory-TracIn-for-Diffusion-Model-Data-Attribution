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
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export QUERY_IDS="${QUERY_IDS:-1,3,4,8}"

OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_timestamp_sign_crossfit_bad_queries/run_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

SOURCE_ARTIFACT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin/train_datapoint_gradient_artifact.npz"
SOURCE_PART_DIR="${SOURCE_ARTIFACT}.parts"
if [[ -d "${SOURCE_PART_DIR}" ]]; then
  source_count="$(find "${SOURCE_PART_DIR}" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
else
  source_count=0
fi
if [[ "${source_count}" != "50" ]]; then
  echo "Expected 50 source projected-gradient parts, found ${source_count}: ${SOURCE_PART_DIR}" >&2
  exit 1
fi
echo "[cached] Q=${QUERY_IDS}; direct-loss train gradients; no residual-RMS forward passes"
CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python \
  "${SHAPES_ROOT}/script/analyze_original_f_timestamp_sign_crossfit.py" \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --query-ids "${QUERY_IDS}" \
  --train-namespace traj_tracin \
  --train-feature-semantics raw_projected_expected_loss_gradient \
  --repeats 20 \
  --random-seed 20260916 \
  --out-dir "${OUT_DIR}"

echo "[done] original-f timestamp-sign crossfit complete: ${OUT_DIR}"
