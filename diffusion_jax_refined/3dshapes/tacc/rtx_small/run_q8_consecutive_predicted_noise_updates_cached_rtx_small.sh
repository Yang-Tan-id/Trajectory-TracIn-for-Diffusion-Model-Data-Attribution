#!/usr/bin/env bash
#SBATCH -J 3d-q8-dupd
#SBATCH -o 3d-q8-dupd-%j.out
#SBATCH -e 3d-q8-dupd-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q8_consecutive_predicted_noise_updates.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export SOURCE_CHECKPOINT_RUN_ID="${SOURCE_CHECKPOINT_RUN_ID:-3507876}"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q8_consecutive_predicted_noise_updates/run_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_checkpoint_sign_crossfit_bad_queries/run_${SOURCE_CHECKPOINT_RUN_ID}"

JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_q8_consecutive_predicted_noise_updates.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --query-id 8 --variant query_train_l2 --method five_bins \
  --namespace predicted_noise_endpoint_x0_original12 \
  --checkpoint-crossfit-dir "${CHECKPOINT_DIR}" --out-dir "${OUT_DIR}"
