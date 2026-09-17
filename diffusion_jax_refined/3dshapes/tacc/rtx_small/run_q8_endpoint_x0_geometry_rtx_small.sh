#!/usr/bin/env bash
#SBATCH -J 3d-q8-x0geom
#SBATCH -o 3d-q8-x0geom-%j.out
#SBATCH -e 3d-q8-x0geom-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q8_endpoint_x0_geometry.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export SOURCE_CHECKPOINT_RUN_ID="${SOURCE_CHECKPOINT_RUN_ID:-3507876}"
NAMESPACE="predicted_noise_endpoint_x0_original12"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q8_endpoint_x0_geometry/run_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_checkpoint_sign_crossfit_bad_queries/run_${SOURCE_CHECKPOINT_RUN_ID}"
mkdir -p "${OUT_DIR}"

export TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_COUNT=12
export TRAJ_TRACIN_PROBE_ALIGNMENT_NEXT_CHECKPOINT=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_FUTURE_MEAN=1

echo "[phase 1/2] cache Q8 predicted noise and true endpoint x0"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-ids 8 --gpus 0 \
  --skip-sampling --skip-score --artifact-namespace "${NAMESPACE}" \
  --query-objective trajectory_predicted_noise_probe \
  --predicted-noise-probe-count 12 --num-snapshots 10 \
  --log-prefix q8_endpoint_x0

unset TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY
unset TRAJ_TRACIN_PROBE_ALIGNMENT_NEXT_CHECKPOINT
unset TRAJ_TRACIN_PROBE_ALIGNMENT_FUTURE_MEAN

echo "[phase 2/2] compare current/next/step noise with x0 directions"
JAX_PLATFORMS=cpu python "${SHAPES_ROOT}/script/analyze_q8_endpoint_x0_geometry.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-id 8 \
  --variant query_train_l2 --method five_bins --namespace "${NAMESPACE}" \
  --checkpoint-crossfit-dir "${CHECKPOINT_DIR}" --out-dir "${OUT_DIR}"

echo "[done] true endpoint x0 geometry: ${OUT_DIR}"
