#!/usr/bin/env bash
#SBATCH -J 3d-q8-ownstate
#SBATCH -o 3d-q8-ownstate-%j.out
#SBATCH -e 3d-q8-ownstate-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q8_checkpoint_own_trajectory_states.py" ]]; do
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
export SOURCE_CHECKPOINT_RUN_ID="${SOURCE_CHECKPOINT_RUN_ID:-3508212}"
export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
NAMESPACE="loss_direction_original_f_checkpoint_own_trajectory_states"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q8_checkpoint_own_trajectory_state_geometry/run_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q8_original_f_checkpoint_own_trajectory_crossfit/run_${SOURCE_CHECKPOINT_RUN_ID}"

echo "[phase 1/2] regenerate Q8 own trajectories and persist all 49x10 states"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-ids 8 --gpus 0 \
  --skip-sampling --skip-score --artifact-namespace "${NAMESPACE}" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 10 --log-prefix q8_ownstate

unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY

echo "[phase 2/2] compare every checkpoint/timestamp state with reference trajectory"
JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_q8_checkpoint_own_trajectory_states.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-id 8 --namespace "${NAMESPACE}" \
  --checkpoint-crossfit-dir "${CHECKPOINT_DIR}" \
  --variant query_train_l2 --method five_bins --out-dir "${OUT_DIR}"

echo "[done] Q8 own-trajectory state geometry: ${OUT_DIR}"
