#!/usr/bin/env bash
#SBATCH -J 3d-q8-pdir
#SBATCH -o 3d-q8-pdir-%j.out
#SBATCH -e 3d-q8-pdir-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q8_parameter_directional_derivatives.py" ]]; do
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
export TRAJ_QUERY_COLLECT_PARAMETER_DIRECTIONAL_DERIVATIVES=1
NAMESPACE="loss_direction_original_f_checkpoint_own_trajectory_parameter_direction"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q8_parameter_directional_derivatives/run_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q8_original_f_checkpoint_own_trajectory_crossfit/run_${SOURCE_CHECKPOINT_RUN_ID}"

echo "[phase 1/2] compute Q8 query gradients and exact parameter-update derivatives"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-ids 8 --gpus 0 \
  --skip-sampling --skip-score --artifact-namespace "${NAMESPACE}" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 10 --log-prefix q8_pdir

unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY
unset TRAJ_QUERY_COLLECT_PARAMETER_DIRECTIONAL_DERIVATIVES

echo "[phase 2/2] summarize derivative signs by checkpoint bin and timestamp"
JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_q8_parameter_directional_derivatives.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-id 8 --namespace "${NAMESPACE}" \
  --checkpoint-crossfit-dir "${CHECKPOINT_DIR}" \
  --variant query_train_l2 --method five_bins --out-dir "${OUT_DIR}"

echo "[done] Q8 parameter directional derivatives: ${OUT_DIR}"
