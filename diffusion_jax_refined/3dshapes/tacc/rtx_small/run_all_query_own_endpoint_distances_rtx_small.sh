#!/usr/bin/env bash
#SBATCH -J 3d-qall-dist
#SBATCH -o 3d-qall-dist-%j.out
#SBATCH -e 3d-qall-dist-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; do
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
export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
export TRAJ_QUERY_CHECKPOINT_OWN_GEOMETRY_ONLY=1
NAMESPACE="loss_direction_original_f_checkpoint_own_endpoint_distances_all10"

echo "[distance-only] generate 49x10 own-trajectory endpoint RMSE; no gradients"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs 200 --query-ids 0,1,2,3,4,5,6,7,8,9 --gpus 0 \
  --skip-sampling --skip-score --artifact-namespace "${NAMESPACE}" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 10 --log-prefix qall_dist

echo "[done] all-query own-endpoint distances"
