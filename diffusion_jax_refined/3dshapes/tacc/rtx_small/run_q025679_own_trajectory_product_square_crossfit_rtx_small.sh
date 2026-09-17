#!/usr/bin/env bash
#SBATCH -J 3d-qrest-sq
#SBATCH -o 3d-qrest-sq-%j.out
#SBATCH -e 3d-qrest-sq-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

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
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
NAMESPACE="loss_direction_original_f_checkpoint_own_trajectory"
QUERY_IDS="0,2,5,6,7,9"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q025679_own_trajectory_product_square/run_${SLURM_JOB_ID}"

echo "[phase 1/2] remaining six own-trajectory original-f query gradients"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-ids "${QUERY_IDS}" --gpus 0 \
  --skip-sampling --skip-score --artifact-namespace "${NAMESPACE}" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 10 --log-prefix q025679_owntraj

unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY

echo "[phase 2/2] product-square LDS and overall-sign crossfit"
CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python \
  "${SHAPES_ROOT}/script/analyze_q1348_own_trajectory_product_square.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --query-ids "${QUERY_IDS}" \
  --original-query-namespace "${NAMESPACE}" \
  --repeats 20 --random-seed 20260916 \
  --out-dir "${OUT_DIR}"

echo "[done] remaining six own-trajectory product-square LDS: ${OUT_DIR}"
