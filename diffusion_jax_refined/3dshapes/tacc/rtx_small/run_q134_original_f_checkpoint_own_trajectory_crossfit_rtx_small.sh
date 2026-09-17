#!/usr/bin/env bash
#SBATCH -J 3d-q134-owntraj
#SBATCH -o 3d-q134-owntraj-%j.out
#SBATCH -e 3d-q134-owntraj-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/summarize_own_trajectory_checkpoint_crossfit.py" ]]; do
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
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q134_original_f_checkpoint_own_trajectory_crossfit/run_${SLURM_JOB_ID}"

echo "[phase 1/3] Q1,Q3,Q4 original-f query gradients on each checkpoint's own trajectory"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-ids 1,3,4 --gpus 0 \
  --skip-sampling --skip-score --artifact-namespace "${NAMESPACE}" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 10 --log-prefix q134_owntraj

unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY

echo "[phase 2/3] build checkpoint components and crossfit signs"
CUDA_VISIBLE_DEVICES=0 JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python \
  "${SHAPES_ROOT}/script/analyze_original_f_checkpoint_sign_crossfit.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --query-ids 1,3,4 --repeats 20 --random-seed 20260916 \
  --original-query-namespace "${NAMESPACE}" --out-dir "${OUT_DIR}"

echo "[phase 3/3] print no-flip and five-bin results"
JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/summarize_own_trajectory_checkpoint_crossfit.py" \
  --summary "${OUT_DIR}/summary.csv"

echo "[done] Q1,Q3,Q4 checkpoint-own-trajectory crossfit: ${OUT_DIR}"
