#!/usr/bin/env bash
#SBATCH -J 3d-qall-endcv
#SBATCH -o 3d-qall-endcv-%j.out
#SBATCH -e 3d-qall-endcv-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_all_query_endpoint_convergence_by_square_sign.py" ]]; do
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
export SOURCE_SQUARE_RUN_ID="${SOURCE_SQUARE_RUN_ID:?set SOURCE_SQUARE_RUN_ID}"
export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
NAMESPACE="${NAMESPACE:-loss_direction_original_f_checkpoint_own_trajectory_endpoints_all10}"
QUERY_IDS="0,1,2,3,4,5,6,7,8,9"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/all_query_endpoint_convergence_by_square_sign/run_${SLURM_JOB_ID}"
SQUARE_RESULTS="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/q_all_own_trajectory_linear_guided_square/run_${SOURCE_SQUARE_RUN_ID}/results.csv"

if [[ "${ANALYZE_ONLY:-0}" != "1" ]]; then
  echo "[phase 1/2] regenerate all ten checkpoint-own trajectories and persist endpoints"
  python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
    --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --epochs "${JAX_EPOCHS}" --query-ids "${QUERY_IDS}" --gpus 0 \
    --skip-sampling --skip-score --artifact-namespace "${NAMESPACE}" \
    --query-objective trajectory_next_checkpoint_noise_mse \
    --num-snapshots 10 --log-prefix qall_ownend
else
  echo "[reuse] ANALYZE_ONLY=1; using stored checkpoint-own endpoints from ${NAMESPACE}"
fi

unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY

echo "[phase 2/2] compare endpoint convergence by query_train_l2 square sign"
JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_all_query_endpoint_convergence_by_square_sign.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-ids "${QUERY_IDS}" \
  --namespace "${NAMESPACE}" --square-results "${SQUARE_RESULTS}" \
  --variant query_train_l2 --strong-threshold 3.0 --out-dir "${OUT_DIR}"

echo "[done] all-query endpoint convergence by square sign: ${OUT_DIR}"
