#!/usr/bin/env bash
#SBATCH -J 3d-xq-eps
#SBATCH -o 3d-xq-eps-%j.out
#SBATCH -e 3d-xq-eps-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_cross_query_predicted_noise_updates.py" ]]; do
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
export SOURCE_SCORE_RUN_ID="${SOURCE_SCORE_RUN_ID:-3509662}"
export SCORE_VARIANT="${SCORE_VARIANT:-query_l2}"
NAMESPACE="predicted_noise_cross_query_own_trajectory"
SCORE_RESULTS="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/all_query_own_trajectory_linear_square_root/run_${SOURCE_SCORE_RUN_ID}/linear_square_root_results.csv"
OUT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/cross_query_predicted_noise_updates/run_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_ONLY=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_COUNT=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_NEXT_CHECKPOINT=1
export TRAJ_TRACIN_PROBE_ALIGNMENT_DELTA_CONTINUITY=1

echo "[phase 1/2] cache current-to-next predicted-noise deltas on every query/checkpoint own state"
python "${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --gpus 0,1 --skip-sampling --skip-score --artifact-namespace "${NAMESPACE}" \
  --query-objective trajectory_predicted_noise_probe \
  --predicted-noise-probe-count 1 --num-snapshots 10 \
  --log-prefix cross_query_predicted_noise_updates

echo "[phase 2/2] compare p1/p1, m1/m1, and p1/m1 update directions"
JAX_PLATFORMS=cpu python \
  "${SHAPES_ROOT}/script/analyze_cross_query_predicted_noise_updates.py" \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --namespace "${NAMESPACE}" --score-results "${SCORE_RESULTS}" \
  --variant "${SCORE_VARIANT}" --out-dir "${OUT_DIR}"

echo "[done] cross-query predicted-noise updates: ${OUT_DIR}"
