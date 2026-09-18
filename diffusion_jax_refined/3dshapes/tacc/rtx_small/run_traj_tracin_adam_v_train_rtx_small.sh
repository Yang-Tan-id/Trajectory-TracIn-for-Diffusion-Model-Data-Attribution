#!/usr/bin/env bash
#SBATCH -J 3d-adamv-train
#SBATCH -o 3d-adamv-train-%j.out
#SBATCH -e 3d-adamv-train-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py" ]]; do candidate="$(dirname "${candidate}")"; done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
STAGE_DIR="${SHAPES_ROOT}/data_attribution/traj_tracin"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS=200 DATAPOINT_MODEL_MODE=prompted_solo SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo TRAJ_QUERY_OBJECTIVE=trajectory_next_checkpoint_noise_mse
export TRAJ_PARAMETER_SOURCE=raw TRAJ_NUM_SNAPSHOTS=10 TRAJ_TRAIN_MC_SAMPLES=10
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}" TRAJ_TRACIN_PROJ_DIM=4096
export TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=0 TRAJ_TRACIN_TRAIN_BATCH_DTYPE=float32
export TRAJ_TRACIN_TRAIN_BATCH_MODE="${TRAJ_TRACIN_TRAIN_BATCH_MODE:-vmap}"
export TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=adam_v_preconditioned
export JAX_NUM_DEVICES=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false

RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
ARTIFACT="${RESULT_ROOT}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin_adam_v/train_datapoint_gradient_artifact.npz"
PART_DIR="${ARTIFACT}.parts"
LOG_DIR="${RESULT_ROOT}/logs/traj_tracin_adam_v_train/${SLURM_JOB_ID}"
mkdir -p "${PART_DIR}" "${LOG_DIR}"
export TRAJ_TRACIN_STAGE_ARTIFACT_PATH="${ARTIFACT}"

cd "${STAGE_DIR}"
pids=()
for shard in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}" TRAJ_TRACIN_CKPT_SHARD_INDEX="${shard}"
    export TRAJ_TRACIN_CKPT_SHARD_COUNT=2 TRAJ_TRACIN_SKIP_STAGE_MERGE=1
    python 01_train_datapoint_gradient.py
  ) >"${LOG_DIR}/gpu_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "Adam-v train shards failed; inspect ${LOG_DIR}" >&2; exit 1; }

count="$(find "${PART_DIR}" -maxdepth 1 -name 'ckpt_*.npz' -type f | wc -l | tr -d ' ')"
[[ "${count}" == 50 ]] || { echo "Expected 50 parts, found ${count}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0 TRAJ_TRACIN_CKPT_SHARD_INDEX=0 TRAJ_TRACIN_CKPT_SHARD_COUNT=1
export TRAJ_TRACIN_SKIP_STAGE_MERGE=0
python 01_train_datapoint_gradient.py
echo "[done] Adam-v train artifact: ${ARTIFACT}"
