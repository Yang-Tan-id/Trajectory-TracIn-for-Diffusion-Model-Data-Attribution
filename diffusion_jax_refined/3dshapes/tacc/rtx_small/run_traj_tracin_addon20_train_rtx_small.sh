#!/usr/bin/env bash
#SBATCH -J 3d-traj-a20t
#SBATCH -o 3d-traj-a20t-%j.out
#SBATCH -e 3d-traj-a20t-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
resolve_repo_root() {
  local start candidate
  for start in "${REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}"; do
    [[ -n "${start}" && -d "${start}" ]] || continue
    candidate="$(cd "${start}" && pwd)"
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done
  done
  return 1
}

REPO_ROOT="$(resolve_repo_root)" || {
  echo "Could not locate the 3D Shapes Traj TracIn train driver" >&2
  exit 1
}
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
STAGE_DIR="${SHAPES_ROOT}/data_attribution/traj_tracin"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export DATAPOINT_MODEL_MODE=prompted_solo
export SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo
export TRAJ_QUERY_OBJECTIVE=trajectory_next_checkpoint_noise_mse
export TRAJ_PARAMETER_SOURCE=raw
export TRAJ_NUM_SNAPSHOTS=20
# Twenty centered, evenly spaced bins over the 1000-state DDIM trajectory.
# These positions do not overlap the original positions 0,111,...,999.
export TRAJ_SNAPSHOT_POSITIONS=25,75,125,175,225,275,325,375,425,475,525,575,625,675,725,775,825,875,925,975
export TRAJ_TRAIN_MC_SAMPLES=10
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
export TRAJ_TRACIN_PROJ_DIM="${TRAJ_TRACIN_PROJ_DIM:-4096}"
export TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=0
export TRAJ_TRACIN_TRAIN_BATCH_DTYPE="${TRAJ_TRACIN_TRAIN_BATCH_DTYPE:-float32}"
export TRAJ_TRACIN_TRAIN_BATCH_MODE="${TRAJ_TRACIN_TRAIN_BATCH_MODE:-vmap}"
export JAX_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export PYTHONUNBUFFERED=1

ARTIFACT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin_addon20_interleaved"
ARTIFACT="${ARTIFACT_DIR}/train_datapoint_gradient_artifact.npz"
PART_DIR="${ARTIFACT}.parts"
LOG_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/traj_tracin_addon20_train"
export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="${ARTIFACT}"
mkdir -p "${LOG_DIR}" "${PART_DIR}"

checkpoint_count="$(${PYTHON_BIN} - "${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_jax" <<'PY'
from pathlib import Path
import sys
print(len(list(Path(sys.argv[1]).glob("*.ckpt"))))
PY
)"
if [[ "${checkpoint_count}" != "50" ]]; then
  echo "Expected 50 base checkpoints, found ${checkpoint_count}" >&2
  exit 1
fi

if [[ -f "${ARTIFACT}" ]]; then
  echo "Complete add-on 20-timestamp train artifact already exists: ${ARTIFACT}"
  exit 0
fi

echo "3D Shapes Traj TracIn add-on train gradient"
echo "checkpoints=50; new_snapshots=20; mc_per_snapshot=10; terms=1000"
echo "snapshot_positions=${TRAJ_SNAPSHOT_POSITIONS}"
echo "diffusion_timesteps=974,924,874,824,774,724,674,624,574,524,474,424,374,324,274,224,174,124,74,24"
echo "GPU 0: even checkpoint indices; GPU 1: odd checkpoint indices"
echo "artifact=${ARTIFACT}"
nvidia-smi

cd "${STAGE_DIR}"
pids=()
for shard in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    export TRAJ_TRACIN_CKPT_SHARD_INDEX="${shard}"
    export TRAJ_TRACIN_CKPT_SHARD_COUNT=2
    export TRAJ_TRACIN_SKIP_STAGE_MERGE=1
    "${PYTHON_BIN}" 01_train_datapoint_gradient.py
  ) >"${LOG_DIR}/gpu_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "[launch] gpu=${shard} shard=${shard}/2 log=${LOG_DIR}/gpu_${shard}.log"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed != 0 )); then
  echo "At least one add-on checkpoint shard failed; inspect ${LOG_DIR}/gpu_*.log" >&2
  exit 1
fi

part_count="$(find "${PART_DIR}" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
if [[ "${part_count}" != "50" ]]; then
  echo "Expected 50 add-on checkpoint parts, found ${part_count}" >&2
  exit 1
fi

echo "[merge] all ${part_count} checkpoint parts are present"
export CUDA_VISIBLE_DEVICES=0
export TRAJ_TRACIN_CKPT_SHARD_INDEX=0
export TRAJ_TRACIN_CKPT_SHARD_COUNT=1
export TRAJ_TRACIN_SKIP_STAGE_MERGE=0
"${PYTHON_BIN}" 01_train_datapoint_gradient.py

[[ -f "${ARTIFACT}" ]] || {
  echo "Merged add-on artifact was not created: ${ARTIFACT}" >&2
  exit 1
}
echo "[done] merged add-on artifact=${ARTIFACT}"
