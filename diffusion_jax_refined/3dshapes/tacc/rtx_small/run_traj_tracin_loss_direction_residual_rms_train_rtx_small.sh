#!/usr/bin/env bash
#SBATCH -J 3d-loss-rms
#SBATCH -o 3d-loss-rms-%j.out
#SBATCH -e 3d-loss-rms-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

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
  echo "Could not locate repository" >&2
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
export TRAJ_NUM_SNAPSHOTS=10
export TRAJ_TRAIN_MC_SAMPLES=10
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
export TRAJ_TRACIN_PROJ_DIM="${TRAJ_TRACIN_PROJ_DIM:-4096}"
export TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=0
export TRAJ_TRACIN_TRAIN_REUSE_GRADIENT_RESIDUAL_RMS=1
export TRAJ_TRACIN_TRAIN_BATCH_DTYPE="${TRAJ_TRACIN_TRAIN_BATCH_DTYPE:-float32}"
export JAX_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

SOURCE_ARTIFACT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin/train_datapoint_gradient_artifact.npz"
SOURCE_PART_DIR="${SOURCE_ARTIFACT}.parts"
ARTIFACT_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin_loss_direction_residual_rms"
ARTIFACT="${ARTIFACT_DIR}/train_datapoint_gradient_artifact.npz"
PART_DIR="${ARTIFACT}.parts"
LOG_DIR="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/traj_tracin_loss_direction_residual_rms_train"
export TRAJ_TRACIN_SOURCE_TRAIN_ARTIFACT="${SOURCE_ARTIFACT}"
export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="${ARTIFACT}"
mkdir -p "${LOG_DIR}" "${PART_DIR}"

source_count="$(find "${SOURCE_PART_DIR}" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
if [[ "${source_count}" != "50" ]]; then
  echo "Expected 50 source gradient checkpoint parts, found ${source_count}: ${SOURCE_PART_DIR}" >&2
  exit 1
fi

echo "3D Shapes normalized expected-loss direction x matching residual RMS"
echo "No train backward: reuse ${SOURCE_PART_DIR}"
echo "Forward-only residual RMS: checkpoints=50 snapshots=10 MC=10"
echo "GPU 0: even checkpoints; GPU 1: odd checkpoints"
echo "output=${PART_DIR}"
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
  echo "[launch] gpu=${shard} log=${LOG_DIR}/gpu_${shard}.log"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || {
  echo "A forward-only checkpoint shard failed; inspect ${LOG_DIR}/gpu_*.log" >&2
  exit 1
}

part_count="$(find "${PART_DIR}" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
if [[ "${part_count}" != "50" ]]; then
  echo "Expected 50 output checkpoint parts, found ${part_count}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${PART_DIR}" <<'PY'
import sys
from pathlib import Path
import numpy as np

part_dir = Path(sys.argv[1])
for path in (part_dir / "ckpt_0000.npz", part_dir / "ckpt_0049.npz"):
    with np.load(path, allow_pickle=False) as data:
        expected = (10, 5000, 4096)
        if data["train_features"].shape != expected:
            raise SystemExit(f"{path}: expected {expected}, got {data['train_features'].shape}")
        if data["residual_rms"].shape != expected[:2]:
            raise SystemExit(f"{path}: invalid residual_rms shape {data['residual_rms'].shape}")
        semantics = str(np.asarray(data["train_feature_semantics"]).item())
        print(f"[verify] {path.name} features={expected} semantics={semantics}")
PY

echo "[done] retained 50 resumable checkpoint parts=${PART_DIR}"
echo "[done] merged duplicate intentionally omitted"
