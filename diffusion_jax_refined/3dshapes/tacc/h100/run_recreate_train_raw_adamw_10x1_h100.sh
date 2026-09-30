#!/usr/bin/env bash
#SBATCH -J 3d-rec-train
#SBATCH -o 3d-rec-train-%j.out
#SBATCH -e 3d-rec-train-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py" ]]; do
  repo="$(dirname "$repo")"
done
[[ -f "$repo/diffusion_jax_refined/3dshapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py" ]] || {
  echo "Could not locate repository" >&2; exit 1;
}

shapes="$repo/diffusion_jax_refined/3dshapes"
stage="$shapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CUDNN_USE_AUTOTUNE=0
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export DATAPOINT_MODEL_MODE=prompted_solo
export SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo
export QUERY="shape_cube,object_hue_0,wall_hue_0,floor_hue_0"
export INITIAL_SEED=0 SAMPLE_SEED=0
export TRAJ_QUERY_OBJECTIVE=trajectory_next_checkpoint_noise_mse
export TRAJ_PARAMETER_SOURCE=raw
export TRAJ_NUM_SNAPSHOTS=10
export TRAJ_TRAIN_MC_SAMPLES=1
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-16}"
export TRAJ_TRACIN_PROJ_DIM="${TRAJ_TRACIN_PROJ_DIM:-4096}"
export TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=0
export TRAJ_TRACIN_TRAIN_BATCH_DTYPE=float32
export TRAJ_TRACIN_TRAIN_BATCH_MODE="${TRAJ_TRACIN_TRAIN_BATCH_MODE:-vmap}"
export TRAJ_TRACIN_TRAIN_NOISE_MODE=checkpoint_timestamp_shared
export TRAJ_TRACIN_STAGE_MODE=train
export TRAJ_TRACIN_SKIP_STAGE_MERGE=1
export JAX_NUM_DEVICES=1

train_root="$shapes/result/$EXPERIMENT_TAG/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient"
raw_artifact="$train_root/traj_tracin_recreate_raw_mc1_aligned10x1/train_datapoint_gradient_artifact.npz"
adamw_artifact="$train_root/traj_tracin_recreate_adamw_dual_mc1_aligned10x1/train_datapoint_gradient_artifact.npz"
log_root="$shapes/result/$EXPERIMENT_TAG/logs/recreate_train_10x1/${SLURM_JOB_ID}"
mkdir -p "${raw_artifact}.parts" "${adamw_artifact}.parts" "$log_root"

run_slot() {
  local slot="$1"; shift
  local local_gpu="$((slot % 4))"
  ibrun -n 1 -o "$slot" env \
    CUDA_VISIBLE_DEVICES="$local_gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 "$@"
}

run_family() {
  local label="$1" artifact="$2" transform="$3"
  local pids=() slot failed=0
  echo "[train:$label] artifact=$artifact transform=$transform"
  for slot in $(seq 0 15); do
    (
      run_slot "$slot" env \
        TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact" \
        TRAJ_TRACIN_STAGE_ARTIFACT_PATH="$artifact" \
        TRAJ_TRACIN_CKPT_SHARD_INDEX="$slot" \
        TRAJ_TRACIN_CKPT_SHARD_COUNT=16 \
        TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM="$transform" \
        python "$stage"
    ) >"$log_root/${label}_slot_${slot}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "$label workers failed; inspect $log_root" >&2; exit 1; }

  local count
  count="$(find "${artifact}.parts" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
  [[ "$count" == 50 ]] || { echo "$label expected 50 parts, found $count" >&2; exit 1; }

  echo "[train:$label] merge 50 resumable checkpoint parts"
  env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact" \
    TRAJ_TRACIN_STAGE_ARTIFACT_PATH="$artifact" \
    TRAJ_TRACIN_CKPT_SHARD_INDEX=0 TRAJ_TRACIN_CKPT_SHARD_COUNT=1 \
    TRAJ_TRACIN_SKIP_STAGE_MERGE=0 \
    TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM="$transform" \
    python "$stage"
}

echo "[definition] 50 checkpoints x 10 timestamps x MC1; 5000 points; projection=4096"
echo "[projection] deterministic checkpoint-specific CountSketch shared by raw/query/AdamW"
echo "[noise] one train-seed key per checkpoint/timestamp shared by all train points and query"
run_family raw "$raw_artifact" none
run_family adamw_dual "$adamw_artifact" adamw_residual_update

python - "$raw_artifact" "$adamw_artifact" <<'PY'
import sys
import numpy as np

raw_path, adamw_path = sys.argv[1:]
with np.load(raw_path, allow_pickle=False) as raw, np.load(adamw_path, allow_pickle=False) as adamw:
    for key in ("score_indices", "ckpt_indices", "timesteps", "snapshot_positions"):
        if not np.array_equal(raw[key], adamw[key]):
            raise SystemExit(f"raw/AdamW alignment mismatch for {key}")
    if "optimizer_history_features" not in adamw.files:
        raise SystemExit("AdamW dual artifact lacks optimizer_history_features")
    if adamw["optimizer_history_features"].shape != adamw["train_features"].shape[:1] + adamw["train_features"].shape[2:]:
        raise SystemExit("optimizer-history shape does not match AdamW terms")
    print("[verified] raw/residual/full-restoration alignment and optimizer history")
PY

echo "[done] raw=$raw_artifact"
echo "[done] AdamW residual=$adamw_artifact::train_features"
echo "[done] AdamW full=residual + $adamw_artifact::optimizer_history_features"
