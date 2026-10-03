#!/usr/bin/env bash
#SBATCH -J 3d-d20-train
#SBATCH -o 3d-d20-train-%j.out
#SBATCH -e 3d-d20-train-%j.err
#SBATCH -p h100
#SBATCH -A IRI26004
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py" ]]; do
  repo="$(dirname "$repo")"
done
[[ -f "$repo/diffusion_jax_refined/3dshapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py" ]] || {
  echo "Could not locate repository" >&2
  exit 1
}

shapes="$repo/diffusion_jax_refined/3dshapes"
stage="$shapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
python_bin="${PYTHON_BIN:-python}"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CUDNN_USE_AUTOTUNE="${TF_CUDNN_USE_AUTOTUNE:-0}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
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
export TRAJ_NUM_SNAPSHOTS=100
export TRAJ_SNAPSHOT_POSITIONS="$($python_bin -c 'import numpy as np; print(",".join(map(str, np.linspace(0, 999, 100, dtype=np.int32))))')"
export TRAJ_TRAIN_MC_SAMPLES=1
# A direction evaluates 100 timestamps together. Batch 4 is intentionally
# conservative; override with TRAJ_SCORE_BATCH_SIZE only after checking HBM.
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-4}"
export TRAJ_TRACIN_PROJ_DIM=4096
export TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=1
export TRAJ_TRACIN_TRAIN_AGGREGATE_NUM_TIMESTEPS=100
export TRAJ_TRACIN_TRAIN_ALIGNED_DIRECTION_COUNT=20
export TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=adamw_residual_update
export TRAJ_TRACIN_TRAIN_BATCH_DTYPE=float32
export TRAJ_TRACIN_TRAIN_BATCH_MODE=vmap
export TRAJ_TRACIN_TRAIN_BATCH_LOG_EVERY="${TRAJ_TRACIN_TRAIN_BATCH_LOG_EVERY:-100}"
export TRAJ_TRACIN_STAGE_MODE=train
export TRAJ_TRACIN_SKIP_STAGE_MERGE=1
export JAX_NUM_DEVICES=1

train_root="$shapes/result/$EXPERIMENT_TAG/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient"
artifact="$train_root/traj_tracin_direction20_mean100t_adamw_dual/train_datapoint_gradient_artifact.npz"
part_dir="${artifact}.parts"
log_root="$shapes/result/$EXPERIMENT_TAG/logs/direction20_mean100t_train/${SLURM_JOB_ID}"
mkdir -p "$part_dir" "$log_root"

run_slot() {
  local slot="$1"
  shift
  local local_gpu="$((slot % 4))"
  ibrun -n 1 -o "$slot" env \
    CUDA_VISIBLE_DEVICES="$local_gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 "$@"
}

echo "[definition] 50 checkpoints; checkpoint-specific 20 directions; each feature=grad(mean aligned 100-t losses)"
echo "[definition] AdamW residual stored; full=residual+history; checkpoint LR included once"
echo "[parallel] 4 H100 nodes x 4 GPUs = 16 checkpoint shards; completed parts are skipped"
echo "[alignment] positions=$TRAJ_SNAPSHOT_POSITIONS"

pids=()
for slot in $(seq 0 15); do
  (
    run_slot "$slot" env \
      TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact" \
      TRAJ_TRACIN_STAGE_ARTIFACT_PATH="$artifact" \
      TRAJ_TRACIN_CKPT_SHARD_INDEX="$slot" \
      TRAJ_TRACIN_CKPT_SHARD_COUNT=16 \
      "$python_bin" "$stage"
  ) >"$log_root/slot_${slot}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || { echo "Train worker failed; inspect $log_root" >&2; exit 1; }

part_count="$(find "$part_dir" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
[[ "$part_count" == 50 ]] || { echo "Expected 50 train parts, found $part_count" >&2; exit 1; }

"$python_bin" - "$part_dir" <<'PY'
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
for index in (0, 49):
    path = root / f"ckpt_{index:04d}.npz"
    with np.load(path, allow_pickle=False) as z:
        assert z["train_features"].shape == (20, 5000, 4096)
        assert z["optimizer_history_features"].shape == (20, 4096)
        assert np.array_equal(z["direction_indices"], np.arange(20))
        assert z["train_timesteps_used"].shape == (100,)
print("[verified] 50 direction-aligned AdamW checkpoint parts")
PY

echo "[done] train parts=$part_dir"
echo "[note] no merged duplicate is created"
