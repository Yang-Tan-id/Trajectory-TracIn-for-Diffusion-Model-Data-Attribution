#!/usr/bin/env bash
#SBATCH -J 3d-d20-train
#SBATCH -o 3d-d20-train-%j.out
#SBATCH -e 3d-d20-train-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
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
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }

export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
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
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-2}"
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

echo "[definition] checkpoint-specific 20 noise directions; each train feature is grad(mean of aligned 100-t losses)"
echo "[definition] AdamW residual stored; full = residual + optimizer_history_features; checkpoint LR already included"
echo "[alignment] positions=$TRAJ_SNAPSHOT_POSITIONS"
echo "[performance] two checkpoint shards; each checkpoint restored once; completed parts are skipped"
nvidia-smi

pids=()
for gpu in 0 1; do
  (
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      JAX_PLATFORMS=cuda \
      JAX_NUM_DEVICES=1 \
      TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact" \
      TRAJ_TRACIN_STAGE_ARTIFACT_PATH="$artifact" \
      TRAJ_TRACIN_CKPT_SHARD_INDEX="$gpu" \
      TRAJ_TRACIN_CKPT_SHARD_COUNT=2 \
      "$python_bin" "$stage"
  ) >"$log_root/gpu_${gpu}.log" 2>&1 &
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
        assert z["train_features"].shape == (20, 5000, 4096), (path, z["train_features"].shape)
        assert z["optimizer_history_features"].shape == (20, 4096)
        assert np.array_equal(z["direction_indices"], np.arange(20))
        assert z["train_timesteps_used"].shape == (100,)
print("[verified] 50 resumable parts; direction/timestep/full-AdamW metadata present")
PY

echo "[done] train parts=$part_dir"
echo "[note] merged artifact intentionally omitted to avoid duplicating the large checkpoint parts"
