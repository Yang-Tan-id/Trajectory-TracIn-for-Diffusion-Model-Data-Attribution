#!/usr/bin/env bash
#SBATCH -J 3d-a10-add
#SBATCH -o 3d-a10-add-%j.out
#SBATCH -e 3d-a10-add-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/data_attribution/traj_tracin/01_train_datapoint_gradient.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
stage="$shapes/data_attribution/traj_tracin"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_SETUP"
elif [[ -f /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh ]]; then
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
else
  echo '[environment] using the currently active Python environment'
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export DATAPOINT_MODEL_MODE=prompted_solo
export SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo
export TRAJ_QUERY_OBJECTIVE=trajectory_next_checkpoint_noise_mse
export TRAJ_PARAMETER_SOURCE=raw
export TRAJ_NUM_SNAPSHOTS=10
# Centered add-on positions. They do not overlap the original
# 0,111,222,333,444,555,666,777,888,999 positions.
export TRAJ_SNAPSHOT_POSITIONS=50,150,250,350,450,550,650,750,850,950
export TRAJ_TRAIN_MC_SAMPLES=10
export TRAJ_TRACIN_PROJ_DIM=4096
export TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=0
export TRAJ_TRACIN_TRAIN_BATCH_DTYPE=float32
export TRAJ_TRACIN_TRAIN_BATCH_MODE="${TRAJ_TRACIN_TRAIN_BATCH_MODE:-vmap}"
export TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=adamw_residual_update
export JAX_NUM_DEVICES=1

result="$shapes/result/$EXPERIMENT_TAG"
namespace=adamw_dual_aligned10x10_addon10
artifact="$result/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin_${namespace}/train_datapoint_gradient_artifact.npz"
parts="${artifact}.parts"
run_id="${SLURM_JOB_ID:-school_$(date +%Y%m%d_%H%M%S)}"
logs="$result/logs/traj_tracin_${namespace}/${run_id}"
export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact"
mkdir -p "$parts" "$logs"

IFS=',' read -r -a train_gpus <<< "${TRAIN_GPUS:-0,1}"
gpu_count="${#train_gpus[@]}"
(( gpu_count > 0 )) || { echo 'TRAIN_GPUS selected no GPUs' >&2; exit 1; }

if [[ -f "$artifact" ]]; then
  echo "[skip] complete add-on AdamW artifact already exists: $artifact"
  exit 0
fi

echo '[train] AdamW-aware add-on: 50 checkpoints x 10 new timestamps x MC10'
echo '[positions] 50,150,250,350,450,550,650,750,850,950'
echo '[timesteps] 949,849,749,649,549,449,349,249,149,49'
echo '[definition] AdamW(mean_MC10 gradient) - AdamW(zero gradient); then CountSketch'
echo '[combine] uniform linear 20t = 0.5 * original10 + 0.5 * addon10'
echo "[gpus] ${TRAIN_GPUS:-0,1}"
echo "[artifact] $artifact"

cd "$stage"
pids=()
for shard in "${!train_gpus[@]}"; do
  gpu="${train_gpus[$shard]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TRAJ_TRACIN_CKPT_SHARD_INDEX="$shard"
    export TRAJ_TRACIN_CKPT_SHARD_COUNT="$gpu_count"
    export TRAJ_TRACIN_SKIP_STAGE_MERGE=1
    "$PYTHON_BIN" 01_train_datapoint_gradient.py
  ) >"$logs/gpu_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || {
  echo "Train shards failed; inspect $logs/gpu_*.log" >&2
  exit 1
}

part_count="$(find "$parts" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
[[ "$part_count" == 50 ]] || {
  echo "Expected 50 checkpoint parts, found $part_count" >&2
  exit 1
}

echo '[merge] all 50 checkpoint parts are present'
export CUDA_VISIBLE_DEVICES="${train_gpus[0]}"
export TRAJ_TRACIN_CKPT_SHARD_INDEX=0
export TRAJ_TRACIN_CKPT_SHARD_COUNT=1
export TRAJ_TRACIN_SKIP_STAGE_MERGE=0
"$PYTHON_BIN" 01_train_datapoint_gradient.py

[[ -f "$artifact" ]] || {
  echo "Merged add-on artifact was not created: $artifact" >&2
  exit 1
}
echo '[cleanup] final artifact verified; removing the 50 add-on checkpoint parts'
find "$parts" -maxdepth 1 -type f -name 'ckpt_*.npz' -delete
rmdir "$parts" 2>/dev/null || true
echo "[done] add-on AdamW aligned10x10 train artifact: $artifact"
