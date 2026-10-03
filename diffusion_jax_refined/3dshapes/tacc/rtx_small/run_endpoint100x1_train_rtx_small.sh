#!/usr/bin/env bash
#SBATCH -J 3d-end100-tr
#SBATCH -o 3d-end100-tr-%j.out
#SBATCH -e 3d-end100-tr-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
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
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
python_bin="${PYTHON_BIN:-python}"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
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
unset TRAJ_SNAPSHOT_POSITIONS
export TRAJ_TRAIN_MC_SAMPLES=1
export TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=1
export TRAJ_TRACIN_TRAIN_AGGREGATE_NUM_TIMESTEPS=100
export TRAJ_TRACIN_TRAIN_TIMESTAMP_CHUNK_SIZE="${TRAJ_TRACIN_TRAIN_TIMESTAMP_CHUNK_SIZE:-100}"
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-2}"
export TRAJ_TRACIN_PROJ_DIM=4096
export TRAJ_TRACIN_TRAIN_BATCH_DTYPE=float32
export TRAJ_TRACIN_TRAIN_BATCH_MODE=vmap
export TRAJ_TRACIN_TRAIN_BATCH_LOG_EVERY="${TRAJ_TRACIN_TRAIN_BATCH_LOG_EVERY:-50}"
export TRAJ_TRACIN_STAGE_MODE=train
export TRAJ_TRACIN_SKIP_STAGE_MERGE=1
export JAX_NUM_DEVICES=1

train_root="$shapes/result/$EXPERIMENT_TAG/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient"
artifact="$train_root/traj_tracin_endpoint100x1_raw/train_datapoint_gradient_artifact.npz"
parts="${artifact}.parts"
logs="$shapes/result/$EXPERIMENT_TAG/logs/endpoint100x1_train/${SLURM_JOB_ID}"
mkdir -p "$parts" "$logs"

echo "[definition] endpoint-TracIn train: grad(mean 100 explicit timestep losses x MC1)"
echo "[parallel] GPU0 even checkpoints; GPU1 odd checkpoints; completed parts skipped"

pids=()
for gpu in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
      TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact" \
      TRAJ_TRACIN_STAGE_ARTIFACT_PATH="$artifact" \
      TRAJ_TRACIN_CKPT_SHARD_INDEX="$gpu" \
      TRAJ_TRACIN_CKPT_SHARD_COUNT=2 \
      "$python_bin" "$stage"
  ) >"$logs/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || { echo "train worker failed; inspect $logs" >&2; exit 1; }

count="$(find "$parts" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
[[ "$count" == 50 ]] || { echo "Expected 50 parts, found $count" >&2; exit 1; }
echo "[done] 50 restartable endpoint100x1 train parts: $parts"
echo "[note] no merged duplicate was created"
