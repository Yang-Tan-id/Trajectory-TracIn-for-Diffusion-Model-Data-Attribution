#!/usr/bin/env bash
#SBATCH -J 3d-rec-raw2
#SBATCH -o 3d-rec-raw2-%j.out
#SBATCH -e 3d-rec-raw2-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00
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
export TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=none
export TRAJ_TRACIN_STAGE_MODE=train
export TRAJ_TRACIN_SKIP_STAGE_MERGE=1
export JAX_NUM_DEVICES=1

train_root="$shapes/result/$EXPERIMENT_TAG/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient"
artifact="$train_root/traj_tracin_recreate_raw_mc1_aligned10x1/train_datapoint_gradient_artifact.npz"
log_root="$shapes/result/$EXPERIMENT_TAG/logs/recreate_train_raw_last2/${SLURM_JOB_ID}"
mkdir -p "${artifact}.parts" "$log_root"

pids=()
for gpu in 0 1; do
  checkpoint_index="$((48 + gpu))"
  (
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      JAX_PLATFORMS=cuda \
      JAX_NUM_DEVICES=1 \
      TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact" \
      TRAJ_TRACIN_STAGE_ARTIFACT_PATH="$artifact" \
      TRAJ_TRACIN_CKPT_SHARD_INDEX="$checkpoint_index" \
      TRAJ_TRACIN_CKPT_SHARD_COUNT=50 \
      python "$stage"
  ) >"$log_root/gpu_${gpu}_ckpt_${checkpoint_index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || {
  echo "Raw checkpoint 48/49 workers failed; inspect $log_root" >&2
  exit 1
}

count="$(find "${artifact}.parts" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
[[ "$count" == 50 ]] || {
  echo "Expected 50 raw parts before merge, found $count" >&2
  exit 1
}

echo "[merge] all 50 raw checkpoint parts are present"
env \
  CUDA_VISIBLE_DEVICES=0 \
  JAX_PLATFORMS=cuda \
  JAX_NUM_DEVICES=1 \
  TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact" \
  TRAJ_TRACIN_STAGE_ARTIFACT_PATH="$artifact" \
  TRAJ_TRACIN_CKPT_SHARD_INDEX=0 \
  TRAJ_TRACIN_CKPT_SHARD_COUNT=1 \
  TRAJ_TRACIN_SKIP_STAGE_MERGE=0 \
  python "$stage"

[[ -f "$artifact" ]] || {
  echo "Merged raw artifact was not created: $artifact" >&2
  exit 1
}
echo "[done] raw=$artifact"
