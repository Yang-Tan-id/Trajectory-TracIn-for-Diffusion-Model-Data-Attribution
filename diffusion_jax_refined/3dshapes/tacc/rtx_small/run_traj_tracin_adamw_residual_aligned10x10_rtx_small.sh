#!/usr/bin/env bash
#SBATCH -J 3d-ares-align
#SBATCH -o 3d-ares-align-%j.out
#SBATCH -e 3d-ares-align-%j.err
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
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS=200 DATAPOINT_MODEL_MODE=prompted_solo SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo
export TRAJ_QUERY_OBJECTIVE=trajectory_next_checkpoint_noise_mse
export TRAJ_PARAMETER_SOURCE=raw
export TRAJ_NUM_SNAPSHOTS=10
export TRAJ_TRAIN_MC_SAMPLES=10
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
export TRAJ_TRACIN_PROJ_DIM=4096
export TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=0
export TRAJ_TRACIN_TRAIN_BATCH_DTYPE=float32
export TRAJ_TRACIN_TRAIN_BATCH_MODE="${TRAJ_TRACIN_TRAIN_BATCH_MODE:-vmap}"
export TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=adamw_residual_update
export JAX_NUM_DEVICES=1

result="$shapes/result/$EXPERIMENT_TAG"
namespace="adamw_residual_aligned10x10"
artifact="$result/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin_${namespace}/train_datapoint_gradient_artifact.npz"
parts="${artifact}.parts"
logs="$result/logs/traj_tracin_${namespace}/${SLURM_JOB_ID}"
mkdir -p "$parts" "$logs"
export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact"

echo '[phase 1/3] aligned train features: 50 checkpoints x 10 timestamps x MC10'
echo '[definition] AdamW(mean_MC10 gradient) - AdamW(zero gradient); then CountSketch'
echo '[weighting] AdamW internal checkpoint LR retained; outer checkpoint LR omitted; timestamps averaged'

cd "$stage"
pids=()
for shard in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES="$shard"
    export TRAJ_TRACIN_CKPT_SHARD_INDEX="$shard"
    export TRAJ_TRACIN_CKPT_SHARD_COUNT=2
    export TRAJ_TRACIN_SKIP_STAGE_MERGE=1
    python 01_train_datapoint_gradient.py
  ) >"$logs/gpu_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "train shards failed; inspect $logs" >&2; exit 1; }

count="$(find "$parts" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
[[ "$count" == 50 ]] || { echo "Expected 50 parts, found $count" >&2; exit 1; }

echo '[phase 2/3] merge checkpoint parts'
export CUDA_VISIBLE_DEVICES=0 TRAJ_TRACIN_CKPT_SHARD_INDEX=0 TRAJ_TRACIN_CKPT_SHARD_COUNT=1
export TRAJ_TRACIN_SKIP_STAGE_MERGE=0
python 01_train_datapoint_gradient.py

echo '[phase 3/3] reuse original aligned query gradients; score four normalization variants; LDS'
cd "$shapes"
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$EXPERIMENT_TAG" --train-seed "$TRAIN_SEED" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 --gpus 0,1 \
  --skip-sampling --skip-query-gradient \
  --train-artifact "$artifact" \
  --score-output-namespace "$namespace"

JAX_PLATFORMS=cpu python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$EXPERIMENT_TAG" --train-seed "$TRAIN_SEED" \
  --score-schemes "$namespace" --prediction-sign=1

echo "[done] aligned MC10 AdamW-residual Traj TracIn: $artifact"
