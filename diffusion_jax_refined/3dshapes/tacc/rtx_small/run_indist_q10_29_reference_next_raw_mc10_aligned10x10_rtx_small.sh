#!/usr/bin/env bash
#SBATCH -J 3d-q10-29-raw
#SBATCH -o 3d-q10-29-raw-%j.out
#SBATCH -e 3d-q10-29-raw-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

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
export JAX_EPOCHS=200
export DATAPOINT_MODEL_MODE=prompted_solo
export SAMPLE_MODEL_MODE=prompted_solo
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
export TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=none
export TRACIN_SCORE_TIMESTEP_ALLOWLIST="0,111,222,333,444,555,666,777,888,999"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=uniform_checkpoint
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform
export TRACIN_SCORE_CONTRACTION=linear
export JAX_NUM_DEVICES=1

result="$shapes/result/$EXPERIMENT_TAG"
score_namespace="raw_mc10_aligned10x10"
artifact="$result/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/traj_tracin/train_datapoint_gradient_artifact.npz"
parts="${artifact}.parts"
logs="$result/logs/traj_tracin_raw_mc10_aligned10x10/${SLURM_JOB_ID}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 10 29)"
query_namespace="loss_direction_original_f_reference_trajectory_100t_indist100q"

mkdir -p "$logs"
export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact"

echo '[1/3] reuse original raw MC10 x 10t train-side artifact or its 50 cached parts'
echo '[definition] CountSketch(mean_MC10 raw loss gradient); no AdamW transform or optimizer history'
if [[ ! -f "$artifact" ]]; then
  count="$(find "$parts" -maxdepth 1 -type f -name 'ckpt_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$count" != 50 ]]; then
    echo "Missing raw artifact and reusable parts: artifact=$artifact parts=$count/50" >&2
    echo 'This reuse-only job will not silently launch an expensive train-gradient recomputation.' >&2
    exit 1
  fi
  echo "[merge] found all 50 existing raw checkpoint parts under $parts"
  cd "$stage"
  export CUDA_VISIBLE_DEVICES=0
  export TRAJ_TRACIN_CKPT_SHARD_INDEX=0
  export TRAJ_TRACIN_CKPT_SHARD_COUNT=1
  export TRAJ_TRACIN_SKIP_STAGE_MERGE=0
  python 01_train_datapoint_gradient.py >"$logs/merge.log" 2>&1
fi

python - "$artifact" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
with np.load(path, allow_pickle=False) as data:
    shape = data["train_features"].shape
    semantics = str(np.asarray(data["train_feature_semantics"]).item())
    transform = str(np.asarray(data.get("train_optimizer_transform", "none")).item())
    timesteps = sorted(set(np.asarray(data["timesteps"], dtype=np.int64).tolist()))
expected_t = [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]
if shape != (500, 5000, 4096):
    raise SystemExit(f"Unexpected raw train shape: {shape}")
if semantics != "projected_expected_loss_gradient" or transform != "none":
    raise SystemExit(f"Not a raw loss-gradient artifact: semantics={semantics} transform={transform}")
if timesteps != expected_t:
    raise SystemExit(f"Unexpected raw train timestamps: {timesteps}")
print(f"[validated] shape={shape} semantics={semantics} transform={transform} timesteps={timesteps}")
PY

echo '[2/3] Q10-Q29 reference-next linear scores; crop cached 100t queries to aligned10x10'
cd "$shapes"
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$EXPERIMENT_TAG" --train-seed "$TRAIN_SEED" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" \
  --train-artifact "$artifact" \
  --score-output-namespace "$score_namespace" \
  --num-snapshots 100

echo '[3/3] cached LDS for raw/query-L2/train-L2/both-L2'
JAX_PLATFORMS=cpu python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$EXPERIMENT_TAG" --train-seed "$TRAIN_SEED" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$score_namespace" --prediction-sign=1

echo "[done] Q10-Q29 reference-next raw-loss MC10 aligned10x10: $artifact"
