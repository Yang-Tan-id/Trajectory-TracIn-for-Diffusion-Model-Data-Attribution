#!/usr/bin/env bash
#SBATCH -J 3d-das-fmc4
#SBATCH -o 3d-das-fmc4-%j.out
#SBATCH -e 3d-das-fmc4-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/data_attribution/das/01_train_datapoint_gradient.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
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
export DAS_NUM_MC_NOISE=4
export DAS_AGGREGATE_MC_GRADIENT=0
export DAS_AGGREGATE_MC_NORMALIZED=0
export DAS_AGGREGATE_MC_FACTORIZED=1
export DAS_PROJ_DIM=4096
export DAS_DAMPING_SWEEP=1
export DAS_SCORE_DENOMINATOR_CACHE=1
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1

train_namespace="factorized_mc4_reference100x1"
original_namespace="factorized_mc4_original100x1"
reference_namespace="factorized_mc4_generation_reference100x1"
original_query_namespace=""
reference_query_namespace="generation_trajectory100x1"
artifact_dir="$shapes/result/$EXPERIMENT_TAG/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/das_${train_namespace}"
export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact_dir/train_datapoint_gradient_artifact.npz"
export DAS_GLOBAL_GRAM_ARTIFACT_PATH="$TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH"
shard0="$artifact_dir/train_datapoint_gradient_artifact.timestamps_000_049.npz"
shard1="$artifact_dir/train_datapoint_gradient_artifact.timestamps_050_099.npz"
timesteps0="0,10,20,30,40,50,61,71,81,91,101,111,121,131,141,151,161,172,182,192,202,212,222,232,242,252,262,272,283,293,303,313,323,333,343,353,363,373,383,394,404,414,424,434,444,454,464,474,484,494"
timesteps1="505,515,525,535,545,555,565,575,585,595,605,616,626,636,646,656,666,676,686,696,706,716,727,737,747,757,767,777,787,797,807,817,827,838,848,858,868,878,888,898,908,918,928,938,949,959,969,979,989,999"

echo "[phase 1/3] DAS train: 100 timestamps x MC4"
echo "[definition] train feature = mean(residual) * mean(projected gradient); averages are separate"
echo "[artifact] $TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH"
if [[ -f "$TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH" ]]; then
  echo "[skip] complete factorized MC4 train artifact exists"
else
  mkdir -p "$artifact_dir"
  cd "$shapes/data_attribution/das"
  pids=()
  if [[ ! -f "$shard0" ]]; then
    (
      export DAS_TIMESTEPS="$timesteps0"
      export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$shard0"
      export DAS_GLOBAL_GRAM_ARTIFACT_PATH="$shard0"
      CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python 01_train_datapoint_gradient.py
    ) >"$artifact_dir/train_gpu_0.log" 2>&1 &
    pids+=("$!")
  else
    echo "[skip] timestamp shard 0 exists: $shard0"
  fi
  if [[ ! -f "$shard1" ]]; then
    (
      export DAS_TIMESTEPS="$timesteps1"
      export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$shard1"
      export DAS_GLOBAL_GRAM_ARTIFACT_PATH="$shard1"
      CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda python 01_train_datapoint_gradient.py
    ) >"$artifact_dir/train_gpu_1.log" 2>&1 &
    pids+=("$!")
  else
    echo "[skip] timestamp shard 1 exists: $shard1"
  fi
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  JAX_PLATFORMS=cpu python "$repo/diffusion_jax_refined/common/merge_das_term_shards.py" \
    --output "$TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH" \
    "$shard0" "$shard1"
fi

unset DAS_AGGREGATE_MC_FACTORIZED
unset DAS_AGGREGATE_MC_GRADIENT
unset DAS_AGGREGATE_MC_NORMALIZED

echo "[phase 2/3] reuse existing original/reference 100x1 query gradients; score with factorized-MC4 train artifact"
cd "$shapes"
python script/run_das_queries_and_scores.py \
  --execute \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --gpus 0,1 \
  --artifact-namespace "$original_query_namespace" \
  --train-artifact-namespace "$train_namespace" \
  --score-output-namespace "$original_namespace" \
  --query-input-mode endpoint_renoise \
  --num-mc-noise 1 \
  --skip-query-gradient \
  --python-bin python

python script/run_das_queries_and_scores.py \
  --execute \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --gpus 0,1 \
  --artifact-namespace "$reference_query_namespace" \
  --train-artifact-namespace "$train_namespace" \
  --score-output-namespace "$reference_namespace" \
  --query-input-mode generation_trajectory \
  --num-mc-noise 1 \
  --skip-query-gradient \
  --python-bin python

echo "[phase 3/3] cached LDS for four targets"
for namespace in "$original_namespace" "$reference_namespace"; do
  JAX_PLATFORMS=cpu python script/run_das_lds_cached.py \
    --execute \
    --experiment "$EXPERIMENT_TAG" \
    --train-seed "$TRAIN_SEED" \
    --query-ids 0,1,2,3,4,5,6,7,8,9 \
    --artifact-namespace "$namespace" \
    --prediction-sign -1 \
    --python-bin python
done

python script/print_das_original_reference_lambda_changes.py \
  --experiment "$EXPERIMENT_TAG" \
  --original-namespace "$original_namespace" \
  --reference-namespace "$reference_namespace" \
  --prediction-sign m1

echo "[done] factorized MC4 DAS reference100x1 pipeline"
