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

namespace="factorized_mc4_reference100x1"
artifact_dir="$shapes/result/$EXPERIMENT_TAG/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/das_${namespace}"
export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$artifact_dir/train_datapoint_gradient_artifact.npz"
export DAS_GLOBAL_GRAM_ARTIFACT_PATH="$TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH"

echo "[phase 1/3] DAS train: 100 timestamps x MC4"
echo "[definition] train feature = mean(residual) * mean(projected gradient); averages are separate"
echo "[artifact] $TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH"
if [[ -f "$TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH" ]]; then
  echo "[skip] complete factorized MC4 train artifact exists"
else
  cd "$shapes/data_attribution/das"
  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python 01_train_datapoint_gradient.py
fi

unset DAS_AGGREGATE_MC_FACTORIZED
unset DAS_AGGREGATE_MC_GRADIENT
unset DAS_AGGREGATE_MC_NORMALIZED

echo "[phase 2/3] standard reference-trajectory 100x1 query; 10 queries; 16 lambdas"
cd "$shapes"
python script/run_das_queries_and_scores.py \
  --execute \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --gpus 0,1 \
  --artifact-namespace "$namespace" \
  --train-artifact-namespace "$namespace" \
  --query-input-mode endpoint_renoise \
  --num-mc-noise 1 \
  --python-bin python

echo "[phase 3/3] cached LDS for four targets"
JAX_PLATFORMS=cpu python script/run_das_lds_cached.py \
  --execute \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --artifact-namespace "$namespace" \
  --prediction-sign -1 \
  --python-bin python

python script/print_das_all_lambdas.py \
  --experiment "$EXPERIMENT_TAG" \
  --artifact-namespace "$namespace" \
  --prediction-sign m1 \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --best-lambda-per-target

echo "[done] factorized MC4 DAS reference100x1 pipeline"
