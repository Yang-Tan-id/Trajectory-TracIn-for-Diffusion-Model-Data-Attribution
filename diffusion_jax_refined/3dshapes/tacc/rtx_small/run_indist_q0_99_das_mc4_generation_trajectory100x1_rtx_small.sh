#!/usr/bin/env bash
#SBATCH -J 3d-q0-99-dtraj
#SBATCH -o 3d-q0-99-dtraj-%j.out
#SBATCH -e 3d-q0-99-dtraj-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_das_queries_and_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_NUM_DEVICES=1

experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 0 99)"
train_namespace="factorized_mc4_reference100x1"
query_namespace="generation_trajectory100x1"
score_namespace="factorized_mc4_generation_reference100x1"

echo '[1/2] Q0-Q99 DAS query: use each saved generation trajectory x_t at every DAS timestep'
echo '[reuse] factorized train-MC4 artifact: 100 timestamps; existing query artifacts are skipped'
python "$shapes/script/run_das_queries_and_scores.py" \
  --execute \
  --experiment "$experiment" \
  --train-seed "$train_seed" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --gpus 0,1 \
  --artifact-namespace "$query_namespace" \
  --train-artifact-namespace "$train_namespace" \
  --score-output-namespace "$score_namespace" \
  --query-input-mode generation_trajectory \
  --num-mc-noise 1

echo '[2/2] cached LDS: Q0-Q99 x all lambdas x four targets'
JAX_PLATFORMS=cpu python "$shapes/script/run_das_lds_cached.py" \
  --execute \
  --experiment "$experiment" \
  --train-seed "$train_seed" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --artifact-namespace "$score_namespace" \
  --prediction-sign -1

echo '[done] Q0-Q99 factorized-MC4 DAS with direct generation-trajectory x_t'
