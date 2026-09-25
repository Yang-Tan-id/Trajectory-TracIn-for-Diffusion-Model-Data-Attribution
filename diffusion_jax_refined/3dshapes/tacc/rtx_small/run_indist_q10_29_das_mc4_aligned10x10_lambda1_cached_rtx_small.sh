#!/usr/bin/env bash
#SBATCH -J 3d-q10-29-d10
#SBATCH -o 3d-q10-29-d10-%j.out
#SBATCH -e 3d-q10-29-d10-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_das_queries_and_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
python_bin="$(command -v python)"

cd "$shapes"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false

query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 10 29)"
score_namespace="factorized_mc4_indist100q_aligned10x10"

echo "[1/2] crop cached full100 DAS artifacts to timestamps 0,111,...,999; lambda=1"
"$python_bin" script/run_das_queries_and_scores.py \
  --execute \
  --experiment experiment1 \
  --train-seed 42 \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --gpus 0 \
  --artifact-namespace indist100q_original100x1 \
  --train-artifact-namespace factorized_mc4_reference100x1 \
  --score-output-namespace "$score_namespace" \
  --score-contraction squared \
  --query-input-mode endpoint_renoise \
  --num-mc-noise 1 \
  --skip-query-gradient \
  --score-timesteps 0,111,222,333,444,555,666,777,888,999 \
  --damping-values 1

echo "[2/2] cached LDS for Q10-Q29"
JAX_PLATFORMS=cpu "$python_bin" script/run_das_lds_cached.py \
  --execute \
  --experiment experiment1 \
  --train-seed 42 \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --artifact-namespace "$score_namespace" \
  --lambdas 1 \
  --prediction-sign -1

echo "[done] Q10-Q29 DAS train-MC4/query-MC1 aligned10x10 lambda=1"
