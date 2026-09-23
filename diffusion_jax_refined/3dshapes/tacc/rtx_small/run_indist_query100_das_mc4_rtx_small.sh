#!/usr/bin/env bash
#SBATCH -J 3d-id100-das4
#SBATCH -o 3d-id100-das4-%j.out
#SBATCH -e 3d-id100-das4-%j.err
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

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 0 119)"
train_namespace="factorized_mc4_reference100x1"
query_namespace="indist100q_original100x1"
score_namespace="factorized_mc4_indist100q_original100x1"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }

echo '[1/2] DAS endpoint-renoise query gradients; reuse 100-timestep factorized train-MC4 artifact'
python "$shapes/script/run_das_queries_and_scores.py" \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0,1 \
  --artifact-namespace "$query_namespace" \
  --train-artifact-namespace "$train_namespace" \
  --score-output-namespace "$score_namespace" \
  --query-input-mode endpoint_renoise --num-mc-noise 1

echo '[2/2] cached DAS LDS: 120 queries x 16 lambdas x four targets'
JAX_PLATFORMS=cpu python "$shapes/script/run_das_lds_cached.py" \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --artifact-namespace "$score_namespace" --prediction-sign -1

echo '[done] factorized train-MC4 DAS for 100 in-distribution plus 20 zero-condition queries'
