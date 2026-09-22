#!/usr/bin/env bash
#SBATCH -J 3d-das100q
#SBATCH -o 3d-das100q-%j.out
#SBATCH -e 3d-das100q-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
set -euo pipefail
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"; while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/build_queries.py" ]]; do repo="$(dirname "$repo")"; done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}" XLA_PYTHON_CLIENT_PREALLOCATE=false
experiment="${EXPERIMENT_TAG:-experiment1}"; seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_seed_0_99.json"; ids="$(seq -s, 0 99)"
train_ns="factorized_mc4_reference100x1"; output_ns="factorized_mc4_original100q_100x1"
python "$shapes/script/build_queries.py" --num-queries 100 --output "$query_file"

echo '[1/4] sample Q0-Q99 using the deterministic four-label rule'
python "$shapes/script/run_traj_tracin_queries_and_scores.py" --execute --experiment "$experiment" --train-seed "$seed" --query-file "$query_file" --query-ids "$ids" --gpus 0,1 --skip-query-gradient --skip-score
echo '[2/4] store 100 DAS original endpoint-renoise query gradients and sweep 16 lambdas'
python "$shapes/script/run_das_queries_and_scores.py" --execute --experiment "$experiment" --train-seed "$seed" --query-file "$query_file" --query-ids "$ids" --gpus 0,1 --artifact-namespace original100q_100x1 --train-artifact-namespace "$train_ns" --score-output-namespace "$output_ns" --query-input-mode endpoint_renoise --num-mc-noise 1
echo '[3/4] cache four LDS targets for all 100 queries x 192 subset models'
python "$shapes/script/run_lds_true_f.py" --execute --experiment "$experiment" --train-seed "$seed" --query-file "$query_file" --query-ids "$ids" --gpus 0,1
echo '[4/4] DAS LDS: 100 queries x 16 lambdas x four targets'
JAX_PLATFORMS=cpu python "$shapes/script/run_das_lds_cached.py" --execute --experiment "$experiment" --train-seed "$seed" --query-file "$query_file" --query-ids "$ids" --artifact-namespace "$output_ns" --prediction-sign -1
echo '[done] DAS factorized-MC4 original 100-query pipeline'
