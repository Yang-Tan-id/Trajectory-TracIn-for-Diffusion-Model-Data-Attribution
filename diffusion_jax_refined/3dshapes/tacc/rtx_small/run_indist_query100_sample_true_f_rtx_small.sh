#!/usr/bin/env bash
#SBATCH -J 3d-id100-eval
#SBATCH -o 3d-id100-eval-%j.out
#SBATCH -e 3d-id100-eval-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/build_queries.py" ]]; do
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

python "$shapes/script/build_queries.py" \
  --num-queries 100 --num-zero-queries 20 \
  --seed-start 100 --selection one_per_category \
  --output "$query_file"

echo '[1/2] sample 100 in-distribution queries plus 20 zero-condition queries with the prompted model'
python "$shapes/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0,1 \
  --skip-query-gradient --skip-score

echo '[2/2] cache the four LDS true-f targets for all 100 queries'
python "$shapes/script/run_lds_true_f.py" \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0,1

echo "[done] in-distribution Q0-Q99 plus zero-condition Q100-Q119: $query_file"
