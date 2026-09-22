#!/usr/bin/env bash
#SBATCH -J 3d-q100-prep
#SBATCH -o 3d-q100-prep-%j.out
#SBATCH -e 3d-q100-prep-%j.err
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
python "$shapes/script/build_queries.py" --num-queries 100 --output "$query_file"

echo '[1/2] sample Q0-Q99 using the deterministic four-label rule'
python "$shapes/script/run_traj_tracin_queries_and_scores.py" --execute --experiment "$experiment" --train-seed "$seed" --query-file "$query_file" --query-ids "$ids" --gpus 0,1 --skip-query-gradient --skip-score
echo '[2/2] cache four LDS targets shared by DAS and own-trajectory scoring'
python "$shapes/script/run_lds_true_f.py" --execute --experiment "$experiment" --train-seed "$seed" --query-file "$query_file" --query-ids "$ids" --gpus 0,1
echo '[done] Q0-Q99 samples and shared true-f cache'
