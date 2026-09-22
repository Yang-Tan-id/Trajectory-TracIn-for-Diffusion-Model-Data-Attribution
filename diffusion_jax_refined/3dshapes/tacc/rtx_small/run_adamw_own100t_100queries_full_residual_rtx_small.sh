#!/usr/bin/env bash
#SBATCH -J 3d-own100q
#SBATCH -o 3d-own100q-%j.out
#SBATCH -e 3d-own100q-%j.err
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
query_file="$shapes/queries_seed_0_99.json"; ids="$(seq -s, 0 99)"; namespace="loss_direction_original_f_checkpoint_own_trajectory_100t_100q"
python "$shapes/script/build_queries.py" --num-queries 100 --output "$query_file"
echo '[1/2] store own-trajectory next-raw query gradients: 100 queries x 100 timestamps'
python "$shapes/script/run_traj_tracin_queries_and_scores.py" --execute --experiment "$experiment" --train-seed "$seed" --query-file "$query_file" --query-ids "$ids" --gpus 0,1 --skip-sampling --skip-score --artifact-namespace "$namespace" --query-objective trajectory_next_checkpoint_noise_mse --num-snapshots 100
echo '[2/2] AdamW FOUR full/residual, RAW/QUERY-L2/TRAIN-L2/BOTH-L2, and cached LDS'
out="$shapes/result/$experiment/eval/adamw_own100t_100queries_full_residual/run_${SLURM_JOB_ID}"
JAX_PLATFORMS=cpu python "$shapes/script/run_adamw_four_event_original_f_scores.py" --experiment "$experiment" --train-seed "$seed" --query-file "$query_file" --query-ids "$ids" --query-namespace "$namespace" --attribution-points 5000 --checkpoint-weighting uniform --contraction linear --methods four,four_residual --out-dir "$out"
echo "[done] $out"
