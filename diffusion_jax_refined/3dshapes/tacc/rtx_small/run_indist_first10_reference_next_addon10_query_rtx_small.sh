#!/usr/bin/env bash
#SBATCH -J 3d-id10-a10q
#SBATCH -o 3d-id10-a10q-%j.out
#SBATCH -e 3d-id10-a10q-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY TRAJ_SNAPSHOT_POSITIONS

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="0,1,2,3,4,5,6,7,8,9"
query_namespace="loss_direction_original_f_reference_trajectory_addon10_indist_first10"

cd "$shapes"
echo '[query] ID Q0-Q9 fixed-reference next-checkpoint gradients at add-on positions'
echo '[positions] 50,150,250,350,450,550,650,750,850,950'
echo '[timesteps] 949,849,749,649,549,449,349,249,149,49'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" --epochs 200 \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0,1 \
  --skip-sampling --skip-score --artifact-namespace "$query_namespace" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --snapshot-positions 50,150,250,350,450,550,650,750,850,950 \
  --log-prefix indist_first10_refnext_addon10

echo '[done] ID first10 add-on10 reference-next query gradients'
