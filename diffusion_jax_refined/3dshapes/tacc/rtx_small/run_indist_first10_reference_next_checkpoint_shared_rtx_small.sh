#!/usr/bin/env bash
#SBATCH -J 3d-id10-refshr
#SBATCH -o 3d-id10-refshr-%j.out
#SBATCH -e 3d-id10-refshr-%j.err
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
query_namespace="loss_direction_original_f_reference_trajectory_100t_indist_first10"
score_namespace="checkpoint_shared_100x1_query100"
train_artifact="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient/traj_tracin_checkpoint_shared_100x1/train_datapoint_gradient_artifact.npz"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
[[ -f "$train_artifact" ]] || { echo "Missing checkpoint-shared train artifact: $train_artifact" >&2; exit 1; }

echo '[1/2] ID Q0-Q9: fixed-reference next-checkpoint gradients, 100 timestamps'
echo '[train] checkpoint-shared 100x1 AdamW features extracted from training'
cd "$shapes"
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" --epochs 200 \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0,1 \
  --skip-sampling --artifact-namespace "$query_namespace" \
  --score-output-namespace "$score_namespace" \
  --train-artifact "$train_artifact" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 100 --log-prefix indist_first10_refnext_shared

echo '[2/2] cached LDS for checkpoint-shared reference-next scores'
JAX_PLATFORMS=cpu python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$score_namespace" --prediction-sign=1

echo "[done] ID first10 checkpoint-shared reference-next | query_namespace=$query_namespace"

