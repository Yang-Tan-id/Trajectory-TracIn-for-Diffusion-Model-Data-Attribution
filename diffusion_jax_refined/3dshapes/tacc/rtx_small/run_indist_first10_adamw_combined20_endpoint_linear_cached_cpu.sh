#!/usr/bin/env bash
#SBATCH -J 3d-id10-a20lin
#SBATCH -o 3d-id10-a20lin-%j.out
#SBATCH -e 3d-id10-a20lin-%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/combine_traj_score_namespaces.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu JAX_NUM_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=uniform_checkpoint
export TRACIN_SCORE_TIMESTEP_WEIGHTING=endpoint_linear
export TRACIN_SCORE_TIMESTEPS_TOTAL=1000

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="0,1,2,3,4,5,6,7,8,9"
query_namespace="loss_direction_original_f_reference_trajectory_100t_indist_first10"
train_root="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient"
original_artifact="$train_root/traj_tracin_adamw_dual_aligned10x10/train_datapoint_gradient_artifact.npz"
addon_artifact="$train_root/traj_tracin_adamw_dual_aligned10x10_addon10/train_datapoint_gradient_artifact.npz"

[[ -f "$original_artifact" ]] || { echo "Missing original10 artifact: $original_artifact" >&2; exit 1; }
[[ -f "$addon_artifact" ]] || { echo "Missing addon10 artifact: $addon_artifact" >&2; exit 1; }

cd "$shapes"

score_grid() {
  local artifact="$1" residual_namespace="$2" full_namespace="$3"
  echo "[score] residual namespace=$residual_namespace weighting=endpoint_linear"
  python script/run_traj_tracin_queries_and_scores.py \
    --execute --experiment "$experiment" --train-seed "$seed" \
    --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
    --skip-sampling --skip-query-gradient \
    --artifact-namespace "$query_namespace" --train-artifact "$artifact" \
    --score-output-namespace "$residual_namespace" --num-snapshots 100
  echo "[score] full namespace=$full_namespace weighting=endpoint_linear"
  python script/run_traj_tracin_queries_and_scores.py \
    --execute --experiment "$experiment" --train-seed "$seed" \
    --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
    --skip-sampling --skip-query-gradient \
    --artifact-namespace "$query_namespace" --train-artifact "$artifact" \
    --score-output-namespace "$full_namespace" --num-snapshots 100 \
    --add-optimizer-history
}

combine_pair() {
  local namespace_a="$1" namespace_b="$2" output_namespace="$3"
  # Sum_t(1000-t): original=5005, add-on=5010, total=10015.
  python script/combine_traj_score_namespaces.py \
    --experiment "$experiment" --train-seed "$seed" \
    --query-file "$query_file" --query-ids "$query_ids" \
    --namespace-a "$namespace_a" --namespace-b "$namespace_b" \
    --output-namespace "$output_namespace" \
    --weight-a 0.4997503744383425 --weight-b 0.5002496255616575
}

score_grid "$original_artifact" \
  adamw_residual_original10_endpoint_linear adamw_full_original10_endpoint_linear
score_grid "$addon_artifact" \
  adamw_residual_addon10_endpoint_linear adamw_full_addon10_endpoint_linear

combine_pair \
  adamw_residual_original10_endpoint_linear \
  adamw_residual_addon10_endpoint_linear \
  adamw_residual_aligned20x10_endpoint_linear
combine_pair \
  adamw_full_original10_endpoint_linear \
  adamw_full_addon10_endpoint_linear \
  adamw_full_aligned20x10_endpoint_linear

echo '[lds] combined20 endpoint-linear residual/full'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes \
  "adamw_residual_aligned20x10_endpoint_linear,adamw_full_aligned20x10_endpoint_linear" \
  --prediction-sign=1

echo '[done] ID first10 combined20 with linear endpoint weighting'
