#!/usr/bin/env bash
#SBATCH -J 3d-id10-a20fix
#SBATCH -o 3d-id10-a20fix-%j.out
#SBATCH -e 3d-id10-a20fix-%j.err
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
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="0,1,2,3,4,5,6,7,8,9"
query_namespace="loss_direction_original_f_reference_trajectory_addon10_indist_first10"
train_artifact="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient/traj_tracin_adamw_dual_aligned10x10_addon10/train_datapoint_gradient_artifact.npz"

original_residual="adamw_residual_aligned10x10"
original_full="adamw_full_aligned10x10"
addon_residual="adamw_residual_aligned10x10_addon10_corrected"
addon_full="adamw_full_aligned10x10_addon10_corrected"
combined_residual="adamw_residual_aligned20x10_combined_corrected"
combined_full="adamw_full_aligned20x10_combined_corrected"

[[ -f "$train_artifact" ]] || { echo "Missing add-on train artifact: $train_artifact" >&2; exit 1; }

cd "$shapes"

echo '[1/5] corrected add-on10 residual score using the matching add-on10 query grid'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" --train-artifact "$train_artifact" \
  --score-output-namespace "$addon_residual" --num-snapshots 10

echo '[2/5] corrected add-on10 full score using residual plus optimizer history'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" --train-artifact "$train_artifact" \
  --score-output-namespace "$addon_full" --num-snapshots 10 \
  --add-optimizer-history

echo '[3/5] uniform combined20 residual = 0.5 original10 + 0.5 corrected add-on10'
python script/combine_traj_score_namespaces.py \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --namespace-a "$original_residual" --namespace-b "$addon_residual" \
  --output-namespace "$combined_residual" --weight-a 0.5 --weight-b 0.5

echo '[4/5] uniform combined20 full = 0.5 original10 + 0.5 corrected add-on10'
python script/combine_traj_score_namespaces.py \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --namespace-a "$original_full" --namespace-b "$addon_full" \
  --output-namespace "$combined_full" --weight-a 0.5 --weight-b 0.5

echo '[5/5] cached LDS for corrected add-on10 and uniform combined20'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes \
  "$addon_residual,$addon_full,$combined_residual,$combined_full" \
  --prediction-sign=1

echo '[done] corrected ID first10 add-on10 and uniform combined20 AdamW scores/LDS'
