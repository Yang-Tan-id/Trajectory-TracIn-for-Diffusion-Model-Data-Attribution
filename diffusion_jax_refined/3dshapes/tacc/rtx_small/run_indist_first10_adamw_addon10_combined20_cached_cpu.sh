#!/usr/bin/env bash
#SBATCH -J 3d-id10-a20cpu
#SBATCH -o 3d-id10-a20cpu-%j.out
#SBATCH -e 3d-id10-a20cpu-%j.err
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
export TRACIN_SCORE_TIMESTEP_ALLOWLIST="49,149,249,349,449,549,649,749,849,949"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=uniform_checkpoint

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="0,1,2,3,4,5,6,7,8,9"
query_namespace="loss_direction_original_f_reference_trajectory_100t_indist_first10"
train_artifact="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient/traj_tracin_adamw_dual_aligned10x10_addon10/train_datapoint_gradient_artifact.npz"

old_residual="adamw_residual_aligned10x10"
old_full="adamw_full_aligned10x10"
addon_residual="adamw_residual_aligned10x10_addon10"
addon_full="adamw_full_aligned10x10_addon10"
combined_residual="adamw_residual_aligned20x10_combined"
combined_full="adamw_full_aligned20x10_combined"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
[[ -f "$train_artifact" ]] || { echo "Missing add-on AdamW artifact: $train_artifact" >&2; exit 1; }

cd "$shapes"
echo '[1/5] add-on10 residual scores: cached reference-100t queries cropped to offset 10t'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" --train-artifact "$train_artifact" \
  --score-output-namespace "$addon_residual" --num-snapshots 100

echo '[2/5] add-on10 full scores: residual update plus stored AdamW optimizer history'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" --train-artifact "$train_artifact" \
  --score-output-namespace "$addon_full" --num-snapshots 100 \
  --add-optimizer-history

echo '[3/5] combine original10 and add-on10 residual LINEAR scores with equal weights'
python script/combine_traj_score_namespaces.py \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --namespace-a "$old_residual" --namespace-b "$addon_residual" \
  --output-namespace "$combined_residual" --weight-a 0.5 --weight-b 0.5

echo '[4/5] combine original10 and add-on10 full LINEAR scores with equal weights'
python script/combine_traj_score_namespaces.py \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --namespace-a "$old_full" --namespace-b "$addon_full" \
  --output-namespace "$combined_full" --weight-a 0.5 --weight-b 0.5

echo '[5/5] cached LDS for add-on10 and combined20 residual/full scores'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$addon_residual,$addon_full,$combined_residual,$combined_full" \
  --prediction-sign=1

echo '[done] ID first10 add-on10 plus uniform combined20 AdamW aligned scores'
