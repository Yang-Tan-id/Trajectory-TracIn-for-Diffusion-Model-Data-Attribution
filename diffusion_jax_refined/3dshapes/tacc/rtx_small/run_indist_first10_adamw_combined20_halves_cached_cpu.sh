#!/usr/bin/env bash
#SBATCH -J 3d-id10-a20half
#SBATCH -o 3d-id10-a20half-%j.out
#SBATCH -e 3d-id10-a20half-%j.err
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

score_subset() {
  local artifact="$1" allowlist="$2" residual_namespace="$3" full_namespace="$4"
  export TRACIN_SCORE_TIMESTEP_ALLOWLIST="$allowlist"
  echo "[score] residual namespace=$residual_namespace t=$allowlist"
  python script/run_traj_tracin_queries_and_scores.py \
    --execute --experiment "$experiment" --train-seed "$seed" \
    --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
    --skip-sampling --skip-query-gradient \
    --artifact-namespace "$query_namespace" --train-artifact "$artifact" \
    --score-output-namespace "$residual_namespace" --num-snapshots 100
  echo "[score] full namespace=$full_namespace t=$allowlist"
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
  python script/combine_traj_score_namespaces.py \
    --experiment "$experiment" --train-seed "$seed" \
    --query-file "$query_file" --query-ids "$query_ids" \
    --namespace-a "$namespace_a" --namespace-b "$namespace_b" \
    --output-namespace "$output_namespace" --weight-a 0.5 --weight-b 0.5
}

# Generation-trajectory first half: high diffusion timesteps.  Each source
# score is a uniform mean over five terms; their equal mixture is a uniform
# mean over all ten early terms.
score_subset "$original_artifact" "555,666,777,888,999" \
  adamw_residual_original5_early_high_t adamw_full_original5_early_high_t
score_subset "$addon_artifact" "549,649,749,849,949" \
  adamw_residual_addon5_early_high_t adamw_full_addon5_early_high_t
combine_pair adamw_residual_original5_early_high_t adamw_residual_addon5_early_high_t \
  adamw_residual_aligned10x10_early_high_t
combine_pair adamw_full_original5_early_high_t adamw_full_addon5_early_high_t \
  adamw_full_aligned10x10_early_high_t

# Generation-trajectory second half: low diffusion timesteps.
score_subset "$original_artifact" "0,111,222,333,444" \
  adamw_residual_original5_late_low_t adamw_full_original5_late_low_t
score_subset "$addon_artifact" "49,149,249,349,449" \
  adamw_residual_addon5_late_low_t adamw_full_addon5_late_low_t
combine_pair adamw_residual_original5_late_low_t adamw_residual_addon5_late_low_t \
  adamw_residual_aligned10x10_late_low_t
combine_pair adamw_full_original5_late_low_t adamw_full_addon5_late_low_t \
  adamw_full_aligned10x10_late_low_t

echo '[lds] early/high-t and late/low-t residual/full'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes \
  "adamw_residual_aligned10x10_early_high_t,adamw_full_aligned10x10_early_high_t,adamw_residual_aligned10x10_late_low_t,adamw_full_aligned10x10_late_low_t" \
  --prediction-sign=1

echo '[done] ID first10 combined20 split into early/high-t and late/low-t halves'
