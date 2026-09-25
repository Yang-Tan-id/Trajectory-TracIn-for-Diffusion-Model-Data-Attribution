#!/usr/bin/env bash
#SBATCH -J 3d-q0-20-prev
#SBATCH -o 3d-q0-20-prev-%j.out
#SBATCH -e 3d-q0-20-prev-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export TRAJ_SNAPSHOT_CHUNK_SIZE=8

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 0 20)"
query_namespace="loss_direction_original_f_reference_trajectory_previous_checkpoint_10t_indist_q0_20"
train_root="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient"
raw_train="$train_root/traj_tracin/train_datapoint_gradient_artifact.npz"
adamw_train="$train_root/traj_tracin_adamw_dual_aligned10x10/train_datapoint_gradient_artifact.npz"
raw_namespace="raw_mc10_aligned10x10_previous_target_linear_previous_lr_q0_20"
residual_namespace="adamw_residual_aligned10x10_previous_target_linear_previous_lr_q0_20"
full_namespace="adamw_full_aligned10x10_previous_target_linear_previous_lr_q0_20"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
[[ -f "$raw_train" ]] || { echo "Missing raw train artifact: $raw_train" >&2; exit 1; }
[[ -f "$adamw_train" ]] || { echo "Missing AdamW train artifact: $adamw_train" >&2; exit 1; }

cd "$shapes"

echo '[1/5] Q0-Q20 current-vs-previous checkpoint predicted-noise MSE gradients (49x10 terms)'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --epochs "${JAX_EPOCHS:-200}" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --gpus 0,1 --skip-sampling --skip-score \
  --artifact-namespace "$query_namespace" \
  --query-objective trajectory_previous_checkpoint_noise_mse \
  --num-snapshots 10 \
  --log-prefix q0_20_reference_previous

export JAX_PLATFORMS=cpu
export JAX_NUM_DEVICES=1
export TRACIN_SCORE_TIMESTEP_ALLOWLIST="0,111,222,333,444,555,666,777,888,999"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=previous_target_checkpoint_lr
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform
export TRACIN_SCORE_CONTRACTION=linear

echo '[2/5] raw/non-AdamW: multiply raw loss gradients by absolute LR[c-1]'
export TRACIN_SCORE_FEATURES_INCLUDE_LR=0
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" --train-artifact "$raw_train" \
  --score-output-namespace "$raw_namespace" --num-snapshots 10

echo '[3/5] AdamW residual: replace embedded LR[c] by LR[c-1] via their ratio'
export TRACIN_SCORE_FEATURES_INCLUDE_LR=1
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" --train-artifact "$adamw_train" \
  --score-output-namespace "$residual_namespace" --num-snapshots 10

echo '[4/5] AdamW full: residual plus optimizer history, with LR[c-1]/LR[c] replacement'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" --train-artifact "$adamw_train" \
  --score-output-namespace "$full_namespace" --num-snapshots 10 \
  --add-optimizer-history

echo '[5/5] cached LDS: raw + AdamW residual/full, four normalizations, four targets'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$raw_namespace,$residual_namespace,$full_namespace" \
  --prediction-sign=1

echo '[done] Q0-Q20 previous-checkpoint target: raw + AdamW residual/full linear scores'
