#!/usr/bin/env bash
#SBATCH -J 3d-q0-20-prevlast
#SBATCH -o 3d-q0-20-prevlast-%j.out
#SBATCH -e 3d-q0-20-prevlast-%j.err
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH -t 02:00:00
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
export JAX_PLATFORMS=cpu
export JAX_NUM_DEVICES=1

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 0 20)"
query_namespace="loss_direction_original_f_reference_trajectory_previous_checkpoint_10t_indist_q0_20"
train_root="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient"
raw_train="$train_root/traj_tracin/train_datapoint_gradient_artifact.npz"
adamw_train="$train_root/traj_tracin_adamw_dual_aligned10x10/train_datapoint_gradient_artifact.npz"
raw_namespace="raw_mc10_previous_target_last_checkpoint_squared_previous_lr_q0_20"
residual_namespace="adamw_residual_previous_target_last_checkpoint_squared_previous_lr_q0_20"
full_namespace="adamw_full_previous_target_last_checkpoint_squared_previous_lr_q0_20"

cd "$shapes"

export TRACIN_SCORE_CHECKPOINT_ALLOWLIST=49
export TRACIN_SCORE_TIMESTEP_ALLOWLIST="0,111,222,333,444,555,666,777,888,999"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=previous_target_checkpoint_lr
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform
export TRACIN_SCORE_CONTRACTION=timestamp_sum_squared

score_one() {
  local train_artifact="$1"
  local namespace="$2"
  local add_history="$3"
  local features_include_lr="$4"
  export TRACIN_SCORE_FEATURES_INCLUDE_LR="$features_include_lr"
  local history_args=()
  if [[ "$add_history" == 1 ]]; then
    history_args+=(--add-optimizer-history)
  fi
  python script/run_traj_tracin_queries_and_scores.py \
    --execute --experiment "$experiment" --train-seed "$seed" \
    --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
    --skip-sampling --skip-query-gradient \
    --artifact-namespace "$query_namespace" --train-artifact "$train_artifact" \
    --score-output-namespace "$namespace" --num-snapshots 10 \
    "${history_args[@]}"
}

echo '[1/4] raw last-checkpoint square; raw gradient multiplied by absolute LR[48]'
score_one "$raw_train" "$raw_namespace" 0 0

echo '[2/4] AdamW residual last-checkpoint square; embedded LR[49] replaced by LR[48]'
score_one "$adamw_train" "$residual_namespace" 0 1

echo '[3/4] AdamW full last-checkpoint square; embedded LR[49] replaced by LR[48]'
score_one "$adamw_train" "$full_namespace" 1 1

echo '[4/4] cached LDS: 21 queries x 3 methods x 4 normalizations x 4 targets'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$raw_namespace,$residual_namespace,$full_namespace" \
  --prediction-sign=1

echo '[done] Q0-Q20 previous-target final checkpoint (c=49,target=c=48) squared scores'
