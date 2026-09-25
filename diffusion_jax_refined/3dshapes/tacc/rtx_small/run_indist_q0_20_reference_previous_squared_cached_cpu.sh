#!/usr/bin/env bash
#SBATCH -J 3d-q0-20-prev2
#SBATCH -o 3d-q0-20-prev2-%j.out
#SBATCH -e 3d-q0-20-prev2-%j.err
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

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
[[ -f "$raw_train" ]] || { echo "Missing raw train artifact: $raw_train" >&2; exit 1; }
[[ -f "$adamw_train" ]] || { echo "Missing AdamW train artifact: $adamw_train" >&2; exit 1; }

cd "$shapes"

export TRACIN_SCORE_TIMESTEP_ALLOWLIST="0,111,222,333,444,555,666,777,888,999"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=previous_target_checkpoint_lr
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform

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

run_family() {
  local contraction="$1"
  local suffix="$2"
  export TRACIN_SCORE_CONTRACTION="$contraction"

  local raw_namespace="raw_mc10_aligned10x10_previous_target_${suffix}_previous_lr_q0_20"
  local residual_namespace="adamw_residual_aligned10x10_previous_target_${suffix}_previous_lr_q0_20"
  local full_namespace="adamw_full_aligned10x10_previous_target_${suffix}_previous_lr_q0_20"

  echo "[score] contraction=$contraction raw/non-AdamW with absolute LR[c-1]"
  score_one "$raw_train" "$raw_namespace" 0 0
  echo "[score] contraction=$contraction AdamW residual with LR[c-1]/LR[c]"
  score_one "$adamw_train" "$residual_namespace" 0 1
  echo "[score] contraction=$contraction AdamW full with LR[c-1]/LR[c]"
  score_one "$adamw_train" "$full_namespace" 1 1

  python script/run_traj_tracin_lds_cached.py \
    --execute --experiment "$experiment" --train-seed "$seed" \
    --query-file "$query_file" --query-ids "$query_ids" \
    --score-schemes "$raw_namespace,$residual_namespace,$full_namespace" \
    --prediction-sign=1
}

echo '[1/2] previous-target termwise-square: sum_(c,t) u[c,t]^2'
run_family squared termwise_squared

echo '[2/2] previous-target timestamp-wise-square: sum_t (sum_c u[c,t])^2'
run_family timestamp_sum_squared timestamp_sum_squared

echo '[done] Q0-Q20 previous-target termwise-square and timestamp-wise-square'
