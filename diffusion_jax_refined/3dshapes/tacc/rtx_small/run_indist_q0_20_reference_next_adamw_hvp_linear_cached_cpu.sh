#!/usr/bin/env bash
#SBATCH -J 3d-q0-20-hvpl
#SBATCH -o 3d-q0-20-hvpl-%j.out
#SBATCH -e 3d-q0-20-hvpl-%j.err
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
query_namespace="loss_direction_original_f_reference_trajectory_10t_indist_q0_20_hvp_next_delta"
train_artifact="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient/traj_tracin_adamw_dual_aligned10x10/train_datapoint_gradient_artifact.npz"
residual_namespace="adamw_residual_aligned10x10_second_order_hvp_next_delta_linear_q0_20"
full_namespace="adamw_full_aligned10x10_second_order_hvp_next_delta_linear_q0_20"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
[[ -f "$train_artifact" ]] || { echo "Missing AdamW train artifact: $train_artifact" >&2; exit 1; }

cd "$shapes"

# Reuse the exact reference-next gradient/HVP artifacts produced by 3d-q0-20-hvp2.
# AdamW train features already contain their checkpoint LR, so use uniform weights.
export TRACIN_SCORE_QUERY_HVP_KEY=query_hvp_features
export TRACIN_SCORE_SECOND_ORDER_COEFFICIENT=0.5
export TRACIN_SCORE_EXPECTED_PROJECTION_SEED="$seed"
export TRACIN_SCORE_TIMESTEP_ALLOWLIST="0,111,222,333,444,555,666,777,888,999"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=uniform_checkpoint
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform
export TRACIN_SCORE_CONTRACTION=linear

echo '[1/3] residual AdamW linear contraction: sum_(c,t) delta_z dot (g_q + 0.5 H_q Delta_c)'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" \
  --train-artifact "$train_artifact" \
  --score-output-namespace "$residual_namespace" \
  --num-snapshots 10

echo '[2/3] full AdamW linear contraction: sum_(c,t) (delta_z + history) dot (g_q + 0.5 H_q Delta_c)'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" \
  --train-artifact "$train_artifact" \
  --score-output-namespace "$full_namespace" \
  --num-snapshots 10 \
  --add-optimizer-history

echo '[3/3] cached LDS: 21 queries x 2 AdamW forms x 4 normalizations x 4 targets'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$residual_namespace,$full_namespace" \
  --prediction-sign=1

echo '[done] Q0-Q20 reference-next second-order AdamW HVP, aligned10x10 linear (no square)'
