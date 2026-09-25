#!/usr/bin/env bash
#SBATCH -J 3d-q0-20-hvp2
#SBATCH -o 3d-q0-20-hvp2-%j.out
#SBATCH -e 3d-q0-20-hvp2-%j.err
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
export TRAJ_SNAPSHOT_CHUNK_SIZE=1
export TRAJ_TRACIN_QUERY_SAVE_HVP=1
export TRAJ_TRACIN_QUERY_HVP_DIRECTION=next_checkpoint_parameter_delta

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 0 20)"
query_namespace="loss_direction_original_f_reference_trajectory_10t_indist_q0_20_hvp_next_delta"
train_artifact="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient/traj_tracin_adamw_dual_aligned10x10/train_datapoint_gradient_artifact.npz"
residual_namespace="adamw_residual_aligned10x10_second_order_hvp_next_delta_timestamp_sum_squared_q0_20"
full_namespace="adamw_full_aligned10x10_second_order_hvp_next_delta_timestamp_sum_squared_q0_20"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
[[ -f "$train_artifact" ]] || { echo "Missing AdamW train artifact: $train_artifact" >&2; exit 1; }

cd "$shapes"

echo '[1/4] Q0-Q20 exact query gradients and H_q(theta[c+1]-theta[c]); two GPUs'
python script/run_traj_tracin_queries_and_scores.py \
  --execute \
  --experiment "$experiment" \
  --train-seed "$seed" \
  --epochs "${JAX_EPOCHS:-200}" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --gpus 0,1 \
  --skip-sampling \
  --skip-score \
  --artifact-namespace "$query_namespace" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 10 \
  --log-prefix q0_20_hvp_next_delta

# The train artifact stores signed AdamW update-space features, including its
# checkpoint learning rate. Do not multiply by another outer learning rate.
export JAX_PLATFORMS=cpu
export JAX_NUM_DEVICES=1
export TRACIN_SCORE_QUERY_HVP_KEY=query_hvp_features
export TRACIN_SCORE_SECOND_ORDER_COEFFICIENT=0.5
export TRACIN_SCORE_EXPECTED_PROJECTION_SEED="$seed"
export TRACIN_SCORE_TIMESTEP_ALLOWLIST="0,111,222,333,444,555,666,777,888,999"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=uniform_checkpoint
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform
export TRACIN_SCORE_CONTRACTION=timestamp_sum_squared

echo '[2/4] residual AdamW: delta_z dot (g_q + 0.5 H_q Delta_c)'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" \
  --train-artifact "$train_artifact" \
  --score-output-namespace "$residual_namespace" \
  --num-snapshots 10

echo '[3/4] full AdamW: (delta_z + history) dot (g_q + 0.5 H_q Delta_c)'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" \
  --train-artifact "$train_artifact" \
  --score-output-namespace "$full_namespace" \
  --num-snapshots 10 \
  --add-optimizer-history

echo '[4/4] cached LDS: 21 queries x 2 AdamW forms x 4 normalizations x 4 targets'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$residual_namespace,$full_namespace" \
  --prediction-sign=1

echo '[done] Q0-Q20 second-order AdamW HVP, aligned10x10 timestamp-wise-square'
