#!/usr/bin/env bash
#SBATCH -J 3d-q0-20-prevraw
#SBATCH -o 3d-q0-20-prevraw-%j.out
#SBATCH -e 3d-q0-20-prevraw-%j.err
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
raw_train="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient/traj_tracin/train_datapoint_gradient_artifact.npz"
score_namespace="raw_mc10_previous_target_last_checkpoint_squared_unit_weight_q0_20"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
[[ -f "$raw_train" ]] || { echo "Missing raw train artifact: $raw_train" >&2; exit 1; }

cd "$shapes"

# Select only the final checkpoint c=49 (whose previous target is c=48).
# Raw features contain no optimizer LR. Give this checkpoint total weight 1;
# uniform timestep weighting distributes it equally over its ten timestamps.
export TRACIN_SCORE_CHECKPOINT_ALLOWLIST=49
export TRACIN_SCORE_TIMESTEP_ALLOWLIST="0,111,222,333,444,555,666,777,888,999"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=uniform_checkpoint
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform
export TRACIN_SCORE_CONTRACTION=timestamp_sum_squared
export TRACIN_SCORE_FEATURES_INCLUDE_LR=0

echo '[1/2] raw last-checkpoint square with checkpoint weight=1 (no learning rate)'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" --train-artifact "$raw_train" \
  --score-output-namespace "$score_namespace" --num-snapshots 10

echo '[2/2] cached LDS: 21 queries x 1 raw method x 4 normalizations x 4 targets'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$score_namespace" \
  --prediction-sign=1

echo '[done] Q0-Q20 previous-target final checkpoint raw square, unit checkpoint weight'
