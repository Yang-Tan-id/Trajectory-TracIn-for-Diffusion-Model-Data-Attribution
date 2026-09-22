#!/usr/bin/env bash
#SBATCH -J 3d-ref100-score
#SBATCH -o 3d-ref100-score-%j.out
#SBATCH -e 3d-ref100-score-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_adamw_four_event_original_f_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1

experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
query_namespace="loss_direction_original_f_reference_trajectory_100t"
score_namespace="traj_tracin_${query_namespace}"
out_dir="$shapes/result/$experiment/eval/adamw_four_event_reference100_score_artifacts/run_${SLURM_JOB_ID}"

cd "$repo"
echo '[score] reference-next 100t; linear FOUR_RESIDUAL + BOTH-L2; uniform checkpoint weights'
JAX_PLATFORMS=cpu python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
  --experiment "$experiment" \
  --train-seed "$train_seed" \
  --query-namespace "$query_namespace" \
  --attribution-points 5000 \
  --checkpoint-weighting uniform \
  --contraction linear \
  --methods four_residual \
  --save-score-namespace "$score_namespace" \
  --save-score-method four_residual \
  --save-score-variant query_train_l2 \
  --out-dir "$out_dir"

echo '[plot] reference-next 100t endpoint top12; positive-LDS ordering'
python "$shapes/script/plot_endpoint_top6_datapoints.py" \
  --experiment "$experiment" \
  --train-seed "$train_seed" \
  --traj-namespace "$query_namespace" \
  --traj-title 'Reference trajectory 100t · next checkpoint · AdamW FOUR_RESIDUAL · BOTH-L2' \
  --traj-ranking-sign 1 \
  --traj-output-stem reference_trajectory_next_four_residual_both_l2 \
  --only-traj \
  --top-k 12 \
  --output-dir "$out_dir/top12"

cp "$out_dir/top12/reference_trajectory_next_four_residual_both_l2_endpoint_top12.png" \
  "$repo/reference_trajectory_next_100t_four_residual_both_l2_endpoint_top12.png"
echo "[done] $out_dir"
