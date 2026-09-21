#!/usr/bin/env bash
#SBATCH -J 3d-own100-score
#SBATCH -o 3d-own100-score-%j.out
#SBATCH -e 3d-own100-score-%j.err
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
if [[ ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_adamw_four_event_original_f_scores.py" ]]; then
  echo "repository root not found" >&2
  exit 1
fi

shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1

experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
query_namespace="loss_direction_original_f_checkpoint_own_trajectory_100t"
score_namespace="traj_tracin_${query_namespace}"
out_dir="$shapes/result/$experiment/eval/adamw_four_event_own100_score_artifacts/run_${SLURM_JOB_ID}"
mkdir -p "$out_dir"

echo "[query] checkpoint-own next-checkpoint predicted-noise delta; 100 timestamps"
echo "[score] FOUR_RESIDUAL + QUERY_TRAIN_L2; linear p1; uniform outer checkpoint weighting"
echo "[artifact namespace] $score_namespace"

cd "$repo"
JAX_PLATFORMS=cpu python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
  --experiment "$experiment" \
  --train-seed "$train_seed" \
  --query-namespace "$query_namespace" \
  --attribution-points 5000 \
  --checkpoint-weighting uniform \
  --contraction linear \
  --save-score-namespace "$score_namespace" \
  --save-score-method four_residual \
  --save-score-variant query_train_l2 \
  --out-dir "$out_dir"

echo "[done] own100 FOUR_RESIDUAL score vectors and LDS: $out_dir"
