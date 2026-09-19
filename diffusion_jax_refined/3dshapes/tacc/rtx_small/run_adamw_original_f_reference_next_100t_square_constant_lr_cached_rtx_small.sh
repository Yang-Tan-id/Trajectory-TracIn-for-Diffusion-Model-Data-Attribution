#!/usr/bin/env bash
#SBATCH -J 3d-adam-r100sq
#SBATCH -o 3d-adam-r100sq-%j.out
#SBATCH -e 3d-adam-r100sq-%j.err
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
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
namespace="loss_direction_original_f_reference_trajectory_100t"
out_dir="$shapes/result/$EXPERIMENT_TAG/eval/adamw_four_event_original_f_reference_next_100t_square_constant_lr/run_${SLURM_JOB_ID}"
mkdir -p "$out_dir"

echo "[cached score] square every checkpoint/timestamp linear term, then sum"
echo "[query] cached reference trajectory, 10 queries x 100 timestamps"
JAX_PLATFORMS=cpu python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --query-namespace "$namespace" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
  --checkpoint-weighting uniform \
  --contraction squared \
  --out-dir "$out_dir"

echo "[done] reference100 squared constant-LR scores: $out_dir"
