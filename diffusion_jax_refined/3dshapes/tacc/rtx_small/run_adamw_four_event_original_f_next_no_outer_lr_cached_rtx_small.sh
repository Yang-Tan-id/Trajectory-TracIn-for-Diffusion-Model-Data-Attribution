#!/usr/bin/env bash
#SBATCH -J 3d-adam4-nolr
#SBATCH -o 3d-adam4-nolr-%j.out
#SBATCH -e 3d-adam4-nolr-%j.err
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
out_dir="$shapes/result/$experiment/eval/adamw_four_event_original_f_next_no_outer_lr/run_${SLURM_JOB_ID}"
mkdir -p "$out_dir"

cd "$repo"
JAX_PLATFORMS=cpu python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
  --experiment "$experiment" --train-seed "$train_seed" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
  --checkpoint-weighting uniform --out-dir "$out_dir"
echo "[done] AdamW original-f next without outer checkpoint LR: $out_dir"
