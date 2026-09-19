#!/usr/bin/env bash
#SBATCH -J 3d-ofnext-4
#SBATCH -o 3d-ofnext-4-%j.out
#SBATCH -e 3d-ofnext-4-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00

set -euo pipefail
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_original_f_next_adamw_raw_four_event_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
out_dir="$shapes/result/$experiment/eval/original_f_next_adamw_raw_four_event/run_${SLURM_JOB_ID}"
mkdir -p "$out_dir"

python "$shapes/script/run_original_f_next_adamw_raw_four_event_scores.py" \
  --experiment "$experiment" --train-seed "$train_seed" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}" --out-dir "$out_dir"
echo "[done] original-f next AdamW/raw four-event scores: $out_dir"
