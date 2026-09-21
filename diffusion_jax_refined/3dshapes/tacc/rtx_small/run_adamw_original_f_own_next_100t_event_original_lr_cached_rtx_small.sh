#!/usr/bin/env bash
#SBATCH -J 3d-own100-evlr
#SBATCH -o 3d-own100-evlr-%j.out
#SBATCH -e 3d-own100-evlr-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 16:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_adamw_four_event_original_f_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exp="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
out="$shapes/result/$exp/eval/adamw_four_event_original_f_own_next_100t_event_original_lr/run_${SLURM_JOB_ID}"
mkdir -p "$out/linear" "$out/squared"

echo '[query] reuse checkpoint-own true next-checkpoint predicted-noise delta; 100 timestamps'
echo '[train] fixed checkpoint theta/m/v; divide E1-E4 by checkpoint LR'
echo '[weight] transform each event separately, then weight once by its original training batch LR'

cd "$shapes"
for contraction in linear squared; do
  echo "[score] contraction=$contraction"
  JAX_PLATFORMS=cpu python script/run_adamw_four_event_original_f_scores.py \
    --experiment "$exp" \
    --train-seed "$seed" \
    --query-namespace loss_direction_original_f_checkpoint_own_trajectory_100t \
    --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
    --checkpoint-weighting uniform \
    --contraction "$contraction" \
    --event-original-lr \
    --out-dir "$out/$contraction"
done

echo "[done] own100 event-original-LR linear and eventwise-square scores: $out"
