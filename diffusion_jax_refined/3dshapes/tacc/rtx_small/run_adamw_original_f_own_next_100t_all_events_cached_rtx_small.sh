#!/usr/bin/env bash
#SBATCH -J 3d-own100-e14
#SBATCH -o 3d-own100-e14-%j.out
#SBATCH -e 3d-own100-e14-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00

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
namespace="loss_direction_original_f_checkpoint_own_trajectory_100t"
out_root="$shapes/result/$experiment/eval/adamw_four_event_original_f_own_next_100t_all_events/run_${SLURM_JOB_ID}"
mkdir -p "$out_root/linear" "$out_root/squared"

echo "[cached query] own trajectory, raw next-checkpoint direction, 100 timestamps"
echo "[methods] E1 E2 E3 E4 FOUR and history-subtracted counterparts"
echo "[weighting] uniform checkpoints; fixed p1"

for contraction in linear squared; do
  JAX_PLATFORMS=cpu python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
    --experiment "$experiment" --train-seed "$train_seed" \
    --query-namespace "$namespace" \
    --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
    --checkpoint-weighting uniform \
    --contraction "$contraction" \
    --include-all-events \
    --out-dir "$out_root/$contraction"
done

echo "[done] cached own100 E1-E4/FOUR linear and squared LDS: $out_root"
