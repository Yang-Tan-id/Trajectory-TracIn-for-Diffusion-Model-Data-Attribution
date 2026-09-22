#!/usr/bin/env bash
#SBATCH -J 3d-ref100-1lrsq
#SBATCH -o 3d-ref100-1lrsq-%j.out
#SBATCH -e 3d-ref100-1lrsq-%j.err
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exp="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
namespace="loss_direction_original_f_reference_trajectory_100t"
out="$shapes/result/$exp/eval/adamw_four_event_original_f_reference_next_100t_single_lr_square/run_${SLURM_JOB_ID}"
mkdir -p "$out"

echo '[query] reuse cached reference-next predicted-noise deltas: 10 queries x 100 timestamps'
echo '[definition] per event: checkpoint_lr * square(q dot (adamw_direction / checkpoint_lr))'
echo '[equivalent] square(q dot adamw_direction) / checkpoint_lr'
echo '[methods] FOUR and FOUR_RESIDUAL; variants=raw,query_l2,train_l2,query_train_l2'

cd "$shapes"
JAX_PLATFORMS=cpu python script/run_adamw_four_event_original_f_scores.py \
  --experiment "$exp" \
  --train-seed "$seed" \
  --query-namespace "$namespace" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
  --checkpoint-weighting uniform \
  --contraction squared \
  --single-checkpoint-lr \
  --out-dir "$out"

echo "[done] reference100 single-LR eventwise-square scores: $out"
