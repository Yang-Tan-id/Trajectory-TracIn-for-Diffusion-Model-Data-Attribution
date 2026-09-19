#!/usr/bin/env bash
#SBATCH -J 3d-adam-own1k
#SBATCH -o 3d-adam-own1k-%j.out
#SBATCH -e 3d-adam-own1k-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; do repo="$(dirname "$repo")"; done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}" TRAIN_SEED="${TRAIN_SEED:-42}" JAX_EPOCHS="${JAX_EPOCHS:-200}"
export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
namespace="loss_direction_original_f_checkpoint_own_trajectory_1000t"
run_root="$shapes/result/$EXPERIMENT_TAG/eval/adamw_four_event_original_f_own_next_1000t_constant_lr/run_${SLURM_JOB_ID}"
mkdir -p "$run_root/linear" "$run_root/squared"
cd "$shapes"

echo "[phase 1/2] checkpoint-own raw next-checkpoint delta: 10 queries x all 1000 timestamps"
python script/run_traj_tracin_queries_and_scores.py --execute \
  --experiment "$EXPERIMENT_TAG" --train-seed "$TRAIN_SEED" --epochs "$JAX_EPOCHS" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 --gpus 0,1 --skip-sampling --skip-score \
  --artifact-namespace "$namespace" --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 1000 --log-prefix adam_own1000
unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY

echo "[phase 2/2] linear and squared scores"
for contraction in linear squared; do
  JAX_PLATFORMS=cpu python script/run_adamw_four_event_original_f_scores.py \
    --experiment "$EXPERIMENT_TAG" --train-seed "$TRAIN_SEED" \
    --query-namespace "$namespace" --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
    --checkpoint-weighting uniform --contraction "$contraction" --out-dir "$run_root/$contraction"
done
echo "[done] own full-1000 raw-next scores: $run_root"
