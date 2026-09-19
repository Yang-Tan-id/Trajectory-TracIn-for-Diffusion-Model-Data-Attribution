#!/usr/bin/env bash
#SBATCH -J 3d-adam-next1k
#SBATCH -o 3d-adam-next1k-%j.out
#SBATCH -e 3d-adam-next1k-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"

reference_namespace="loss_direction_original_f_reference_trajectory_1000t"
own_namespace="loss_direction_original_f_checkpoint_own_trajectory_1000t"
run_root="$shapes/result/$EXPERIMENT_TAG/eval/adamw_four_event_original_f_reference_own_next_1000t_constant_lr/run_${SLURM_JOB_ID}"
reference_out="$run_root/reference"
own_out="$run_root/own"
reference_square_out="$run_root/reference_square"
own_square_out="$run_root/own_square"
mkdir -p "$reference_out" "$own_out" "$reference_square_out" "$own_square_out"

cd "$shapes"
echo "[phase 1/4] fixed reference trajectory: raw next-checkpoint delta, all 1000 timestamps"
unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY
python script/run_traj_tracin_queries_and_scores.py \
  --execute \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --epochs "$JAX_EPOCHS" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --gpus 0,1 \
  --skip-sampling \
  --skip-score \
  --artifact-namespace "$reference_namespace" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 1000 \
  --log-prefix adam_ref1000

echo "[phase 2/4] checkpoint-own trajectory: raw next-checkpoint delta, all 1000 timestamps"
export TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY=1
python script/run_traj_tracin_queries_and_scores.py \
  --execute \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --epochs "$JAX_EPOCHS" \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --gpus 0,1 \
  --skip-sampling \
  --skip-score \
  --artifact-namespace "$own_namespace" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 1000 \
  --log-prefix adam_own1000
unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY

echo "[phase 3/4] linear and squared original-F scores with uniform checkpoint weights"
JAX_PLATFORMS=cpu python script/run_adamw_four_event_original_f_scores.py \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --query-namespace "$reference_namespace" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
  --checkpoint-weighting uniform \
  --contraction linear \
  --out-dir "$reference_out"

JAX_PLATFORMS=cpu python script/run_adamw_four_event_original_f_scores.py \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --query-namespace "$own_namespace" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
  --checkpoint-weighting uniform \
  --contraction linear \
  --out-dir "$own_out"

JAX_PLATFORMS=cpu python script/run_adamw_four_event_original_f_scores.py \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --query-namespace "$reference_namespace" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
  --checkpoint-weighting uniform \
  --contraction squared \
  --out-dir "$reference_square_out"

JAX_PLATFORMS=cpu python script/run_adamw_four_event_original_f_scores.py \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --query-namespace "$own_namespace" \
  --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
  --checkpoint-weighting uniform \
  --contraction squared \
  --out-dir "$own_square_out"

echo "[phase 4/4] print reference and own side by side"
python script/print_adamw_original_f_reference_vs_own.py \
  --reference "$reference_out" \
  --own "$own_out" \
  --compact-raw-residual

python script/print_adamw_original_f_reference_vs_own.py \
  --reference "$reference_square_out" \
  --own "$own_square_out" \
  --compact-raw-residual

echo "[done] full 1000-timestamp reference/own raw-next scores: $run_root"
