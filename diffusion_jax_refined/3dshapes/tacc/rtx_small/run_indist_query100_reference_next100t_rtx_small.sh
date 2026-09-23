#!/usr/bin/env bash
#SBATCH -J 3d-id100-refn
#SBATCH -o 3d-id100-refn-%j.out
#SBATCH -e 3d-id100-refn-%j.err
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
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
unset TRAJ_QUERY_USE_CHECKPOINT_OWN_TRAJECTORY

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 0 119)"
query_namespace="loss_direction_original_f_reference_trajectory_100t_indist100q"
out_root="$shapes/result/$experiment/eval/adamw_reference_next100t_indist100q/run_${SLURM_JOB_ID}"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
mkdir -p "$out_root/linear" "$out_root/squared"

echo '[1/3] fixed-reference next-checkpoint loss gradients: 120 queries x 100 timestamps'
python "$shapes/script/run_traj_tracin_queries_and_scores.py" \
  --execute --experiment "$experiment" --train-seed "$seed" --epochs "$JAX_EPOCHS" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0,1 \
  --skip-sampling --skip-score \
  --artifact-namespace "$query_namespace" \
  --query-objective trajectory_next_checkpoint_noise_mse \
  --num-snapshots 100 --log-prefix indist_refnext100

echo '[2/3] signed linear FOUR/E1/full/residual scores and per-query LDS'
JAX_PLATFORMS=cpu python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --query-namespace "$query_namespace" --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
  --checkpoint-weighting uniform --contraction linear --out-dir "$out_root/linear"

echo '[3/3] termwise-squared FOUR/E1/full/residual scores and per-query LDS'
JAX_PLATFORMS=cpu python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --query-namespace "$query_namespace" --attribution-points "${ATTRIBUTION_POINTS:-5000}" \
  --checkpoint-weighting uniform --contraction squared --out-dir "$out_root/squared"

echo "[done] in-distribution reference-next Traj TracIn: $out_root"
