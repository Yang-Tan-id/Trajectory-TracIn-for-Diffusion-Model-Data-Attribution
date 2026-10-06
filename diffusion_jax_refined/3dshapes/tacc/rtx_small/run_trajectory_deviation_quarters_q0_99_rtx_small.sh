#!/usr/bin/env bash
#SBATCH -J 3d-traj-q4
#SBATCH -o 3d-traj-q4-%j.out
#SBATCH -e 3d-traj-q4-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_lds_true_f.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LDS_SAVE_PER_SNAPSHOT=1
export LDS_REQUIRE_PER_SNAPSHOT=1

query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"

cd "$shapes"
"$python_bin" script/run_lds_true_f.py \
  --execute \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --subset-seeds "${LDS_SUBSET_SEEDS:-0,1,2}" \
  --target-functions trajectory_state_mse \
  --gpus "${LDS_TRUE_F_GPUS:-0,1}" \
  --python-bin "$python_bin"

"$python_bin" script/summarize_trajectory_deviation_quarters.py \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --expected-models 192

echo "[done] Q0-Q99 x 192 LDS models trajectory-deviation quarter statistics"
