#!/usr/bin/env bash
#SBATCH -J 3d-top1k-10q
#SBATCH -o 3d-top1k-10q-%j.out
#SBATCH -e 3d-top1k-10q-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_top1000_removal_counterfactual.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export JAX_NUM_DEVICES=1
export JAX_DATA_PARALLEL=0
export JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-16}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"

method="${REMOVAL_METHOD:?Set REMOVAL_METHOD to traj_next, traj_previous, or das_mc4_lambda1}"
case "$method" in
  traj_next|traj_previous|das_mc4_lambda1) ;;
  *) echo "Invalid REMOVAL_METHOD: $method" >&2; exit 1 ;;
esac

query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
cd "$shapes"
nvidia-smi

for query_id in $(seq 0 9); do
  echo "================================================================"
  echo "[query] method=$method Q$query_id (train)"
  python script/run_top1000_removal_counterfactual.py \
    --method "$method" \
    --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "${EXPERIMENT_TAG:-experiment1}" \
    --train-seed "${TRAIN_SEED:-42}" \
    --epochs "${JAX_EPOCHS:-200}" \
    --topk 1000 \
    --stage train

  echo "[query] method=$method Q$query_id (same-condition trajectory evaluation)"
  python script/run_top1000_removal_counterfactual.py \
    --method "$method" \
    --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "${EXPERIMENT_TAG:-experiment1}" \
    --train-seed "${TRAIN_SEED:-42}" \
    --epochs "${JAX_EPOCHS:-200}" \
    --topk 1000 \
    --stage eval
done

echo "[done] method=$method Q0-Q9 top-1000 removal models and counterfactual evaluation"
