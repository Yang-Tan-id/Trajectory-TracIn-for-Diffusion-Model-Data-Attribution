#!/usr/bin/env bash
#SBATCH -J 3d-top1k-rm
#SBATCH -o 3d-top1k-rm-%A_%a.out
#SBATCH -e 3d-top1k-rm-%A_%a.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH --array=0-29%2
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

task="${SLURM_ARRAY_TASK_ID}"
query_id="$((task % 10))"
method_id="$((task / 10))"
case "$method_id" in
  0) method=traj_next ;;
  1) method=traj_previous ;;
  2) method=das_mc4_lambda1 ;;
  *) echo "Invalid method id: $method_id" >&2; exit 1 ;;
esac

query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
echo "[task] array=$task method=$method query=Q$query_id"
nvidia-smi

cd "$shapes"
python script/run_top1000_removal_counterfactual.py \
  --method "$method" \
  --query-id "$query_id" \
  --query-file "$query_file" \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --epochs "${JAX_EPOCHS:-200}" \
  --topk 1000 \
  --stage train

# Use a fresh process after training so JAX releases the training executable and
# evaluates the removal checkpoint against the original full-model trajectory.
python script/run_top1000_removal_counterfactual.py \
  --method "$method" \
  --query-id "$query_id" \
  --query-file "$query_file" \
  --experiment "${EXPERIMENT_TAG:-experiment1}" \
  --train-seed "${TRAIN_SEED:-42}" \
  --epochs "${JAX_EPOCHS:-200}" \
  --topk 1000 \
  --stage eval
