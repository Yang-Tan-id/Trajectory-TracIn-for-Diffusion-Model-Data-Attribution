#!/usr/bin/env bash
#SBATCH -J 3d-rfown100
#SBATCH -o 3d-rfown100-%j.out
#SBATCH -e 3d-rfown100-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/combine_reference_own100_scores_gpu.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export JAX_NUM_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
out_dir="$shapes/result/$experiment/eval/reference_own100_combined/run_${SLURM_JOB_ID}"
mkdir -p "$out_dir"

echo "[query] cached reference + own raw next-checkpoint directions; 100 timestamps"
echo "[reductions] linear_sum, square_sum, absolute_sum"
echo "[weighting] uniform checkpoints; fixed p1"

python "$shapes/script/combine_reference_own100_scores_gpu.py" \
  --experiment "$experiment" \
  --train-seed "$train_seed" \
  --device gpu \
  --out-dir "$out_dir" \
  2>&1 | tee "$out_dir/run.log"

echo "[done] reference+own100 combined scores: $out_dir"
