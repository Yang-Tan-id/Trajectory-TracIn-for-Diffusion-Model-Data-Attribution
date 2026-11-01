#!/usr/bin/env bash
#SBATCH -J 3d-sl100-lds
#SBATCH -o 3d-sl100-lds-%j.out
#SBATCH -e 3d-sl100-lds-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_lds_true_f.py" ]]; do
  repo="$(dirname "$repo")"
done
[[ -f "$repo/diffusion_jax_refined/3dshapes/script/run_lds_true_f.py" ]] || {
  echo "Could not locate repository" >&2
  exit 1
}

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

shapes="$repo/diffusion_jax_refined/3dshapes"
python_bin="${PYTHON_BIN:-python}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 0 99)"
cache_namespace="simple_loss_100x1_shared_seed0"

# 100 evenly spaced DDIM/DDPM indices from 999 down to 0.  The paired seeds
# give one independent probe per timestamp, shared across every subset model.
timesteps="$($python_bin -c 'import numpy as np; print(",".join(map(str, np.linspace(999, 0, 100, dtype=np.int32))))')"
noise_seeds="$($python_bin -c 'print(",".join(map(str, range(100000, 100100))))')"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[definition] Shape3D simple-loss LDS target: 100 uniformly spaced timestamps x MC1"
echo "[definition] probes are paired and shared across all 192 subset models"
echo "[scope] Q0-Q99; only five listed methods x raw-gradient/AdamW are scored"
echo "[cache] $cache_namespace"

cd "$shapes"
"$python_bin" script/run_lds_true_f.py \
  --execute \
  --experiment experiment1 \
  --train-seed 42 \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --subset-seeds 0,1,2 \
  --target-functions simple_loss \
  --gpus 0,1 \
  --simple-loss-num-mc 100 \
  --simple-loss-mc-seed 0 \
  --simple-loss-timesteps "$timesteps" \
  --simple-loss-noise-seeds "$noise_seeds" \
  --pair-simple-loss-probes \
  --target-cache-namespace "$cache_namespace" \
  --python-bin "$python_bin"

export JAX_PLATFORMS=cpu
"$python_bin" script/evaluate_five_method_simple_loss_100x1.py \
  --experiment experiment1 \
  --train-seed 42 \
  --query-file "$query_file" \
  --query-ids 0-99 \
  --cache-namespace "$cache_namespace" \
  --expected-models 192

cd "$repo"
"$python_bin" diffusion_jax_refined/script/export_pint_lds_txt.py \
  --dataset shape3d \
  --repo-root "$repo" \
  --query-ids 0-99 \
  --expected-subsets 192 \
  --targets endpoint_contarfactual,traj_contarfactual,simple_loss,simple_loss_100x1,noise_trajectory \
  --output "$repo/shape3d_five_method_lds_q0_99_all.txt"

echo "[done] $repo/shape3d_five_method_lds_q0_99_all.txt"
