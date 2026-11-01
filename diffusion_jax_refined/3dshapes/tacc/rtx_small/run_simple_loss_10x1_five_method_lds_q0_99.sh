#!/usr/bin/env bash
#SBATCH -J 3d-sl10-lds
#SBATCH -o 3d-sl10-lds-%j.out
#SBATCH -e 3d-sl10-lds-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

source_repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python_bin="${PYTHON_BIN:-python}"
cache_namespace="simple_loss_10x1_t0_999_uniform_split_seed0"

# Execute imports from node-local storage while keeping the persistent result
# and dataset trees on /work.  This avoids Stampede3 /work metadata failures.
runtime_root="${SLURM_TMPDIR:-/tmp/${USER}/shape3d_sl10_${SLURM_JOB_ID:-$$}}"
repo="$runtime_root/repo"
mkdir -p "$repo"
tar -C "$source_repo" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='diffusion_jax_refined/3dshapes/result' \
  -cf - \
  diffusion_jax_refined/common \
  diffusion_jax_refined/legacy_jax \
  diffusion_jax_refined/script \
  diffusion_jax_refined/3dshapes | tar -C "$repo" -xf -
ln -s "$source_repo/diffusion_jax_refined/3dshapes/result" \
  "$repo/diffusion_jax_refined/3dshapes/result"
ln -s "$source_repo/diffusion_jax_refined/dataset" \
  "$repo/diffusion_jax_refined/dataset"

shapes="$repo/diffusion_jax_refined/3dshapes"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
cd "$repo"
timesteps="$($python_bin -c 'from diffusion_jax_refined.common.predicted_noise_kl import uniform_timesteps; print(",".join(map(str, uniform_timesteps(1000,10).tolist())))')"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SIMPLE_LOSS_EVAL_CHUNK_SIZE="${SIMPLE_LOSS_EVAL_CHUNK_SIZE:-10}"
export LDS_TRUE_F_LOG_ROOT="${LDS_TRUE_F_LOG_ROOT:-$(dirname "$0")/worker_logs_${SLURM_JOB_ID:-$$}}"
mkdir -p "$LDS_TRUE_F_LOG_ROOT"

echo "[definition] Shape3D simple-loss LDS target: 10 uniform timesteps over the full valid range t=0..999 x MC1"
echo "[timesteps] $timesteps"
echo "[definition] noise keys: jax.random.split(PRNGKey(0), 10), shared by every query/model"
echo "[runtime] node-local code=$repo; persistent results=$source_repo/diffusion_jax_refined/3dshapes/result"
echo "[logs] worker logs=$LDS_TRUE_F_LOG_ROOT"
echo "[cache] $cache_namespace"
echo "[order] checkpoint-major: one restore followed by all 100 queries"

cd "$shapes"
"$python_bin" script/run_lds_true_f.py \
  --execute \
  --experiment experiment1 \
  --train-seed 42 \
  --query-file "$query_file" \
  --query-ids "$(seq -s, 0 99)" \
  --subset-seeds 0,1,2 \
  --target-functions simple_loss \
  --gpus 0,1 \
  --simple-loss-num-mc 10 \
  --simple-loss-mc-seed 0 \
  --simple-loss-timesteps "$timesteps" \
  --uniform-split-simple-loss-probes \
  --model-major-simple-loss \
  --target-cache-namespace "$cache_namespace" \
  --python-bin "$python_bin"

export JAX_PLATFORMS=cpu
"$python_bin" script/evaluate_five_method_simple_loss_100x1.py \
  --experiment experiment1 \
  --train-seed 42 \
  --query-file "$query_file" \
  --query-ids 0-99 \
  --cache-namespace "$cache_namespace" \
  --expected-models 192 \
  --expected-probes 10 \
  --target-name simple_loss_10x1 \
  --shape3d-pint-banks both \
  --transforms both

cd "$repo"
"$python_bin" diffusion_jax_refined/script/export_pint_lds_txt.py \
  --dataset shape3d \
  --repo-root "$repo" \
  --query-ids 0-99 \
  --expected-subsets 192 \
  --targets endpoint_contarfactual,traj_contarfactual,simple_loss_10x1,noise_trajectory \
  --shape3d-pint-banks both \
  --transforms both \
  --output "$source_repo/shape3d_twelve_config_lds_all_targets_simple_loss_10x1_t0_999_q0_99.txt"

echo "[done] $source_repo/shape3d_twelve_config_lds_all_targets_simple_loss_10x1_t0_999_q0_99.txt"
