#!/usr/bin/env bash
#SBATCH -J 3d-sl50-lds
#SBATCH -o 3d-sl50-lds-%j.out
#SBATCH -e 3d-sl50-lds-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
shapes="$repo/diffusion_jax_refined/3dshapes"
python_bin="${PYTHON_BIN:-python}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
cache_namespace="simple_loss_50x1_t0_500_uniform_split_seed0"
timesteps="$($python_bin -c 'from diffusion_jax_refined.common.predicted_noise_kl import uniform_timesteps; print(",".join(map(str, uniform_timesteps(501,50).tolist())))')"

export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SIMPLE_LOSS_EVAL_CHUNK_SIZE="${SIMPLE_LOSS_EVAL_CHUNK_SIZE:-50}"

echo "[definition] Shape3D simple-loss LDS target: 50 uniform timesteps over inclusive range t=0..500 x MC1"
echo "[definition] noise keys: jax.random.split(PRNGKey(0), 50), shared by every query/model"
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
  --simple-loss-num-mc 50 \
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
  --expected-probes 50 \
  --target-name simple_loss_50x1 \
  --shape3d-pint-banks both \
  --transforms adamw

cd "$repo"
"$python_bin" diffusion_jax_refined/script/export_pint_lds_txt.py \
  --dataset shape3d \
  --repo-root "$repo" \
  --query-ids 0-99 \
  --expected-subsets 192 \
  --targets endpoint_contarfactual,traj_contarfactual,simple_loss_50x1,noise_trajectory \
  --shape3d-pint-banks both \
  --transforms adamw \
  --output "$repo/shape3d_seven_config_lds_all_targets_simple_loss_50x1_t0_500_q0_99.txt"

echo "[done] $repo/shape3d_seven_config_lds_all_targets_simple_loss_50x1_t0_500_q0_99.txt"
