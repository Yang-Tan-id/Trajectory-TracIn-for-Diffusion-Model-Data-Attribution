#!/usr/bin/env bash
#SBATCH -J 3d-id10-a20each
#SBATCH -o 3d-id10-a20each-%j.out
#SBATCH -e 3d-id10-a20each-%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/score_adamw_single_timestamp_sweep.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu JAX_NUM_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="0,1,2,3,4,5,6,7,8,9"

cd "$shapes"

echo '[1/2] fused residual/full single-timestamp scores; one load per 39G artifact'
python script/score_adamw_single_timestamp_sweep.py \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids"

schemes=""
for kind in residual full; do
  for timestep in 0 49 111 149 222 249 333 349 444 449 549 555 649 666 749 777 849 888 949 999; do
    printf -v scheme 'adamw_%s_single_timestamp_t%03d' "$kind" "$timestep"
    schemes="${schemes:+$schemes,}$scheme"
  done
done

echo '[2/2] cached LDS: 20 timestamps x residual/full x four normalizations x four targets x 10 queries'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$schemes" --prediction-sign=1

echo '[done] ID first10 AdamW single-timestamp LDS sweep'
