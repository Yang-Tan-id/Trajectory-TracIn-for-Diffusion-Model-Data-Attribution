#!/usr/bin/env bash
#SBATCH -J 3d-das-lgram
#SBATCH -o 3d-das-lgram-%j.out
#SBATCH -e 3d-das-lgram-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/score_das_local_gram_mixed.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$shapes"
lambdas="0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000"
common=(
  --experiment experiment1
  --train-seed 42
  --query-file queries_in_distribution_plus_zero_seed_100_219.json
  --query-ids 0,1,2,3,4,5,6,7,8,9
  --partition-seed 0
  --lambdas "$lambdas"
)

echo "[1/3] two GPU lanes: independent local Gram inverses, then mixed 5000-point scores"
(
  CUDA_VISIBLE_DEVICES=0 python script/score_das_local_gram_mixed.py \
    "${common[@]}" --group-counts 2,8,20 --device gpu:0
) >"$repo/das_local_gram_gpu0_${SLURM_JOB_ID}.log" 2>&1 &
pid0=$!
(
  CUDA_VISIBLE_DEVICES=1 python script/score_das_local_gram_mixed.py \
    "${common[@]}" --group-counts 4,10 --device gpu:0
) >"$repo/das_local_gram_gpu1_${SLURM_JOB_ID}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

echo "[2/3] cached LDS for every group count and lambda"
for groups in 2 4 8 10 20; do
  namespace="factorized_mc4_indist100q_original100x1_localgram_mix_g${groups}_seed0"
  JAX_PLATFORMS=cpu python script/run_das_lds_cached.py \
    --execute \
    --experiment experiment1 \
    --train-seed 42 \
    --query-file queries_in_distribution_plus_zero_seed_100_219.json \
    --query-ids 0,1,2,3,4,5,6,7,8,9 \
    --artifact-namespace "$namespace" \
    --lambdas "$lambdas" \
    --prediction-sign -1
done

echo "[3/3] query means, including the existing global-5000 baseline"
python script/print_das_local_gram_mixed_lds.py \
  --experiment experiment1 \
  --query-file queries_in_distribution_plus_zero_seed_100_219.json \
  --query-ids 0,1,2,3,4,5,6,7,8,9 \
  --group-counts 1,2,4,8,10,20 \
  --partition-seed 0 \
  --lambdas "$lambdas" \
  --prediction-sign m1

echo "[done] mixed local-Gram DAS subset experiment"
