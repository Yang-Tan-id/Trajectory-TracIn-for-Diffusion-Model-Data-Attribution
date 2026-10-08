#!/usr/bin/env bash
#SBATCH -J 3d-end-score2
#SBATCH -o 3d-end-score2-%j.out
#SBATCH -e 3d-end-score2-%j.err
#SBATCH -p spr
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/square_existing_endpoint_tracin_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu

query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
ids="$(seq -s, 0 99)"
scheme="endpoint_tracin_adamw_full_train10x10_query100x1_final_score_squared_q0_99"

python "$shapes/script/square_existing_endpoint_tracin_scores.py" \
  --experiment experiment1 --train-seed 42 \
  --query-file "$query_file" --query-ids "$ids"

python "$shapes/script/run_traj_tracin_lds_cached.py" --execute \
  --experiment experiment1 --train-seed 42 \
  --query-file "$query_file" --query-ids "$ids" \
  --score-schemes "$scheme" --prediction-sign -1

echo "[done] AdamW endpoint-TracIn final datapoint score squared Q0-Q99"
