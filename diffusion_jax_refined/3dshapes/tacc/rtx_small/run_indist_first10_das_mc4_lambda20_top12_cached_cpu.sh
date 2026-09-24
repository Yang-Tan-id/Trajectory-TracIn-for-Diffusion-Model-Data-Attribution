#!/usr/bin/env bash
#SBATCH -J 3d-id10-d20top
#SBATCH -o 3d-id10-d20top-%j.out
#SBATCH -e 3d-id10-d20top-%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/plot_endpoint_top6_datapoints.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

experiment="${EXPERIMENT_TAG:-experiment1}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
das_namespace="factorized_mc4_indist_first10_original100x1"
out_dir="$shapes/result/$experiment/eval/das_factorized_mc4_indist_first10_top12/run_${SLURM_JOB_ID}"

cd "$shapes"
python script/plot_endpoint_top6_datapoints.py \
  --experiment "$experiment" \
  --train-seed "${TRAIN_SEED:-42}" \
  --query-file "$query_file" \
  --das-namespace "$das_namespace" \
  --das-lambda 20 \
  --das-ranking-sign 1 \
  --only-das \
  --top-k 12 \
  --output-dir "$out_dir"

echo "[done] $out_dir/das_mc4_lambda_20_endpoint_top12.png"
