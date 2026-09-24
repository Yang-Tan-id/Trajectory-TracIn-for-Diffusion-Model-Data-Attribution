#!/usr/bin/env bash
#SBATCH -J 3d-id10-a10top
#SBATCH -o 3d-id10-a10top-%j.out
#SBATCH -e 3d-id10-a10top-%j.err
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
namespace="adamw_residual_aligned10x10"
ranking_sign="${RANKING_SIGN:--1}"
case "$ranking_sign" in
  -1) direction_tag=negative; direction_title='most negative scores' ;;
  1) direction_tag=positive; direction_title='top positive scores' ;;
  *) echo "RANKING_SIGN must be -1 or 1, got: $ranking_sign" >&2; exit 1 ;;
esac
out_dir="$shapes/result/$experiment/eval/adamw_residual_aligned10x10_indist_first10_top12/run_${SLURM_JOB_ID}"

cd "$shapes"

echo '[1/2] aligned10x10 AdamW residual RAW Top-12'
python script/plot_endpoint_top6_datapoints.py \
  --experiment "$experiment" --train-seed "${TRAIN_SEED:-42}" \
  --query-file "$query_file" \
  --traj-namespace "$namespace" \
  --traj-score-component score \
  --traj-title "ID first10 · reference-next aligned10x10 · AdamW residual · RAW · ${direction_title}" \
  --traj-ranking-sign "$ranking_sign" \
  --traj-output-stem "reference_next_aligned10x10_adamw_residual_raw_${direction_tag}" \
  --only-traj --top-k 12 --output-dir "$out_dir"

echo '[2/2] aligned10x10 AdamW residual BOTH-L2 Top-12'
python script/plot_endpoint_top6_datapoints.py \
  --experiment "$experiment" --train-seed "${TRAIN_SEED:-42}" \
  --query-file "$query_file" \
  --traj-namespace "$namespace" \
  --traj-score-component score_query_train_l2_normalized \
  --traj-title "ID first10 · reference-next aligned10x10 · AdamW residual · BOTH-L2 · ${direction_title}" \
  --traj-ranking-sign "$ranking_sign" \
  --traj-output-stem "reference_next_aligned10x10_adamw_residual_both_l2_${direction_tag}" \
  --only-traj --top-k 12 --output-dir "$out_dir"

echo "[done] $out_dir"
