#!/usr/bin/env bash
#SBATCH -J 3d-id10-f4top
#SBATCH -o 3d-id10-f4top-%j.out
#SBATCH -e 3d-id10-f4top-%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/plot_endpoint_top6_datapoints.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="0,1,2,3,4,5,6,7,8,9"
query_namespace="loss_direction_original_f_reference_trajectory_100t_indist_first10"
score_variant="${SCORE_VARIANT:-raw}"
ranking_sign="${RANKING_SIGN:-1}"
case "$score_variant" in
  raw|query_l2|train_l2|query_train_l2) ;;
  *) echo "Unsupported SCORE_VARIANT: $score_variant" >&2; exit 1 ;;
esac
case "$ranking_sign" in
  -1) direction_tag=negative; direction_title='most negative scores' ;;
  1) direction_tag=positive; direction_title='top positive scores' ;;
  *) echo "RANKING_SIGN must be -1 or 1, got: $ranking_sign" >&2; exit 1 ;;
esac
score_stem="adamw_training_events_reference_next100t_indist_first10_linear_four_${score_variant}"
score_namespace="traj_tracin_${score_stem}"
out_dir="$shapes/result/$experiment/eval/adamw_training_events_reference_next100t_indist_first10_top12/run_${SLURM_JOB_ID}"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
count="$(find "$shapes/result/$experiment/sample_ddim_eta0_1000" -type f -path "*/seed_*_query_gradient_${query_namespace}/traj_tracin/query_gradient_artifact.npz" | wc -l | tr -d ' ')"
[[ "$count" == 10 ]] || { echo "Expected 10 cached reference query artifacts, found $count" >&2; exit 1; }
mkdir -p "$out_dir/top12"

cd "$shapes"
echo "[1/2] save per-datapoint LINEAR FOUR ${score_variant} scores from cached reference 100t queries"
python script/run_adamw_four_event_original_f_scores.py \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --query-namespace "$query_namespace" --attribution-points 5000 \
  --checkpoint-weighting uniform --contraction linear --methods four \
  --save-score-namespace "$score_namespace" \
  --save-score-method four --save-score-variant "$score_variant" \
  --out-dir "$out_dir/score_summary"

echo "[2/2] render endpoint plus top-12 ${direction_title} training datapoints"
python script/plot_endpoint_top6_datapoints.py \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" \
  --traj-namespace "$score_stem" \
  --traj-title "ID first10 · reference-next 100t · training AdamW FOUR · LINEAR ${score_variant} · ${direction_title}" \
  --traj-ranking-sign "$ranking_sign" \
  --traj-output-stem "reference_next100t_adamw_four_linear_${score_variant}_${direction_tag}" \
  --only-traj --top-k 12 --output-dir "$out_dir/top12"

echo "[done] $out_dir/top12/reference_next100t_adamw_four_linear_${score_variant}_${direction_tag}_endpoint_top12.png"
