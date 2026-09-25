#!/usr/bin/env bash
#SBATCH -J 3d-q10-29-sqlr
#SBATCH -o 3d-q10-29-sqlr-%j.out
#SBATCH -e 3d-q10-29-sqlr-%j.err
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH -t 02:00:00
#SBATCH -A IRI26004

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_traj_tracin_queries_and_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export JAX_NUM_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export TRACIN_SCORE_TIMESTEP_ALLOWLIST="0,111,222,333,444,555,666,777,888,999"
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=previous_checkpoint_lr
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform
export TRACIN_SCORE_CONTRACTION=squared

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 10 29)"
query_namespace="loss_direction_original_f_reference_trajectory_100t_indist100q"
train_artifact="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient/traj_tracin_adamw_dual_aligned10x10/train_datapoint_gradient_artifact.npz"
residual_namespace="adamw_residual_aligned10x10_squared_previous_lr"
full_namespace="adamw_full_aligned10x10_squared_previous_lr"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
[[ -f "$train_artifact" ]] || { echo "Missing aligned AdamW train artifact: $train_artifact" >&2; exit 1; }

sample_root="$shapes/result/$experiment/sample_ddim_eta0_1000"
missing=0
for initial_seed in $(seq 110 129); do
  printf -v seed_name '%06d' "$initial_seed"
  if ! find "$sample_root" -type f \
      -path "*/seed_${seed_name}_query_gradient_${query_namespace}/traj_tracin/query_gradient_artifact.npz" \
      -print -quit | grep -q .; then
    echo "Missing reference-next artifact for initial seed $initial_seed" >&2
    missing=1
  fi
done
[[ "$missing" == 0 ]] || exit 1

cd "$shapes"
echo '[1/3] Q10-Q29 reference-next residual: termwise square with previous-checkpoint LR'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" \
  --train-artifact "$train_artifact" \
  --score-output-namespace "$residual_namespace" \
  --num-snapshots 100

echo '[2/3] Q10-Q29 reference-next full AdamW with previous-checkpoint LR'
python script/run_traj_tracin_queries_and_scores.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
  --skip-sampling --skip-query-gradient \
  --artifact-namespace "$query_namespace" \
  --train-artifact "$train_artifact" \
  --score-output-namespace "$full_namespace" \
  --num-snapshots 100 \
  --add-optimizer-history

echo '[3/3] 640 cached LDS evaluations'
python script/run_traj_tracin_lds_cached.py \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$residual_namespace,$full_namespace" \
  --prediction-sign=1

echo '[done] Q10-Q29 reference-next AdamW aligned10x10 termwise-square with previous-checkpoint LR'
