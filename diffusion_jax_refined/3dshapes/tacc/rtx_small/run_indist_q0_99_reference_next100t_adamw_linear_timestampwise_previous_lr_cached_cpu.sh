#!/usr/bin/env bash
#SBATCH -J 3d-q0-99-a100
#SBATCH -o 3d-q0-99-a100-%j.out
#SBATCH -e 3d-q0-99-a100-%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH -t 12:00:00
#SBATCH -A IRI26004

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_adamw_four_event_original_f_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export JAX_NUM_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="$(seq -s, 0 99)"
query_namespace="loss_direction_original_f_reference_trajectory_100t_indist100q"
event_root="$shapes/result/$experiment/fixed_checkpoint_adamw_four_events_n5000"
sample_root="$shapes/result/$experiment/sample_ddim_eta0_1000"
out_root="$shapes/result/$experiment/eval/adamw_training_events_reference_next100t_indist_q0_99/run_${SLURM_JOB_ID}"

[[ -f "$query_file" ]] || { echo "Missing query manifest: $query_file" >&2; exit 1; }
[[ -d "$event_root" ]] || { echo "Missing fixed-checkpoint AdamW events: $event_root" >&2; exit 1; }

query_list="${SLURM_TMPDIR:-/tmp}/reference_q0_99_events_${SLURM_JOB_ID:-$$}.txt"
find "$sample_root" -type f \
  -path "*/seed_*_query_gradient_${query_namespace}/traj_tracin/query_gradient_artifact.npz" \
  >"$query_list"
missing=0
for initial_seed in $(seq 100 199); do
  printf -v seed_name '%06d' "$initial_seed"
  if ! grep -q "/seed_${seed_name}_query_gradient_" "$query_list"; then
    echo "Missing reference query artifact for initial seed $initial_seed" >&2
    missing=1
  fi
done
[[ "$missing" == 0 ]] || exit 1

mkdir -p "$out_root/linear" "$out_root/timestamp_sum_squared_previous_lr"
cd "$repo"

echo '[1/2] original non-aligned 100t linear AdamW FOUR/full and FOUR_RESIDUAL'
python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --query-namespace "$query_namespace" --attribution-points 5000 \
  --methods four,four_residual \
  --checkpoint-weighting uniform --contraction linear \
  --out-dir "$out_root/linear"

echo '[2/2] non-aligned 100t: sum checkpoints within each timestamp, square, then sum timestamps'
echo '[weighting] checkpoint c receives the stored outer LR weight from checkpoint c-1; checkpoint 0 is zero'
python "$shapes/script/run_adamw_four_event_original_f_scores.py" \
  --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --query-namespace "$query_namespace" --attribution-points 5000 \
  --methods four,four_residual \
  --checkpoint-weighting previous_checkpoint_lr \
  --contraction timestamp_sum_squared \
  --out-dir "$out_root/timestamp_sum_squared_previous_lr"

echo "[done] Q0-Q99 non-aligned AdamW linear and timestamp-wise-square: $out_root"
