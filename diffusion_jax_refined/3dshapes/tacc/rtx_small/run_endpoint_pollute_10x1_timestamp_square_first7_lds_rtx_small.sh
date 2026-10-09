#!/usr/bin/env bash
#SBATCH -J 3d-ep10-f7
#SBATCH -o 3d-ep10-f7-%j.out
#SBATCH -e 3d-ep10-f7-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
marker="diffusion_jax_refined/3dshapes/script/combine_endpoint_pollute_10x1_timestamp_square_first7.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do repo="$(dirname "$repo")"; done
[[ -f "$repo/$marker" ]] || { echo "cannot locate repository; set REPO_ROOT" >&2; exit 1; }

shapes="$repo/diffusion_jax_refined/3dshapes"
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
run_tag="${SLURM_JOB_ID:-${RUN_TAG:-manual}}"
log_root="$shapes/result/$experiment/logs/endpoint_pollute_10x1_timestamp_square_first7/$run_tag"
mkdir -p "$log_root"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

ibrun -n 1 -o 0 \
  "$python_bin" "$shapes/script/combine_endpoint_pollute_10x1_timestamp_square_first7.py" \
  --experiment "$experiment" --train-seed "$train_seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --workers "${COMBINE_WORKERS:-8}" \
  >"$log_root/combine.log" 2>&1

scheme="recreate_adamw_full_polluted_endpoint_delta_l2normalized_timestamp_aware_square_first7_q0_99"
ibrun -n 1 -o 0 \
  "$python_bin" "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "$experiment" --train-seed "$train_seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$scheme" --prediction-sign 1 \
  >"$log_root/lds.log" 2>&1

"$python_bin" "$shapes/script/print_aligned10x10_lds.py" \
  --experiment "$experiment" --query-file "$query_file" \
  --query-ids "$query_ids" \
  --scheme-group endpoint_pollute_adamw10x1_timestamp_square_first7 \
  --prediction-sign p1 --workers "${PRINT_WORKERS:-16}" \
  >"$log_root/summary.txt" 2>&1

echo "[done] first-seven timestamp-square score and P1 LDS"
