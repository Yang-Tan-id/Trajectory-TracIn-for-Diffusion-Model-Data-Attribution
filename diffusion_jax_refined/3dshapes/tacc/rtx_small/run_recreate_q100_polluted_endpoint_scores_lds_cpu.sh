#!/usr/bin/env bash
#SBATCH -J 3d-q100-pscore
#SBATCH -o 3d-q100-pscore-%j.out
#SBATCH -e 3d-q100-pscore-%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH -t 24:00:00
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
export TRACIN_SCORE_CHECKPOINT_WEIGHTING=stored_lr
export TRACIN_SCORE_TIMESTEP_WEIGHTING=uniform

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_ids="${QUERY_IDS:-100}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
train_root="$shapes/result/$experiment/model/prompted_solo/seed_${seed}_train_gradient"
raw_train="$train_root/traj_tracin_recreate_raw_mc1_aligned10x1/train_datapoint_gradient_artifact.npz"
adamw_train="$train_root/traj_tracin_recreate_adamw_dual_mc1_aligned10x1/train_datapoint_gradient_artifact.npz"
score_train_kinds="${SCORE_TRAIN_KINDS:-raw,adamw_residual,adamw_full}"

if [[ ",$score_train_kinds," == *,raw,* ]]; then
  [[ -f "$raw_train" ]] || { echo "Missing raw train artifact: $raw_train" >&2; exit 1; }
fi
if [[ ",$score_train_kinds," == *,adamw_residual,* || ",$score_train_kinds," == *,adamw_full,* ]]; then
  [[ -f "$adamw_train" ]] || { echo "Missing AdamW train artifact: $adamw_train" >&2; exit 1; }
fi

IFS=',' read -r -a train_kinds <<<"$score_train_kinds"
for train_kind in "${train_kinds[@]}"; do
  case "$train_kind" in
    raw|adamw_residual|adamw_full) ;;
    *) echo "Invalid SCORE_TRAIN_KINDS entry: $train_kind" >&2; exit 1 ;;
  esac
done

declare -A query_namespaces=(
  [delta_raw]="recreate_q100_polluted_endpoint_next_delta_raw_10t"
  [delta_l2normalized]="recreate_q100_polluted_endpoint_next_delta_l2normalized_10t"
)

score_schemes=()

run_score() {
  local train_kind="$1" delta_kind="$2" reduction="$3"
  local train_artifact add_history=()
  case "$train_kind" in
    raw) train_artifact="$raw_train" ;;
    adamw_residual) train_artifact="$adamw_train" ;;
    adamw_full)
      train_artifact="$adamw_train"
      add_history=(--add-optimizer-history)
      ;;
    *) echo "unknown train kind: $train_kind" >&2; return 1 ;;
  esac

  local namespace="recreate_${train_kind}_polluted_endpoint_${delta_kind}_${reduction}_q100"
  local contraction="$reduction"
  [[ "$reduction" == "termwise_squared" ]] || true
  export TRACIN_SCORE_CONTRACTION="$contraction"

  echo "[score] train=$train_kind delta=$delta_kind reduction=$reduction"
  python "$shapes/script/run_traj_tracin_queries_and_scores.py" \
    --execute --experiment "$experiment" --train-seed "$seed" \
    --query-file "$query_file" --query-ids "$query_ids" --gpus 0 \
    --skip-sampling --skip-query-gradient \
    --artifact-namespace "${query_namespaces[$delta_kind]}" \
    --train-artifact "$train_artifact" \
    --score-output-namespace "$namespace" \
    --num-snapshots 10 \
    "${add_history[@]}"
  score_schemes+=("$namespace")
}

cd "$shapes"
for delta_kind in delta_raw delta_l2normalized; do
  for train_kind in "${train_kinds[@]}"; do
    run_score "$train_kind" "$delta_kind" linear
    run_score "$train_kind" "$delta_kind" termwise_squared
    run_score "$train_kind" "$delta_kind" timestamp_sum_squared
  done
done

schemes_csv="$(IFS=,; echo "${score_schemes[*]}")"
echo "[lds] ${#score_schemes[@]} score families x 4 normalization variants x 4 targets"
python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$schemes_csv" --prediction-sign=1

echo "[done] Q${query_ids} polluted-endpoint raw/normalized delta scores and LDS"
