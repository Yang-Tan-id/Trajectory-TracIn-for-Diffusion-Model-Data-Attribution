#!/usr/bin/env bash
#SBATCH -J 3d-ep10-tsq
#SBATCH -o 3d-ep10-tsq-%j.out
#SBATCH -e 3d-ep10-tsq-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
marker="diffusion_jax_refined/3dshapes/script/score_endpoint_pollute_10x1_timestamp_square.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do repo="$(dirname "$repo")"; done
[[ -f "$repo/$marker" ]] || { echo "cannot locate repository; set REPO_ROOT" >&2; exit 1; }

shapes="$repo/diffusion_jax_refined/3dshapes"
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
# A login shell can retain an exported RUN_TAG from an earlier submission.
# Within Slurm, the current job id must win so logs never mix across retries.
run_tag="${SLURM_JOB_ID:-${RUN_TAG:-manual}}"
log_root="$shapes/result/$experiment/logs/endpoint_pollute_10x1_per_timestamp_square/$run_tag"
mkdir -p "$log_root"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

run_task() {
  local offset="$1" gpu="$2"
  shift 2
  ibrun -n 1 -o "$offset" env CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 "$@"
}

pids=()
for shard in 0 1; do
  run_task "$shard" "$shard" \
    "$python_bin" "$shapes/script/score_endpoint_pollute_10x1_timestamp_square.py" \
    --experiment "$experiment" --train-seed "$train_seed" \
    --query-file "$query_file" --query-ids "$query_ids" \
    --shard-index "$shard" --shard-count 2 \
    >"$log_root/gpu_${shard}.log" 2>&1 &
  pids[$shard]=$!
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "score worker failed; inspect $log_root" >&2; exit 1; }

timestamps=(0 111 222 333 444 555 666 777 888 999)
schemes=()
for timestep in "${timestamps[@]}"; do
  schemes+=("recreate_adamw_full_polluted_endpoint_delta_l2normalized_timestamp_aware_square_t$(printf '%03d' "$timestep")_q0_99")
done
schemes_csv="$(IFS=,; echo "${schemes[*]}")"
prediction_sign="${PREDICTION_SIGN:-1}"
case "$prediction_sign" in
  1|1.0) prediction_sign_tag=p1 ;;
  -1|-1.0) prediction_sign_tag=m1 ;;
  *) echo "PREDICTION_SIGN must be 1 or -1, got $prediction_sign" >&2; exit 1 ;;
esac

ibrun -n 1 -o 0 env JAX_PLATFORMS=cpu \
  "$python_bin" "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "$experiment" --train-seed "$train_seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$schemes_csv" --prediction-sign "$prediction_sign" \
  >"$log_root/lds.log" 2>&1

"$python_bin" "$shapes/script/print_endpoint_pollute_10x1_timestamp_square_lds.py" \
  --experiment "$experiment" --query-file "$query_file" \
  --query-ids "$query_ids" \
  --prediction-sign "$prediction_sign_tag" \
  --workers "${PRINT_WORKERS:-16}" \
  >"$log_root/summary.txt" 2>&1

echo "[done] endpoint-pollute AdamW-full 10x1 per-timestamp aware-square Q0-Q99"
