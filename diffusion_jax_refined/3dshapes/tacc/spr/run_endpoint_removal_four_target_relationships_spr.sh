#!/usr/bin/env bash
#SBATCH -J 3d-rm-4target
#SBATCH -o 3d-rm-4target-%j.out
#SBATCH -e 3d-rm-4target-%j.err
#SBATCH -p spr
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH -t 04:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
marker="diffusion_jax_refined/3dshapes/script/evaluate_removal_target_relationships.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do repo="$(dirname "$repo")"; done
[[ -f "$repo/$marker" ]] || { echo "cannot locate repository; set REPO_ROOT" >&2; exit 1; }

shapes="$repo/diffusion_jax_refined/3dshapes"
script="$repo/$marker"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
python_bin="/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python"
[[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export JAX_NUM_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${SCRATCH:-/tmp}/matplotlib-${USER}"
mkdir -p "$MPLCONFIGDIR"

experiment="${EXPERIMENT_TAG:-experiment1}"
query_ids="${QUERY_IDS:-$(seq -s, 0 59)}"
worker_count="${SLURM_NTASKS:-8}"
log_root="$shapes/result/$experiment/logs/endpoint_removal_four_target_relationships/${SLURM_JOB_ID}"
mkdir -p "$log_root"
removal_root="$shapes/result/$experiment/retrac_endpoint_topk_removal_60q/endpoint_tracin_adamw_full_100x1_raw_neg"
manifest="$log_root/completed_metrics_snapshot.txt"
if [[ -n "${METRICS_MANIFEST:-}" ]]; then
  [[ -f "$METRICS_MANIFEST" ]] || { echo "missing METRICS_MANIFEST: $METRICS_MANIFEST" >&2; exit 1; }
  cp "$METRICS_MANIFEST" "$manifest"
else
  find "$removal_root" -type f -name counterfactual_metrics.json -print | sort > "$manifest"
fi
snapshot_count="$(wc -l < "$manifest" | tr -d ' ')"
(( snapshot_count > 0 )) || { echo "no completed removal checkpoints found" >&2; exit 1; }

echo "[definition] frozen snapshot of completed Endpoint-TracIn removal checkpoints: $snapshot_count"
echo "[definition] evaluate noise_trajectory + simple_loss on CPU; no training"
echo "[parallel] workers=$worker_count cpus_per_worker=${SLURM_CPUS_PER_TASK:-6}"
echo "[logs] $log_root"

pids=()
for ((slot=0; slot<worker_count; slot++)); do
  ibrun -n 1 -o "$slot" env \
    JAX_PLATFORMS=cpu JAX_NUM_DEVICES=1 \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    "$python_bin" "$script" \
      --mode evaluate \
      --experiment "$experiment" \
      --query-file "$query_file" \
      --query-ids "$query_ids" \
      --metrics-manifest "$manifest" \
      --noise-chunk-size "${NOISE_CHUNK_SIZE:-8}" \
      --shard-index "$slot" \
      --shard-count "$worker_count" \
      >"$log_root/worker_${slot}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "one or more workers failed; inspect $log_root" >&2; exit 1; }

ibrun -n 1 -o 0 env JAX_PLATFORMS=cpu JAX_NUM_DEVICES=1 \
  "$python_bin" "$script" \
    --mode summarize \
    --experiment "$experiment" \
    --query-file "$query_file" \
    --query-ids "$query_ids" \
    --metrics-manifest "$manifest" \
    | tee "$log_root/summary.log"

echo "[done] four-target relationship experiment complete"
