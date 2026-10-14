#!/usr/bin/env bash
#SBATCH -J 3d-rm-noise-t
#SBATCH -o 3d-rm-noise-t-%j.out
#SBATCH -e 3d-rm-noise-t-%j.err
#SBATCH -p skx-dev
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH -t 02:00:00

set -euo pipefail

repo="${REPO_ROOT:?export REPO_ROOT before sbatch}"
manifest="${METRICS_MANIFEST:?export METRICS_MANIFEST before sbatch}"
script="$repo/diffusion_jax_refined/3dshapes/script/evaluate_removal_noise_timestamp_relationships.py"
query_file="$repo/diffusion_jax_refined/3dshapes/queries_in_distribution_plus_zero_seed_100_219.json"
python_bin="/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python"
[[ -f "$script" ]] || { echo "missing script: $script" >&2; exit 1; }
[[ -s "$manifest" ]] || { echo "missing manifest: $manifest" >&2; exit 1; }
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
query_ids="${QUERY_IDS:-$(seq -s, 0 56)}"
worker_count="${SLURM_NTASKS:-8}"
log_root="$repo/diffusion_jax_refined/3dshapes/result/$experiment/logs/endpoint_removal_noise_timestamp_relationships/${SLURM_JOB_ID}"
mkdir -p "$log_root"
cp "$manifest" "$log_root/completed_metrics_snapshot.txt"

echo "[definition] per-timestamp predicted-noise deviation vs endpoint/trajectory"
echo "[definition] selected=$(wc -l < "$manifest") timestamps=all chunk_size=${NOISE_CHUNK_SIZE:-8}"
echo "[parallel] direct single-node workers=$worker_count"
echo "[logs] $log_root"

pids=()
for ((slot=0; slot<worker_count; slot++)); do
  env JAX_PLATFORMS=cpu JAX_NUM_DEVICES=1 \
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

env JAX_PLATFORMS=cpu JAX_NUM_DEVICES=1 \
  "$python_bin" "$script" \
    --mode summarize \
    --experiment "$experiment" \
    --query-file "$query_file" \
    --query-ids "$query_ids" \
    --metrics-manifest "$manifest" \
    | tee "$log_root/summary.log"

echo "[done] per-timestamp relationship analysis complete"
