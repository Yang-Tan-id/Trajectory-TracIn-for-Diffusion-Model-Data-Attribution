#!/usr/bin/env bash
#SBATCH -J 3d-end100-t1k
#SBATCH -o 3d-end100-t1k-%j.out
#SBATCH -e 3d-end100-t1k-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
marker="diffusion_jax_refined/3dshapes/script/run_retrac_endpoint_topk_removal.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do
  repo="$(dirname "$repo")"
done
[[ -f "$repo/$marker" ]] || {
  echo "cannot locate repository; set REPO_ROOT" >&2
  exit 1
}

shapes="$repo/diffusion_jax_refined/3dshapes"
slot_worker="$shapes/tacc/rtx_small/run_endpoint_tracin_adamw100x1_top1000_resume_slot.sh"
python_bin="/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python"
slot_count=2

[[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 1; }
[[ -f "$slot_worker" ]] || { echo "missing worker: $slot_worker" >&2; exit 1; }

export REPO_ROOT="$repo"
export PYTHON_BIN="$python_bin"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export JAX_NUM_DEVICES=1
export JAX_DATA_PARALLEL=0
export JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-16}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"

log_root="$shapes/result/$EXPERIMENT_TAG/logs/endpoint_tracin_adamw100x1_top1000_resume/${SLURM_JOB_ID}"
mkdir -p "$log_root"

echo "[definition] resume Endpoint-TracIn AdamW-full top-1000 only; Q0-Q59"
echo "[definition] existing checkpoints and evaluations are skipped"
echo "[parallel] 1 RTX node x 2 GPUs = $slot_count static query shards"
echo "[logs] $log_root"

# Start a real Slurm step immediately and validate every score artifact before
# assigning expensive retraining work to the GPUs.
ibrun -n 1 -o 0 env JAX_PLATFORMS=cpu \
  bash "$slot_worker" validate 0 1 >"$log_root/preflight.log" 2>&1

pids=()
for ((slot=0; slot<slot_count; slot+=1)); do
  ibrun -n 1 -o "$slot" env \
    CUDA_VISIBLE_DEVICES="$slot" \
    JAX_PLATFORMS=cuda \
    JAX_NUM_DEVICES=1 \
    bash "$slot_worker" run "$slot" "$slot_count" \
    >"$log_root/slot_${slot}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || {
  echo "worker failed; inspect $log_root" >&2
  exit 1
}

ibrun -n 1 -o 0 env JAX_PLATFORMS=cpu \
  "$python_bin" "$shapes/script/summarize_endpoint_tracin_adamw100x1_topk_removal.py" \
  --experiment "$EXPERIMENT_TAG" \
  --query-ids "$(seq -s, 0 59)" \
  | tee "$log_root/summary.txt"

echo "[done] all 60 Endpoint-TracIn top-1000 retrain/eval tasks complete"
