#!/usr/bin/env bash
#SBATCH -J 3d-ret-ep-rm
#SBATCH -o 3d-ret-ep-rm-%j.out
#SBATCH -e 3d-ret-ep-rm-%j.err
#SBATCH -p h100
#SBATCH -A IRI26004
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_retrac_endpoint_topk_removal.py" ]]; do
  repo="$(dirname "$repo")"
done
[[ -f "$repo/diffusion_jax_refined/3dshapes/script/run_retrac_endpoint_topk_removal.py" ]] || {
  echo "Could not locate repository" >&2
  exit 1
}

shapes="$repo/diffusion_jax_refined/3dshapes"
worker="$shapes/script/run_retrac_endpoint_topk_removal.py"
slot_worker="$shapes/tacc/h100/run_retrac_endpoint_topk_removal_slot.sh"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export REPO_ROOT="$repo"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export TF_CUDNN_USE_AUTOTUNE="${TF_CUDNN_USE_AUTOTUNE:-0}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export JAX_NUM_DEVICES=1
export JAX_DATA_PARALLEL=0
export JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-16}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"

log_root="$shapes/result/$EXPERIMENT_TAG/logs/retrac_endpoint_topk_removal_60q/${SLURM_JOB_ID}"
mkdir -p "$log_root"

echo "[definition] Q0-Q59; methods=ReTrac(-query/train-L2),endpoint-pollute(timestamp-square train-L2 raw sign)"
echo "[definition] topk=400 (2% of 20k),1000 (5% of 20k); attributed candidates=5000"
echo "[definition] 240 independent retrains; retain final epoch-200 checkpoint; evaluate endpoint and trajectory MSE"
echo "[parallel] 4 H100 nodes x 4 GPUs = 16 static task shards, 15 retrains per GPU"
echo "[logs] $log_root"

echo "[preflight] Python, JAX, 3D Shapes adapter, and local H100 visibility"
ibrun -n 1 -o 0 env JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 python - <<'PY'
import jax
from DM__training_3DSHAPES_pixel import TrainConfig, train

devices = jax.devices("gpu")
if len(devices) != 4:
    raise RuntimeError(f"expected 4 visible H100 GPUs on the launch node, found {devices}")
if "exclude_indices" not in TrainConfig.__dataclass_fields__:
    raise RuntimeError("3D Shapes TrainConfig lacks exclude_indices support")
print(f"[preflight] JAX={jax.__version__}; GPUs={devices}; adapter/train import OK")
PY

echo "[preflight] validate all 120 score artifacts at maximum topk"
for method in retrac_adamw_both_l2_neg endpoint_pollute_adamw_timestamp_train_l2; do
  for query_id in $(seq 0 59); do
    python "$worker" \
      --method "$method" \
      --query-id "$query_id" \
      --query-file "$query_file" \
      --experiment "$EXPERIMENT_TAG" \
      --train-seed "$TRAIN_SEED" \
      --epochs "$JAX_EPOCHS" \
      --topk 1000 \
      --stage validate >/dev/null
  done
done
echo "[preflight] all score artifacts valid"

run_slot() {
  local slot="$1"
  local gpu="$((slot % 4))"
  ibrun -n 1 -o "$slot" env \
    CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    REPO_ROOT="$repo" EXPERIMENT_TAG="$EXPERIMENT_TAG" \
    TRAIN_SEED="$TRAIN_SEED" JAX_EPOCHS="$JAX_EPOCHS" \
    bash "$slot_worker" "$slot" 16
}

pids=()
for slot in $(seq 0 15); do
  run_slot "$slot" >"$log_root/slot_${slot}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || {
  echo "At least one worker failed; inspect $log_root" >&2
  exit 1
}

python "$shapes/script/summarize_retrac_endpoint_topk_removal.py" \
  --experiment "$EXPERIMENT_TAG" \
  --query-ids "$(seq -s, 0 59)" | tee "$log_root/summary.txt"

echo "[done] all 240 retrain/eval tasks and paired summary complete"
