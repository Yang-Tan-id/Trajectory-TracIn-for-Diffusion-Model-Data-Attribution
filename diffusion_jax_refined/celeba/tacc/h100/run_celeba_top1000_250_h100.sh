#!/usr/bin/env bash
#SBATCH -J celeba-t1k-250
#SBATCH -o celeba-t1k-250-%j.out
#SBATCH -e celeba-t1k-250-%j.err
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
marker="diffusion_jax_refined/celeba/script/celeba_top1000_250_plan.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do
  repo="$(dirname "$repo")"
done
[[ -f "$repo/$marker" ]] || { echo "ERROR: repository not found" >&2; exit 2; }

celeba="$repo/diffusion_jax_refined/celeba"
plan="$celeba/script/celeba_top1000_250_plan.py"
worker="$celeba/script/run_endpoint_top500_removal.py"
slot_worker="$celeba/tacc/h100/run_celeba_top1000_250_slot.sh"
query_file="$celeba/queries_attr40_val25_x4.json"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin}"

export REPO_ROOT="$repo"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-attr40_experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export PYTHONPATH="$celeba:$repo/diffusion_jax_refined:$repo/diffusion_jax_refined/legacy_jax${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export JAX_NUM_DEVICES=1
export JAX_DATA_PARALLEL=0
export JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-128}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"

log_root="$celeba/result/$EXPERIMENT_TAG/logs/endpoint_top1000_removal_250/${SLURM_JOB_ID}"
mkdir -p "$log_root"
task_file="$log_root/tasks.tsv"
python "$plan" --mode tasks --query-file "$query_file" > "$task_file"
[[ "$(wc -l < "$task_file" | tr -d ' ')" == 250 ]] || {
  echo "ERROR: task plan does not contain 250 rows" >&2
  exit 2
}

echo "[definition] full-10000 candidate top-1000 removal; output=endpoint_top1000_removal_q0_69"
echo "[definition] Both-L2: Pollute/TracIn/ReTrac Q10-Q69 (180 models)"
echo "[definition] Endpoint-TracIn raw Q0-Q69 (70 models)"
echo "[definition] total=250 independent retrain+evaluation tasks"
echo "[parallel] 4 H100 nodes x 4 GPUs = 16 static resumable slots"
echo "[logs] $log_root"

echo "[preflight] H100 visibility and CelebA training adapter"
ibrun -n 1 -o 0 env JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 python - <<'PY'
import jax
from DM__training_CELEBA_pixel import TrainConfig, train

devices = jax.devices("gpu")
if not devices:
    raise RuntimeError("H100 preflight cannot see a GPU")
if "exclude_indices" not in TrainConfig.__dataclass_fields__:
    raise RuntimeError("CelebA TrainConfig lacks exclude_indices")
print(f"[preflight] JAX={jax.__version__}; GPUs={devices}; training import OK")
PY

echo "[preflight] validate all 250 score/input contracts"
while IFS=$'\t' read -r method query_id; do
  JAX_PLATFORMS=cpu python "$worker" \
    --method "$method" \
    --query-id "$query_id" \
    --query-file "$query_file" \
    --experiment "$EXPERIMENT_TAG" \
    --train-seed "$TRAIN_SEED" \
    --epochs "$JAX_EPOCHS" \
    --topk 1000 \
    --output-group endpoint_top1000_removal_q0_69 \
    --stage validate >/dev/null
done < "$task_file"
echo "[preflight] all 250 tasks valid"

run_slot() {
  local slot="$1"
  local gpu="$((slot % 4))"
  ibrun -n 1 -o "$slot" env \
    CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    REPO_ROOT="$repo" EXPERIMENT_TAG="$EXPERIMENT_TAG" \
    TRAIN_SEED="$TRAIN_SEED" JAX_EPOCHS="$JAX_EPOCHS" \
    bash "$slot_worker" "$slot" 16 "$task_file"
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
  echo "ERROR: at least one slot failed; inspect $log_root/slot_*.log" >&2
  exit 1
}

completed="$(find "$celeba/result/$EXPERIMENT_TAG/endpoint_top1000_removal_q0_69" \
  -type f -name counterfactual_metrics.json | wc -l | tr -d ' ')"
echo "[completion] metrics=$completed/250"
[[ "$completed" == 250 ]] || {
  echo "ERROR: expected 250 completed metrics" >&2
  exit 1
}
echo "[done] all 250 CelebA top-1000 removal models trained and evaluated"
