#!/usr/bin/env bash
#SBATCH -J celeba-rnd-t1k
#SBATCH -o celeba-rnd-t1k-%j.out
#SBATCH -e celeba-rnd-t1k-%j.err
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
marker="diffusion_jax_refined/celeba/script/celeba_random_top1000_70_plan.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do
  repo="$(dirname "$repo")"
done
[[ -f "$repo/$marker" ]] || { echo "ERROR: repository not found" >&2; exit 2; }

celeba="$repo/diffusion_jax_refined/celeba"
plan="$celeba/script/celeba_random_top1000_70_plan.py"
worker="$celeba/script/run_endpoint_top500_removal.py"
slot_worker="$celeba/tacc/rtx_small/run_celeba_random_top1000_q0_69_slot.sh"
query_file="$celeba/queries_attr40_val25_x4.json"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin}"

export REPO_ROOT="$repo"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-attr40_experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export RANDOM_REMOVAL_SEED="${RANDOM_REMOVAL_SEED:-2026}"
export PYTHONPATH="$celeba:$repo/diffusion_jax_refined:$repo/diffusion_jax_refined/legacy_jax${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export JAX_NUM_DEVICES=1
export JAX_DATA_PARALLEL=0
export JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-128}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

log_root="$celeba/result/$EXPERIMENT_TAG/logs/random_top1000_removal_q0_69/${SLURM_JOB_ID}"
mkdir -p "$log_root"
task_file="$log_root/tasks.tsv"
python "$plan" > "$task_file"
[[ "$(wc -l < "$task_file" | tr -d ' ')" == 70 ]] || {
  echo "ERROR: task plan does not contain 70 rows" >&2
  exit 2
}

echo "[definition] Q0-Q69; uniform random top-1000 removal from full 10000"
echo "[definition] deterministic seed=(base=$RANDOM_REMOVAL_SEED, query_id)"
echo "[definition] one independently retrained and evaluated model per query"
echo "[parallel] one rtx-small node x 2 GPUs = 2 resumable slots"
echo "[logs] $log_root"

echo "[preflight] GPU visibility and CelebA training adapter"
ibrun -n 1 -o 0 env JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 python - <<'PY'
import jax
from DM__training_CELEBA_pixel import TrainConfig

devices = jax.devices("gpu")
if not devices:
    raise RuntimeError("rtx-small preflight cannot see a GPU")
if "exclude_indices" not in TrainConfig.__dataclass_fields__:
    raise RuntimeError("CelebA TrainConfig lacks exclude_indices")
print(f"[preflight] JAX={jax.__version__}; GPUs={devices}; training import OK")
PY

echo "[preflight] validate one representative runtime contract"
IFS=$'\t' read -r preflight_method preflight_query < "$task_file"
JAX_PLATFORMS=cpu python "$worker" \
  --method "$preflight_method" \
  --query-id "$preflight_query" \
  --query-file "$query_file" \
  --experiment "$EXPERIMENT_TAG" \
  --train-seed "$TRAIN_SEED" \
  --epochs "$JAX_EPOCHS" \
  --topk 1000 \
  --random-seed "$RANDOM_REMOVAL_SEED" \
  --output-group endpoint_random_top1000_removal_q0_69 \
  --stage validate
echo "[preflight] representative runtime contract valid"

run_slot() {
  local slot="$1"
  ibrun -n 1 -o "$slot" env \
    CUDA_VISIBLE_DEVICES="$slot" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
    REPO_ROOT="$repo" EXPERIMENT_TAG="$EXPERIMENT_TAG" \
    TRAIN_SEED="$TRAIN_SEED" JAX_EPOCHS="$JAX_EPOCHS" \
    RANDOM_REMOVAL_SEED="$RANDOM_REMOVAL_SEED" \
    bash "$slot_worker" "$slot" 2 "$task_file"
}

pids=()
for slot in 0 1; do
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

result_dir="$celeba/result/$EXPERIMENT_TAG/endpoint_random_top1000_removal_q0_69/random_full10k_seeded"
completed="$(find "$result_dir" -type f -name counterfactual_metrics.json 2>/dev/null | wc -l | tr -d ' ')"
echo "[completion] metrics=$completed/70"
[[ "$completed" == 70 ]] || {
  echo "ERROR: expected 70 completed metrics" >&2
  exit 1
}
echo "[done] all 70 CelebA random top-1000 removal models trained and evaluated"
