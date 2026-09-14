#!/usr/bin/env bash
#SBATCH -J 3d-traj-a100s
#SBATCH -o 3d-traj-a100s-%j.out
#SBATCH -e 3d-traj-a100s-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
resolve_repo_root() {
  local start candidate
  for start in "${REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}"; do
    [[ -n "${start}" && -d "${start}" ]] || continue
    candidate="$(cd "${start}" && pwd)"
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_traj_tracin_aligned100x1_stream.py" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done
  done
  return 1
}

REPO_ROOT="$(resolve_repo_root)" || {
  echo "Could not locate the 3D Shapes aligned100x1 streaming driver" >&2
  exit 1
}
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
DRIVER="${SHAPES_ROOT}/script/run_traj_tracin_aligned100x1_stream.py"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export STREAM_BATCH_SIZE="${STREAM_BATCH_SIZE:-8}"
export GPU_SLOTS=16
export GPU_PER_NODE=4
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CUDNN_USE_AUTOTUNE="${TF_CUDNN_USE_AUTOTUNE:-0}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"

LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/traj_tracin_aligned100x1_stream_h100/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

run_slot() {
  local slot="$1"
  shift
  local gpu="$((slot % GPU_PER_NODE))"
  ibrun -n 1 -o "${slot}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
    "$@"
}

wait_all() {
  local label="$1"
  shift
  local failed=0 pid
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  if (( failed != 0 )); then
    echo "At least one ${label} worker failed; inspect ${LOG_ROOT}" >&2
    exit 1
  fi
}

echo "3D Shapes aligned 100 timestamps x 1 MC streaming Traj TracIn"
echo "nodes=4 gpus=16 queries=10 attribution_points=5000 checkpoints=49"
echo "query gradients are cached first; train gradients are never saved"
echo "repo=${REPO_ROOT} experiment=${EXPERIMENT_TAG} train_seed=${TRAIN_SEED}"
echo "logs=${LOG_ROOT}"

echo "[phase 1/3] query gradients: 10 queries on slots 0-9"
query_pids=()
for query_id in $(seq 0 9); do
  slot="${query_id}"
  gpu="$((slot % GPU_PER_NODE))"
  echo "[query launch] query=${query_id} slot=${slot} local_gpu=${gpu}"
  (
    run_slot "${slot}" \
      "${PYTHON_BIN}" "${DRIVER}" query \
      --experiment "${EXPERIMENT_TAG}" \
      --train-seed "${TRAIN_SEED}" \
      --epochs "${JAX_EPOCHS}" \
      --query-id "${query_id}" \
      --gpu "${gpu}" \
      --python-bin "${PYTHON_BIN}"
  ) >"${LOG_ROOT}/query_${query_id}.log" 2>&1 &
  query_pids+=("$!")
done
wait_all "query-gradient" "${query_pids[@]}"
echo "[phase 1/3] all 10 query-gradient artifacts complete"

echo "[phase 2/3] streaming train gradients and scores: 16 candidate shards"
stream_pids=()
for slot in $(seq 0 15); do
  gpu="$((slot % GPU_PER_NODE))"
  echo "[stream launch] shard=${slot}/16 slot=${slot} local_gpu=${gpu}"
  (
    run_slot "${slot}" \
      "${PYTHON_BIN}" "${DRIVER}" stream \
      --experiment "${EXPERIMENT_TAG}" \
      --train-seed "${TRAIN_SEED}" \
      --epochs "${JAX_EPOCHS}" \
      --shard-index "${slot}" \
      --shard-count 16 \
      --batch-size "${STREAM_BATCH_SIZE}" \
      --python-bin "${PYTHON_BIN}"
  ) >"${LOG_ROOT}/stream_shard_${slot}.log" 2>&1 &
  stream_pids+=("$!")
done
wait_all "stream-score" "${stream_pids[@]}"
echo "[phase 2/3] all 16 stream-score shards complete"

echo "[phase 3/3] merge shards and materialize 4 variants x 10 queries"
"${PYTHON_BIN}" "${DRIVER}" merge \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --shard-count 16 \
  --python-bin "${PYTHON_BIN}"

echo "[done] score namespace=traj_tracin_aligned100x1_stream"
echo "[done] no train_datapoint_gradient_artifact.npz was written"
