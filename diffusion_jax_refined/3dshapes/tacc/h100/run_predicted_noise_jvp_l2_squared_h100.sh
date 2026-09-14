#!/usr/bin/env bash
#SBATCH -J 3d-pnoise-jvp
#SBATCH -o 3d-pnoise-jvp-%j.out
#SBATCH -e 3d-pnoise-jvp-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
resolve_repo_root() {
  local candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
  candidate="$(cd "${candidate}" && pwd)"
  while [[ "${candidate}" != "/" ]]; do
    if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    candidate="$(dirname "${candidate}")"
  done
  return 1
}

REPO_ROOT="$(resolve_repo_root)" || { echo "Could not locate repository" >&2; exit 1; }
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
QUERY_DRIVER="${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"

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
export GPU_PER_NODE=4
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"

LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/predicted_noise_jvp_l2_squared_h100/${SLURM_JOB_ID}"
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
  (( failed == 0 )) || { echo "At least one ${label} worker failed; inspect ${LOG_ROOT}" >&2; exit 1; }
}

echo "3D Shapes predicted-noise JVP L2-squared attribution"
echo "reuse=train TrajTracIn 50 checkpoints x 10 timestamps x 10-MC mean loss"
echo "query=one Gaussian output probe per checkpoint/timestamp; projected dim=4096"
echo "outputs=constant and learning-rate-squared scores; query gradients are deleted after merge"
echo "logs=${LOG_ROOT}"

echo "[phase 1/4] transient predicted-noise probe query gradients"
query_pids=()
for query_id in $(seq 0 9); do
  slot="${query_id}"
  gpu="$((slot % GPU_PER_NODE))"
  (
    run_slot "${slot}" \
      "${PYTHON_BIN}" "${QUERY_DRIVER}" \
      --execute \
      --experiment "${EXPERIMENT_TAG}" \
      --train-seed "${TRAIN_SEED}" \
      --epochs "${JAX_EPOCHS}" \
      --query-ids "${query_id}" \
      --gpus "${gpu}" \
      --skip-sampling \
      --skip-score \
      --artifact-namespace predicted_noise_jvp_l2_squared \
      --query-objective trajectory_predicted_noise_probe \
      --num-snapshots 10 \
      --python-bin "${PYTHON_BIN}"
  ) >"${LOG_ROOT}/query_${query_id}.log" 2>&1 &
  query_pids+=("$!")
done
wait_all "query-probe" "${query_pids[@]}"

echo "[phase 2/4] 16 checkpoint score shards"
score_pids=()
for slot in $(seq 0 15); do
  (
    run_slot "${slot}" \
      "${PYTHON_BIN}" "${SCORE_DRIVER}" score-shard \
      --experiment "${EXPERIMENT_TAG}" \
      --train-seed "${TRAIN_SEED}" \
      --epochs "${JAX_EPOCHS}" \
      --run-id "${SLURM_JOB_ID}" \
      --shard-index "${slot}" \
      --shard-count 16
  ) >"${LOG_ROOT}/score_shard_${slot}.log" 2>&1 &
  score_pids+=("$!")
done
wait_all "JVP-L2 score" "${score_pids[@]}"

echo "[phase 3/4] merge, materialize, and remove transient query gradients"
"${PYTHON_BIN}" "${SCORE_DRIVER}" merge \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --run-id "${SLURM_JOB_ID}" \
  --shard-count 16 \
  --cleanup-query-artifacts

echo "[phase 4/4] cached LDS for constant and lr2 variants"
"${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --score-schemes predicted_noise_jvp_l2_squared

echo "[done] score namespace=traj_tracin_predicted_noise_jvp_l2_squared"
