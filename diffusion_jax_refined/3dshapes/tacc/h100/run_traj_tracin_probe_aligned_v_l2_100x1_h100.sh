#!/usr/bin/env bash
#SBATCH -J 3d-vl2-a100
#SBATCH -o 3d-vl2-a100-%j.out
#SBATCH -e 3d-vl2-a100-%j.err
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
      if [[ -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_expected_residual_jacobian_scores.py" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done
  done
  return 1
}

REPO_ROOT="$(resolve_repo_root)" || { echo "Could not locate repository" >&2; exit 1; }
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
QUERY_DRIVER="${SHAPES_ROOT}/script/run_traj_tracin_queries_and_scores.py"
SCORE_DRIVER="${SHAPES_ROOT}/script/run_expected_residual_jacobian_scores.py"
STAGE="${SHAPES_ROOT}/data_attribution/traj_tracin/01_train_datapoint_gradient.py"

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
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CUDNN_USE_AUTOTUNE="${TF_CUDNN_USE_AUTOTUNE:-0}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"

TRAIN_NAMESPACE="traj_tracin_expected_residual_jacobian_probe_aligned_100x1"
ORIGINAL_QUERY_NAMESPACE="expected_residual_jacobian_probe_aligned_100x1_original_f"
PREDICTED_QUERY_NAMESPACE="expected_residual_jacobian_probe_aligned_100x1_predicted_noise"
SCORE_PREFIX="traj_tracin_expected_residual_jacobian_probe_aligned_100x1_v_l2"
ARTIFACT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_solo/seed_${TRAIN_SEED}_train_gradient/${TRAIN_NAMESPACE}/train_datapoint_gradient_artifact.npz"
PART_DIR="${ARTIFACT}.parts"
LOG_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/logs/traj_tracin_probe_aligned_v_l2_100x1_h100/${SLURM_JOB_ID}"
mkdir -p "${PART_DIR}" "${LOG_ROOT}"

run_slot() {
  local slot="$1"
  shift
  local gpu="$((slot % 4))"
  ibrun -n 1 -o "${slot}" env CUDA_VISIBLE_DEVICES="${gpu}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda "$@"
}

wait_all() {
  local label="$1"
  shift
  local failed=0 pid
  for pid in "$@"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "At least one ${label} worker failed; inspect ${LOG_ROOT}" >&2; exit 1; }
}

echo "3D Shapes probe-aligned v-L2 100 timestamps x 1 MC"
echo "nodes=4 gpus=16 checkpoints=50 queries=10 batch=${STREAM_BATCH_SIZE}"
echo "artifact=${ARTIFACT}"
echo "logs=${LOG_ROOT}"

run_query_phase() {
  local namespace="$1"
  local objective="$2"
  local label="$3"
  local pids=() query_id slot gpu
  echo "[query phase] ${label}"
  for query_id in $(seq 0 9); do
    slot="${query_id}"
    gpu="$((slot % 4))"
    (
      run_slot "${slot}" "${PYTHON_BIN}" "${QUERY_DRIVER}" \
        --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
        --epochs "${JAX_EPOCHS}" --query-ids "${query_id}" --gpus "${gpu}" \
        --skip-sampling --skip-score --artifact-namespace "${namespace}" \
        --query-objective "${objective}" --num-snapshots 100 --python-bin "${PYTHON_BIN}"
    ) >"${LOG_ROOT}/query_${label}_${query_id}.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${label} query" "${pids[@]}"
}

echo "[phase 1/5] original next-raw query gradients"
run_query_phase "${ORIGINAL_QUERY_NAMESPACE}" trajectory_next_checkpoint_noise_mse original

echo "[phase 2/5] predicted-noise query gradients"
run_query_phase "${PREDICTED_QUERY_NAMESPACE}" trajectory_predicted_noise_probe predicted

echo "[phase 3/5] 16 saved train-gradient checkpoint shards"
train_pids=()
for slot in $(seq 0 15); do
  (
    run_slot "${slot}" env \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" TRAIN_SEED="${TRAIN_SEED}" JAX_EPOCHS="${JAX_EPOCHS}" \
      QUERY="shape_cube,object_hue_0,wall_hue_0,floor_hue_0" INITIAL_SEED=0 SAMPLE_SEED=0 \
      DATAPOINT_MODEL_MODE=prompted_solo SAMPLE_MODEL_MODE=prompted_solo ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo \
      TRAJ_QUERY_OBJECTIVE=trajectory_predicted_noise_probe TRAJ_PARAMETER_SOURCE=raw \
      TRAJ_NUM_SNAPSHOTS=100 TRAJ_TRAIN_MC_SAMPLES=1 TRAJ_TRACIN_PROJ_DIM=4096 \
      TRAJ_SCORE_BATCH_SIZE="${STREAM_BATCH_SIZE}" TRAJ_USE_SAVED_TRAJECTORY=0 \
      TRAJ_TRACIN_STAGE_MODE=train TRAJ_TRACIN_STAGE_ARTIFACT_PATH="${ARTIFACT}" \
      TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="${ARTIFACT}" \
      TRAJ_TRACIN_CKPT_SHARD_INDEX="${slot}" TRAJ_TRACIN_CKPT_SHARD_COUNT=16 \
      TRAJ_TRACIN_SKIP_STAGE_MERGE=1 TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS=0 \
      TRAJ_TRACIN_TRAIN_DECOMPOSE_RESIDUAL_JACOBIAN=1 TRAJ_TRACIN_JACOBIAN_NORM_PROBES=1 \
      TRAJ_TRACIN_TRAIN_BATCH_MODE=vmap TRAJ_TRACIN_TRAIN_BATCH_DTYPE=float32 \
      "${PYTHON_BIN}" "${STAGE}"
  ) >"${LOG_ROOT}/train_ckpt_shard_${slot}.log" 2>&1 &
  train_pids+=("$!")
done
wait_all train-gradient "${train_pids[@]}"

part_count="$(find "${PART_DIR}" -maxdepth 1 -type f -name 'ckpt_*.npz' | wc -l | tr -d ' ')"
[[ "${part_count}" == "50" ]] || { echo "Expected 50 checkpoint parts; found ${part_count}" >&2; exit 1; }

echo "[phase 4/5] 16 saved-gradient score shards"
score_pids=()
for slot in $(seq 0 15); do
  (
    run_slot "${slot}" "${PYTHON_BIN}" "${SCORE_DRIVER}" score-shard \
      --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" --epochs "${JAX_EPOCHS}" \
      --run-id "${SLURM_JOB_ID}" --shard-index "${slot}" --shard-count 16 \
      --num-checkpoints 50 --num-snapshots 100 --train-namespace "${TRAIN_NAMESPACE}" \
      --original-query-namespace "${ORIGINAL_QUERY_NAMESPACE}" \
      --predicted-query-namespace "${PREDICTED_QUERY_NAMESPACE}" \
      --score-namespace-prefix "${SCORE_PREFIX}"
  ) >"${LOG_ROOT}/score_ckpt_shard_${slot}.log" 2>&1 &
  score_pids+=("$!")
done
wait_all score "${score_pids[@]}"

"${PYTHON_BIN}" "${SCORE_DRIVER}" merge \
  --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" --epochs "${JAX_EPOCHS}" \
  --run-id "${SLURM_JOB_ID}" --shard-count 16 --num-checkpoints 50 --num-snapshots 100 \
  --train-namespace "${TRAIN_NAMESPACE}" --original-query-namespace "${ORIGINAL_QUERY_NAMESPACE}" \
  --predicted-query-namespace "${PREDICTED_QUERY_NAMESPACE}" --score-namespace-prefix "${SCORE_PREFIX}"

echo "[phase 5/5] cached LDS: 4 scores x 10 queries x 4 targets = 160"
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" --score-schemes \
  expected_residual_jacobian_probe_aligned_100x1_v_l2_original_f,expected_residual_jacobian_probe_aligned_100x1_v_l2_predicted_noise

echo "[done] ${SCORE_PREFIX}: raw and query-L2 for original-f and predicted-noise-f"
