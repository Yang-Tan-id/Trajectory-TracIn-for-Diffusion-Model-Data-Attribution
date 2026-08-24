#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/stampede3_das" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/_stampede3_das_lib.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_stampede3_das_lib.sh
source "${SCRIPT_DIR}/_stampede3_das_lib.sh"
stampede3_das_init

PROJECTED_SCRIPT="${CIFAR2_ROOT}/tacc/h100/projected_traj_tracin_score_sweep.sh"
if [[ ! -f "${PROJECTED_SCRIPT}" ]]; then
  echo "Missing projected Traj-TracIn helper: ${PROJECTED_SCRIPT}" >&2
  exit 2
fi

ATTR_NUM_SLOTS="${ATTR_NUM_SLOTS:-${GPU_SLOTS:-2}}"
GPU_PER_NODE="${GPU_PER_NODE:-2}"
PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 7)}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 23)}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
TRAJ_QUERY_OBJECTIVE_VALUE="${TRAJ_QUERY_OBJECTIVE_VALUE:-trajectory_next_checkpoint_noise_mse}"
TRAJ_PARAMETER_SOURCE_VALUE="${TRAJ_PARAMETER_SOURCE_VALUE:-raw}"
PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
PROJECTED_DIMS="${PROJECTED_DIMS:-4096}"
PROJECTED_ARTIFACT_DIR_NAME_VALUE="${PROJECTED_ARTIFACT_DIR_NAME:-projected_traj_tracin_artifacts_${TRAJ_PARAMETER_SOURCE_VALUE}}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/10_5_traj_tracin_raw_nextckpt_query_cache/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export TF_CUDNN_USE_AUTOTUNE="${TF_CUDNN_USE_AUTOTUNE:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

query_task_lines() {
  local seed
  for seed in ${UNPROMPTED_SEEDS_TEXT}; do
    printf 'unprompted_solo|unprompted_solo|unprompted|%s|1\n' "${seed}"
  done
  for seed in ${PROMPTED_SEEDS_TEXT}; do
    printf 'prompted_solo|prompted_solo|horse|%s|0\n' "${seed}"
    printf 'prompted_solo|prompted_solo|automobile|%s|0\n' "${seed}"
    printf 'prompted_multi|prompted_solo|horse,automobile|%s|0\n' "${seed}"
  done
}

query_artifact_for_task() {
  local score_mode="$1"
  local query="$2"
  local seed="$3"
  local unprompted_flag="$4"
  if [[ "${unprompted_flag}" == "1" || "${score_mode}" == unprompted_* ]]; then
    printf '%s/result/%s/%s/%s/train_seed_%s/unprompted/initial_seed_%s/shared_query/proj_%s/query_gradient_artifact.npz' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${PROJECTED_ARTIFACT_DIR_NAME_VALUE}" "${score_mode}" "${TRAIN_SEED}" "${seed}" "${PROJECTED_CACHE_DIM}"
  else
    printf '%s/result/%s/%s/%s/train_seed_%s/query_%s/initial_seed_%s/shared_query/proj_%s/query_gradient_artifact.npz' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${PROJECTED_ARTIFACT_DIR_NAME_VALUE}" "${score_mode}" "${TRAIN_SEED}" "$(path_tag "${query}")" "${seed}" "${PROJECTED_CACHE_DIM}"
  fi
}

ensure_sample() {
  local sample_mode="$1"
  local query_env="$2"
  local seed="$3"
  local prompt_tag seed_tag ckpt_stem sample_run_root sample_done_file sample_lock_dir
  prompt_tag="$(path_tag "${query_env}")"
  printf -v seed_tag "%06d" "${seed}"
  printf -v ckpt_stem "seed_%s_epoch_%04d" "${TRAIN_SEED}" "${JAX_EPOCHS}"
  sample_run_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/sample/cifar/prompt_${prompt_tag}/model_${sample_mode}__ckpt_${ckpt_stem}"
  mkdir -p "${sample_run_root}"
  sample_done_file="${sample_run_root}/seed_${seed_tag}/trajectory_xt.npy"
  sample_lock_dir="${sample_run_root}/.sample_seed_${seed_tag}.lock"

  if [[ -f "${sample_done_file}" ]]; then
    echo "[sample] reuse existing ${sample_done_file}" >&2
  elif mkdir "${sample_lock_dir}" 2>/dev/null; then
    trap 'rmdir "${sample_lock_dir}" 2>/dev/null || true' RETURN
    if [[ -f "${sample_done_file}" ]]; then
      echo "[sample] reuse existing ${sample_done_file}" >&2
    else
      echo "[sample] generating ${sample_done_file}" >&2
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      TRAIN_SEED="${TRAIN_SEED}" \
      SAMPLE_MODEL_MODE="${sample_mode}" \
      ATTRIBUTION_SAMPLE_MODEL_MODE="${sample_mode}" \
      UNPROMPTED_SAMPLE_MODEL_MODE="${sample_mode}" \
      QUERY="${query_env}" \
      INITIAL_SEED="${seed}" \
      SAMPLE_SEED="${seed}" \
      SAMPLE_SEEDS="${seed}" \
      bash scripts/00_sample_for_attribution.sh >&2
    fi
    rmdir "${sample_lock_dir}" 2>/dev/null || true
    trap - RETURN
  else
    echo "[sample] waiting for locked sample ${sample_done_file}" >&2
    waited=0
    while [[ ! -f "${sample_done_file}" ]]; do
      if (( waited >= ${SAMPLE_LOCK_WAIT_SECONDS:-21600} )); then
        echo "Timed out waiting for sample ${sample_done_file}" >&2
        exit 1
      fi
      sleep 30
      waited=$((waited + 30))
    done
    echo "[sample] reuse after wait ${sample_done_file}" >&2
  fi
  printf '%s' "${sample_run_root}"
}

run_one_query_task() {
  local i="$1"
  local slot="$2"
  local gpu="$3"
  local sample_mode score_mode query seed unprompted_flag
  IFS='|' read -r sample_mode score_mode query seed unprompted_flag <<<"${TASKS[$i]}"
  local artifact
  artifact="$(query_artifact_for_task "${score_mode}" "${query}" "${seed}" "${unprompted_flag}")"
  local log="${LOG_ROOT}/task_${i}__query_cache__${score_mode}__$(path_tag "${query}")__seed_${seed}.log"
  if [[ -f "${artifact}" ]]; then
    echo "[query-cache-skip] task=${i} existing=${artifact}" >"${log}"
    return 0
  fi

  local query_env="${query}"
  if [[ "${unprompted_flag}" == "1" ]]; then
    query_env="unconditional"
  fi
  local sample_run_root
  sample_run_root="$(ensure_sample "${sample_mode}" "${query_env}" "${seed}")"

  echo "[worker ${slot}] query-cache task=${i} sample_mode=${sample_mode} score_mode=${score_mode} query=${query} seed=${seed} gpu=${gpu} -> ${log}"
  run_gpu_slot "${slot}" env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    GPU_IDS="${gpu}" \
    JAX_NUM_DEVICES=1 \
    PYTHONUNBUFFERED=1 \
    TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE_VALUE}" \
    TRAJ_PARAMETER_SOURCE="${TRAJ_PARAMETER_SOURCE_VALUE}" \
    TRAJ_SAVE_QUERY_NORMALIZED_SCORES=0 \
    PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM}" \
    PROJECTED_DIMS="${PROJECTED_DIMS}" \
    PROJECTED_ARTIFACT_DIR_NAME="${PROJECTED_ARTIFACT_DIR_NAME_VALUE}" \
    RUN_TRAIN_STAGE=0 \
    RUN_QUERY_STAGE=1 \
    RUN_SCORE_SWEEP=0 \
    ATTRIBUTION_SAMPLE_DIR="${sample_run_root}" \
    CIFAR2_ROOT="${CIFAR2_ROOT}" \
    REFINE_ROOT="${REFINE_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
    TRAIN_SEED="${TRAIN_SEED}" \
    SAMPLE_MODEL_MODE="${sample_mode}" \
    UNPROMPTED_SAMPLE_MODEL_MODE="${sample_mode}" \
    ATTRIBUTION_SAMPLE_MODEL_MODE="${sample_mode}" \
    ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}" \
    UNPROMPTED_SCORE_MODEL_MODE="${score_mode}" \
    QUERY="${query_env}" \
    INITIAL_SEED="${seed}" \
    SAMPLE_SEED="${seed}" \
    SAMPLE_SEEDS="${seed}" \
    UNPROMPTED="${unprompted_flag}" \
    bash "${PROJECTED_SCRIPT}" \
    >"${log}" 2>&1
}

mapfile -t TASKS < <(query_task_lines)
total_tasks="${#TASKS[@]}"

echo "Projected Traj-TracIn raw next-checkpoint query-gradient cache"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; total_query_tasks=${total_tasks}"
echo "objective=${TRAJ_QUERY_OBJECTIVE_VALUE}; parameter_source=${TRAJ_PARAMETER_SOURCE_VALUE}; projected_cache_dim=${PROJECTED_CACHE_DIM}"
echo "slots=${ATTR_NUM_SLOTS}; gpu_per_node=${GPU_PER_NODE}; artifact_dir=${PROJECTED_ARTIFACT_DIR_NAME_VALUE}; logs=${LOG_ROOT}"

pids=()
for ((slot = 0; slot < ATTR_NUM_SLOTS; slot++)); do
  gpu=$((slot % GPU_PER_NODE))
  worker_log="${LOG_ROOT}/worker_${slot}.log"
  echo "Launch query-cache worker slot=${slot} gpu=${gpu} -> ${worker_log}"
  (
    for ((i = slot; i < total_tasks; i += ATTR_NUM_SLOTS)); do
      run_one_query_task "${i}" "${slot}" "${gpu}"
    done
  ) >"${worker_log}" 2>&1 &
  pids+=("$!")
done

wait_all "${pids[@]}"

missing=0
for ((i = 0; i < total_tasks; i++)); do
  IFS='|' read -r _sample_mode score_mode query seed unprompted_flag <<<"${TASKS[$i]}"
  artifact="$(query_artifact_for_task "${score_mode}" "${query}" "${seed}" "${unprompted_flag}")"
  if [[ -f "${artifact}" ]]; then
    echo "Found query artifact for task=${i}: ${artifact}"
  else
    echo "Missing query artifact for task=${i}: ${artifact}" >&2
    missing=$((missing + 1))
  fi
done
if (( missing > 0 )); then
  echo "Query-cache job missing ${missing} expected query artifacts." >&2
  exit 1
fi
echo "Query-cache job complete."
