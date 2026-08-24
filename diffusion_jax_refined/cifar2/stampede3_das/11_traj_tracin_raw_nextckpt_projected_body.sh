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
TRAJ_RANGES_TEXT="${TRAJ_RANGES:-1-2500 2501-5000 5001-7500 7501-10000}"
TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-4}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
TRAJ_QUERY_OBJECTIVE_VALUE="${TRAJ_QUERY_OBJECTIVE_VALUE:-trajectory_next_checkpoint_noise_mse}"
TRAJ_PARAMETER_SOURCE_VALUE="${TRAJ_PARAMETER_SOURCE_VALUE:-raw}"
PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
PROJECTED_DIMS="${PROJECTED_DIMS:-4096}"
PROJECTED_SCORE_DIM="${PROJECTED_SCORE_DIM:-4096}"
PROJECTED_SCORE_VARIANT="${PROJECTED_SCORE_VARIANT:-raw}"
PROJECTED_ARTIFACT_DIR_NAME_VALUE="${PROJECTED_ARTIFACT_DIR_NAME:-projected_traj_tracin_artifacts_${TRAJ_PARAMETER_SOURCE_VALUE}}"
TRAIN_SCORE_INDEX_RANGES="${TRAIN_SCORE_INDEX_RANGES:-1-10000}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/11_traj_tracin_raw_nextckpt_projected/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export TF_CUDNN_USE_AUTOTUNE="${TF_CUDNN_USE_AUTOTUNE:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

range_tag() {
  local value="$1"
  value="${value//:/-}"
  local start="${value%-*}"
  local end="${value#*-}"
  printf 'range_%s_%s' "${start}" "${end}"
}

task_lines() {
  local seed range
  for seed in ${UNPROMPTED_SEEDS_TEXT}; do
    for range in ${TRAJ_RANGES_TEXT}; do
      printf 'unprompted_solo|unprompted_solo|unprompted|%s|1|%s\n' "${seed}" "${range}"
    done
  done
  for seed in ${PROMPTED_SEEDS_TEXT}; do
    for range in ${TRAJ_RANGES_TEXT}; do
      printf 'prompted_solo|prompted_solo|horse|%s|0|%s\n' "${seed}" "${range}"
      printf 'prompted_solo|prompted_solo|automobile|%s|0|%s\n' "${seed}" "${range}"
      printf 'prompted_multi|prompted_solo|horse,automobile|%s|0|%s\n' "${seed}" "${range}"
    done
  done
}

algorithm_dir_for_task() {
  local unprompted_flag="$1"
  local range="$2"
  local tag
  tag="$(range_tag "${range}")"
  if [[ "${unprompted_flag}" == "1" ]]; then
    printf 'traj_tracin_%s_unprompted_%s_%s' \
      "${TRAJ_QUERY_OBJECTIVE_VALUE}" "${TRAJ_PARAMETER_SOURCE_VALUE}" "${tag}"
  else
    printf 'traj_tracin_%s_%s_%s' \
      "${TRAJ_QUERY_OBJECTIVE_VALUE}" "${TRAJ_PARAMETER_SOURCE_VALUE}" "${tag}"
  fi
}

score_file_for_task() {
  local score_mode="$1"
  local query="$2"
  local seed="$3"
  local unprompted_flag="$4"
  local range="$5"
  local algorithm_dir
  algorithm_dir="$(algorithm_dir_for_task "${unprompted_flag}" "${range}")"
  if [[ "${unprompted_flag}" == "1" || "${score_mode}" == unprompted_* ]]; then
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/unprompted/initial_seed_%s/%s/scores.npy' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${score_mode}" "${TRAIN_SEED}" "${seed}" "${algorithm_dir}"
  else
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/%s/scores.npy' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${score_mode}" "${TRAIN_SEED}" "$(path_tag "${query}")" "${seed}" "${algorithm_dir}"
  fi
}

materialize_compat_scores() {
  local score_file="$1"
  local compat_dir
  compat_dir="$(dirname "${score_file}")"
  local projected_dir="${compat_dir}/proj_${PROJECTED_SCORE_DIM}/${PROJECTED_SCORE_VARIANT}"
  if [[ ! -f "${projected_dir}/scores.npy" || ! -f "${projected_dir}/score_indices.npy" ]]; then
    echo "Projected score output is missing: ${projected_dir}" >&2
    exit 1
  fi
  ln -sfn "proj_${PROJECTED_SCORE_DIM}/${PROJECTED_SCORE_VARIANT}/scores.npy" "${compat_dir}/scores.npy"
  ln -sfn "proj_${PROJECTED_SCORE_DIM}/${PROJECTED_SCORE_VARIANT}/score_indices.npy" "${compat_dir}/score_indices.npy"
  echo "[compat] ${compat_dir}/scores.npy -> proj_${PROJECTED_SCORE_DIM}/${PROJECTED_SCORE_VARIANT}/scores.npy"
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
      bash scripts/00_sample_for_attribution.sh
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

run_one_task() {
  local i="$1"
  local slot="$2"
  local gpu="$3"
  local sample_mode score_mode query seed unprompted_flag range
  IFS='|' read -r sample_mode score_mode query seed unprompted_flag range <<<"${TASKS[$i]}"
  local score_file
  score_file="$(score_file_for_task "${score_mode}" "${query}" "${seed}" "${unprompted_flag}" "${range}")"
  local log="${LOG_ROOT}/task_${i}__projected_raw_next__${score_mode}__$(path_tag "${query}")__seed_${seed}__$(range_tag "${range}").log"
  if [[ -f "${score_file}" ]]; then
    echo "[projected-skip] task=${i} existing=${score_file}" >"${log}"
    return 0
  fi
  if [[ -f "$(dirname "${score_file}")/proj_${PROJECTED_SCORE_DIM}/${PROJECTED_SCORE_VARIANT}/scores.npy" ]]; then
    materialize_compat_scores "${score_file}" >"${log}" 2>&1
    return 0
  fi

  local query_env="${query}"
  if [[ "${unprompted_flag}" == "1" ]]; then
    query_env="unconditional"
  fi
  local sample_run_root algorithm_dir
  sample_run_root="$(ensure_sample "${sample_mode}" "${query_env}" "${seed}")"
  algorithm_dir="$(algorithm_dir_for_task "${unprompted_flag}" "${range}")"
  local query_component query_artifact
  if [[ "${unprompted_flag}" == "1" || "${score_mode}" == unprompted_* ]]; then
    query_component="unprompted"
  else
    query_component="query_$(path_tag "${query_env}")"
  fi
  query_artifact="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/${PROJECTED_ARTIFACT_DIR_NAME_VALUE}/${score_mode}/train_seed_${TRAIN_SEED}/${query_component}/initial_seed_${seed}/shared_query/proj_${PROJECTED_CACHE_DIM}/query_gradient_artifact.npz"
  if [[ ! -f "${query_artifact}" ]]; then
    echo "Missing precomputed query gradient artifact: ${query_artifact}" >"${log}"
    echo "Run 10.5 query-cache before 11, or set RUN_QUERY_STAGE=1 manually." >>"${log}"
    return 1
  fi

  echo "[worker ${slot}] task=${i} range=${range} sample_mode=${sample_mode} score_mode=${score_mode} query=${query} seed=${seed} gpu=${gpu} -> ${log}"
  run_gpu_slot "${slot}" env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    GPU_IDS="${gpu}" \
    JAX_NUM_DEVICES=1 \
    PYTHONUNBUFFERED=1 \
    ATTRIBUTION_TQDM_MININTERVAL="${ATTRIBUTION_TQDM_MININTERVAL:-1}" \
    ATTRIBUTION_TQDM_LEAVE="${ATTRIBUTION_TQDM_LEAVE:-1}" \
    SCORE_INDEX_RANGES="${range}" \
    ATTRIBUTION_RANGES="${range}" \
    PROJECTED_SCORE_INDEX_RANGES="${range}" \
    TRAIN_SCORE_INDEX_RANGES="${TRAIN_SCORE_INDEX_RANGES}" \
    TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE_VALUE}" \
    TRAJ_PARAMETER_SOURCE="${TRAJ_PARAMETER_SOURCE_VALUE}" \
    TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE}" \
    TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE}" \
    TRAJ_SAVE_QUERY_NORMALIZED_SCORES=0 \
    PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM}" \
    PROJECTED_DIMS="${PROJECTED_DIMS}" \
    PROJECTED_ARTIFACT_DIR_NAME="${PROJECTED_ARTIFACT_DIR_NAME_VALUE}" \
    INCLUDE_RAW=1 \
    RUN_TRAIN_STAGE=1 \
    RUN_QUERY_STAGE=0 \
    RUN_SCORE_SWEEP=1 \
    PROJECTED_TRAIN_PARALLEL_AXIS=score_index \
    SCORE_ALGORITHM_DIR="${algorithm_dir}" \
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
  materialize_compat_scores "${score_file}" >>"${log}" 2>&1
}

mapfile -t TASKS < <(task_lines)
total_tasks="${#TASKS[@]}"

echo "Stampede3 projected Traj-TracIn raw-parameter next-checkpoint attribution"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; objective=${TRAJ_QUERY_OBJECTIVE_VALUE}; parameter_source=${TRAJ_PARAMETER_SOURCE_VALUE}"
echo "total_tasks=${total_tasks}; slots=${ATTR_NUM_SLOTS}; gpu_per_node=${GPU_PER_NODE}; ranges=${TRAJ_RANGES_TEXT}; logs=${LOG_ROOT}"
echo "projected_cache_dim=${PROJECTED_CACHE_DIM}; projected_dims=${PROJECTED_DIMS}; compat_variant=proj_${PROJECTED_SCORE_DIM}/${PROJECTED_SCORE_VARIANT}"
echo "projected_artifact_dir_name=${PROJECTED_ARTIFACT_DIR_NAME_VALUE}"

pids=()
for ((slot = 0; slot < ATTR_NUM_SLOTS; slot++)); do
  gpu=$((slot % GPU_PER_NODE))
  worker_log="${LOG_ROOT}/worker_${slot}.log"
  echo "Launch worker slot=${slot} gpu=${gpu} -> ${worker_log}"
  (
    for ((i = slot; i < total_tasks; i += ATTR_NUM_SLOTS)); do
      run_one_task "${i}" "${slot}" "${gpu}"
    done
  ) >"${worker_log}" 2>&1 &
  pids+=("$!")
done

wait_all "${pids[@]}"

missing=0
for ((i = 0; i < total_tasks; i++)); do
  IFS='|' read -r _sample_mode score_mode query seed unprompted_flag range <<<"${TASKS[$i]}"
  score_file="$(score_file_for_task "${score_mode}" "${query}" "${seed}" "${unprompted_flag}" "${range}")"
  if [[ -f "${score_file}" ]]; then
    echo "Found projected TrajTracIn raw-next compat artifact for task=${i}: ${score_file}"
  else
    echo "Missing projected TrajTracIn raw-next compat artifact for task=${i}: ${score_file}" >&2
    missing=$((missing + 1))
  fi
done
if (( missing > 0 )); then
  echo "Projected job 11 missing ${missing} expected score artifacts." >&2
  exit 1
fi
echo "Projected job 11 complete."
