#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIFAR2_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REFINE_ROOT="$(cd "${CIFAR2_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${REFINE_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment_67}"
TRAIN_SEED="${TRAIN_SEED:-67}"
QUERY="${QUERY:-horse}"
INITIAL_SEED="${INITIAL_SEED:-${SAMPLE_SEED:-0}}"
ATTRIBUTION_SCORE_MODEL_MODE="${ATTRIBUTION_SCORE_MODEL_MODE:-${SAMPLE_MODEL_MODE:-prompted_solo}}"
SAMPLE_MODEL_MODE="${SAMPLE_MODEL_MODE:-${ATTRIBUTION_SCORE_MODEL_MODE}}"
SCORE_INDEX_RANGES="${PROJECTED_SCORE_INDEX_RANGES:-1-10000}"
TRAIN_SCORE_INDEX_RANGES="${TRAIN_SCORE_INDEX_RANGES:-1-10000}"
PROJECTED_TRAIN_SHARD_RANGES="${PROJECTED_TRAIN_SHARD_RANGES:-}"
TRAIN_SHARD_INDEX="${TRAIN_SHARD_INDEX:-}"
PROJECTED_DIMS="${PROJECTED_DIMS:-4096}"
PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
NORMALIZE_EPS="${NORMALIZE_EPS:-${TRAJ_QUERY_NORMALIZE_EPS:-1e-8}}"
RUN_TRAIN_STAGE="${RUN_TRAIN_STAGE:-1}"
RUN_QUERY_STAGE="${RUN_QUERY_STAGE:-1}"
RUN_SCORE_SWEEP="${RUN_SCORE_SWEEP:-1}"
INCLUDE_RAW="${INCLUDE_RAW:-1}"
SKIP_EXISTING_SCORES="${SKIP_EXISTING_SCORES:-1}"
SHARE_TRAIN_ARTIFACT="${SHARE_TRAIN_ARTIFACT:-1}"
SHARE_QUERY_ARTIFACT="${SHARE_QUERY_ARTIFACT:-1}"
LOCK_POLL_SECONDS="${LOCK_POLL_SECONDS:-60}"
GPU_IDS="${GPU_IDS:-0}"

export REPO_ROOT CIFAR2_ROOT REFINE_ROOT EXPERIMENT_TAG TRAIN_SEED QUERY INITIAL_SEED SAMPLE_MODEL_MODE
export ATTRIBUTION_SCORE_MODEL_MODE SCORE_INDEX_RANGES TRAIN_SCORE_INDEX_RANGES PROJECTED_TRAIN_SHARD_RANGES TRAIN_SHARD_INDEX
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
export TRAJ_TRACIN_PROJ_DIM="${PROJECTED_CACHE_DIM}"

path_tag() {
  local value="$1"
  value="${value//,/__}"
  value="${value//+/_}"
  value="${value//[^A-Za-z0-9._-]/_}"
  while [[ "${value}" == *"__"* ]]; do value="${value//__/_}"; done
  value="${value#_}"
  value="${value%_}"
  printf '%s' "${value:-unprompted}"
}

range_tag() {
  local value="$1"
  value="${value//,/__}"
  value="${value//:/-}"
  value="${value//-/_}"
  path_tag "${value}"
}

canonical_train_mode() {
  local mode="$1"
  case "${mode}" in
    prompted_multi|prompted_jax|prompted)
      printf '%s' "prompted_solo"
      ;;
    unprompted_multi|unprompted_jax|unprompted)
      printf '%s' "unprompted_solo"
      ;;
    *)
      printf '%s' "${mode}"
      ;;
  esac
}

is_unprompted_mode() {
  [[ "${ATTRIBUTION_SCORE_MODEL_MODE}" == unprompted_* || "${UNPROMPTED:-0}" =~ ^(1|true|True|yes)$ ]]
}

query_component() {
  if is_unprompted_mode; then
    printf '%s' "unprompted"
  else
    printf 'query_%s' "$(path_tag "${QUERY}")"
  fi
}

default_sample_dir() {
  if is_unprompted_mode; then
    printf '%s/result/%s/sample/cifar/prompt_unconditional/model_%s__ckpt_seed_%s_epoch_%04d' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${SAMPLE_MODEL_MODE}" "${TRAIN_SEED}" "${JAX_EPOCHS:-200}"
  else
    printf '%s/result/%s/sample/cifar/prompt_%s/model_%s__ckpt_seed_%s_epoch_%04d' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "$(path_tag "${QUERY}")" "${SAMPLE_MODEL_MODE}" "${TRAIN_SEED}" "${JAX_EPOCHS:-200}"
  fi
}

run_original_config_with_stage() {
  local stage_mode="$1"
  local artifact_path="$2"
  (
    cd "${REFINE_ROOT}/legacy_jax"
    export TRAJ_TRACIN_STAGE_MODE="${stage_mode}"
    export TRAJ_TRACIN_STAGE_ARTIFACT_PATH="${artifact_path}"
    export DATAPOINT_MODEL_MODE="${DATAPOINT_MODEL_MODE:-}"
    export ATTRIBUTION_SAMPLE_DIR="${ATTRIBUTION_SAMPLE_DIR:-}"
    export SAMPLE_SEED="${SAMPLE_SEED:-${INITIAL_SEED}}"
    if is_unprompted_mode; then
      export UNPROMPTED=1
    fi
    TRAJ_TRACIN_STAGE_MODE="${stage_mode}" \
    TRAJ_TRACIN_STAGE_ARTIFACT_PATH="${artifact_path}" \
    "${PYTHON_BIN}" "${REFINE_ROOT}/common/run_original_attribution_config.py" \
      "${CIFAR2_ROOT}/data_attribution/traj_tracin/CONFIG.py"
  )
}

run_train_stage_with_lock() {
  local artifact_path="$1"
  local lock_dir="${artifact_path}.lock"

  if [[ -f "${artifact_path}" ]]; then
    echo "[stage train] found existing projected train artifact: ${artifact_path}"
    return
  fi

  if mkdir "${lock_dir}" 2>/dev/null; then
    trap 'rm -rf "${lock_dir}"' RETURN
    if [[ -f "${artifact_path}" ]]; then
      echo "[stage train] artifact appeared after lock acquisition: ${artifact_path}"
      return
    fi
    echo "[stage train] acquired lock: ${lock_dir}"
    DATAPOINT_MODEL_MODE="${TRAIN_MODE_CANONICAL}" \
    ATTRIBUTION_SAMPLE_DIR="${SAMPLE_DIR}" \
    TRAJ_USE_SAVED_TRAJECTORY=0 \
    SCORE_INDEX_RANGES="${TRAIN_COMPUTE_RANGE}" \
    run_original_config_with_stage train "${artifact_path}"
    rm -rf "${lock_dir}"
    trap - RETURN
    return
  fi

  echo "[stage train] another worker owns ${lock_dir}; waiting for ${artifact_path}"
  while [[ ! -f "${artifact_path}" ]]; do
    sleep "${LOCK_POLL_SECONDS}"
  done
  echo "[stage train] shared train artifact is ready: ${artifact_path}"
}

split_words() {
  local text="$1"
  text="${text//,/ }"
  printf '%s\n' ${text}
}

score_outputs_complete() {
  local dim variant
  local variants=(query_l2_normalized train_l2_normalized query_train_l2_normalized)
  if [[ "${INCLUDE_RAW}" == "1" ]]; then
    variants=(raw "${variants[@]}")
  fi
  while IFS= read -r dim; do
    for variant in "${variants[@]}"; do
      [[ -f "${SCORE_OUT_ROOT}/proj_${dim}/${variant}/scores.npy" ]] || return 1
      [[ -f "${SCORE_OUT_ROOT}/proj_${dim}/${variant}/score_indices.npy" ]] || return 1
    done
  done < <(split_words "${PROJECTED_DIMS}")
  return 0
}

selected_train_range() {
  if [[ -z "${PROJECTED_TRAIN_SHARD_RANGES}" ]]; then
    printf '%s' "${TRAIN_SCORE_INDEX_RANGES}"
    return
  fi
  if [[ -z "${TRAIN_SHARD_INDEX}" ]]; then
    echo "TRAIN_SHARD_INDEX is required when PROJECTED_TRAIN_SHARD_RANGES is set." >&2
    exit 2
  fi
  shard_ranges=()
  while IFS= read -r shard_range; do
    shard_ranges+=("${shard_range}")
  done < <(split_words "${PROJECTED_TRAIN_SHARD_RANGES}")
  local shard_index="$((10#${TRAIN_SHARD_INDEX}))"
  if (( shard_index < 0 || shard_index >= ${#shard_ranges[@]} )); then
    echo "TRAIN_SHARD_INDEX=${TRAIN_SHARD_INDEX} out of range for ${#shard_ranges[@]} shard ranges." >&2
    exit 2
  fi
  printf '%s' "${shard_ranges[${shard_index}]}"
}

TRAIN_MODE_CANONICAL="$(canonical_train_mode "${ATTRIBUTION_SCORE_MODEL_MODE}")"
SAMPLE_DIR="${ATTRIBUTION_SAMPLE_DIR:-$(default_sample_dir)}"
QCOMP="$(query_component)"
RTAG="$(range_tag "${SCORE_INDEX_RANGES}")"
TRAIN_RTAG="$(range_tag "${TRAIN_SCORE_INDEX_RANGES}")"
TRAIN_COMPUTE_RANGE="$(selected_train_range)"
TRAIN_COMPUTE_RTAG="$(range_tag "${TRAIN_COMPUTE_RANGE}")"
if [[ "${SCORE_INDEX_RANGES}" == "${TRAIN_SCORE_INDEX_RANGES}" ]]; then
  SCORE_ALGORITHM_DIR="${SCORE_ALGORITHM_DIR:-traj_tracin_projected}"
else
  SCORE_ALGORITHM_DIR="${SCORE_ALGORITHM_DIR:-traj_tracin_projected_range_${RTAG}}"
fi

PROJECTED_ARTIFACT_DIR_NAME="${PROJECTED_ARTIFACT_DIR_NAME:-projected_traj_tracin_artifacts}"
ARTIFACT_BASE="${PROJECTED_ARTIFACT_BASE:-${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/${PROJECTED_ARTIFACT_DIR_NAME}/${ATTRIBUTION_SCORE_MODEL_MODE}/train_seed_${TRAIN_SEED}}"
if [[ "${SHARE_TRAIN_ARTIFACT}" == "1" ]]; then
  TRAIN_ARTIFACT_ROOT="${PROJECTED_TRAIN_ARTIFACT_ROOT:-${ARTIFACT_BASE}/shared_train/range_${TRAIN_RTAG}/proj_${PROJECTED_CACHE_DIM}}"
else
  TRAIN_ARTIFACT_ROOT="${PROJECTED_TRAIN_ARTIFACT_ROOT:-${ARTIFACT_BASE}/${QCOMP}/initial_seed_${INITIAL_SEED}/range_${TRAIN_RTAG}/proj_${PROJECTED_CACHE_DIM}}"
fi
if [[ "${SHARE_QUERY_ARTIFACT}" == "1" ]]; then
  QUERY_ARTIFACT_ROOT="${PROJECTED_QUERY_ARTIFACT_ROOT:-${ARTIFACT_BASE}/${QCOMP}/initial_seed_${INITIAL_SEED}/shared_query/proj_${PROJECTED_CACHE_DIM}}"
else
  QUERY_ARTIFACT_ROOT="${PROJECTED_QUERY_ARTIFACT_ROOT:-${ARTIFACT_BASE}/${QCOMP}/initial_seed_${INITIAL_SEED}/range_${RTAG}/proj_${PROJECTED_CACHE_DIM}}"
fi
TRAIN_ARTIFACT="${TRAIN_ARTIFACT:-${TRAIN_ARTIFACT_ROOT}/train_datapoint_gradient_artifact.npz}"
if [[ -n "${PROJECTED_TRAIN_SHARD_RANGES}" ]]; then
  TRAIN_SHARD_ARTIFACT="${TRAIN_SHARD_ARTIFACT:-${TRAIN_ARTIFACT_ROOT}/shards/range_${TRAIN_COMPUTE_RTAG}/train_datapoint_gradient_artifact.npz}"
else
  TRAIN_SHARD_ARTIFACT="${TRAIN_SHARD_ARTIFACT:-${TRAIN_ARTIFACT}}"
fi
QUERY_ARTIFACT="${QUERY_ARTIFACT:-${QUERY_ARTIFACT_ROOT}/query_gradient_artifact.npz}"
SCORE_OUT_ROOT="${SCORE_OUT_ROOT:-${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/attribution_score/${ATTRIBUTION_SCORE_MODEL_MODE}/train_seed_${TRAIN_SEED}/${QCOMP}/initial_seed_${INITIAL_SEED}/${SCORE_ALGORITHM_DIR}}"

mkdir -p "${TRAIN_ARTIFACT_ROOT}" "$(dirname "${TRAIN_SHARD_ARTIFACT}")" "${QUERY_ARTIFACT_ROOT}" "${SCORE_OUT_ROOT}"

echo "Projected Traj-TracIn score sweep"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; score_mode=${ATTRIBUTION_SCORE_MODEL_MODE}; query=${QUERY}; initial_seed=${INITIAL_SEED}"
echo "range=${SCORE_INDEX_RANGES}; cache_dim=${PROJECTED_CACHE_DIM}; dims=${PROJECTED_DIMS}; normalize_eps=${NORMALIZE_EPS}"
echo "train_range=${TRAIN_SCORE_INDEX_RANGES}"
echo "train_compute_range=${TRAIN_COMPUTE_RANGE}; train_shard_index=${TRAIN_SHARD_INDEX:-none}"
echo "share_train_artifact=${SHARE_TRAIN_ARTIFACT}"
echo "share_query_artifact=${SHARE_QUERY_ARTIFACT}"
echo "sample_dir=${SAMPLE_DIR}"
echo "train_artifact=${TRAIN_ARTIFACT}"
echo "train_shard_artifact=${TRAIN_SHARD_ARTIFACT}"
echo "query_artifact=${QUERY_ARTIFACT}"
echo "score_out_root=${SCORE_OUT_ROOT}"
echo "gpus=${CUDA_VISIBLE_DEVICES}"

if [[ "${RUN_TRAIN_STAGE}" == "1" && -n "${PROJECTED_TRAIN_SHARD_RANGES}" && ! -f "${TRAIN_SHARD_ARTIFACT}" ]]; then
  echo "[stage train] writing projected train shard artifact"
  run_train_stage_with_lock "${TRAIN_SHARD_ARTIFACT}"
elif [[ "${RUN_TRAIN_STAGE}" == "1" && -z "${PROJECTED_TRAIN_SHARD_RANGES}" && ! -f "${TRAIN_ARTIFACT}" ]]; then
  echo "[stage train] writing projected train artifact"
  run_train_stage_with_lock "${TRAIN_ARTIFACT}"
else
  echo "[stage train] skipped; RUN_TRAIN_STAGE=${RUN_TRAIN_STAGE}, full_exists=$([[ -f "${TRAIN_ARTIFACT}" ]] && echo 1 || echo 0), shard_exists=$([[ -f "${TRAIN_SHARD_ARTIFACT}" ]] && echo 1 || echo 0)"
fi

if [[ "${RUN_QUERY_STAGE}" == "1" && ! -f "${QUERY_ARTIFACT}" ]]; then
  echo "[stage query] writing projected query artifact"
  ATTRIBUTION_SAMPLE_DIR="${SAMPLE_DIR}" \
  SAMPLE_SEED="${INITIAL_SEED}" \
  run_original_config_with_stage query "${QUERY_ARTIFACT}"
else
  echo "[stage query] skipped; RUN_QUERY_STAGE=${RUN_QUERY_STAGE}, exists=$([[ -f "${QUERY_ARTIFACT}" ]] && echo 1 || echo 0)"
fi

if [[ "${RUN_SCORE_SWEEP}" == "1" ]]; then
  if [[ "${SKIP_EXISTING_SCORES}" == "1" ]] && score_outputs_complete; then
    echo "[score sweep] skipped; all projected score outputs already exist under ${SCORE_OUT_ROOT}"
    exit 0
  fi
  if [[ ! -f "${TRAIN_ARTIFACT}" ]]; then
    echo "Missing merged train artifact for score sweep: ${TRAIN_ARTIFACT}" >&2
    echo "Run projected_traj_tracin_train_cache_array_h100.sh first, or disable shard mode." >&2
    exit 1
  fi
  raw_arg=()
  if [[ "${INCLUDE_RAW}" != "1" ]]; then
    raw_arg+=(--no-raw)
  fi
  echo "[score sweep] combining projected artifacts"
  "${PYTHON_BIN}" "${REFINE_ROOT}/common/projected_traj_tracin_score_sweep.py" \
    --train-artifact "${TRAIN_ARTIFACT}" \
    --query-artifact "${QUERY_ARTIFACT}" \
    --out-root "${SCORE_OUT_ROOT}" \
    --proj-dims "${PROJECTED_DIMS}" \
    --normalize-eps "${NORMALIZE_EPS}" \
    --score-index-ranges "${SCORE_INDEX_RANGES}" \
    --score-index-base 1 \
    "${raw_arg[@]}"
else
  echo "[score sweep] skipped; RUN_SCORE_SWEEP=${RUN_SCORE_SWEEP}"
fi

echo "Projected Traj-TracIn score sweep complete."
