#!/usr/bin/env bash
#SBATCH -J cifar2-proj-traj-stream
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH -o cifar2-proj-traj-stream-%j.out
#SBATCH -e cifar2-proj-traj-stream-%j.err

set -euo pipefail

SCRIPT_DIR_CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR_CANDIDATE}/projected_traj_tracin_score_sweep.sh" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_CANDIDATE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/cifar2/tacc/h100/projected_traj_tracin_score_sweep.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/cifar2/tacc/h100" && pwd)"
else
  echo "Could not locate projected_traj_tracin_score_sweep.sh from ${SCRIPT_DIR_CANDIDATE} or SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}." >&2
  exit 2
fi
CIFAR2_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REFINE_ROOT="$(cd "${CIFAR2_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${REFINE_ROOT}/.." && pwd)"

source "${CIFAR2_ROOT}/stampede3_das/_stampede3_das_lib.sh"
stampede3_das_init

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_SLOTS="${GPU_SLOTS:-16}"
GPU_PER_NODE="${GPU_PER_NODE:-4}"
PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-1024}"
STREAM_PROJ_DIM="${STREAM_PROJ_DIM:-${PROJECTED_CACHE_DIM}}"
STREAM_PROJ_DIMS="${STREAM_PROJ_DIMS:-${STREAM_PROJ_DIM}}"
STREAM_SCORE_RANGES="${STREAM_SCORE_RANGES:-1-625 626-1250 1251-1875 1876-2500 2501-3125 3126-3750 3751-4375 4376-5000 5001-5625 5626-6250 6251-6875 6876-7500 7501-8125 8126-8750 8751-9375 9376-10000}"
STREAM_TASK_SET="${STREAM_TASK_SET:-all}"
TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
TRAJ_QUERY_NORMALIZE_EPS="${TRAJ_QUERY_NORMALIZE_EPS:-1e-8}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHON_BIN PROJECTED_CACHE_DIM STREAM_PROJ_DIM STREAM_PROJ_DIMS TRAJ_SCORE_BATCH_SIZE TRAJ_QUERY_NORMALIZE_EPS

split_words() {
  local text="$1"
  text="${text//,/ }"
  printf '%s\n' ${text}
}

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

score_modes_for_task_set() {
  case "${STREAM_TASK_SET}" in
    prompted|prompted_solo)
      printf '%s\n' prompted_solo
      ;;
    unprompted|unprompted_solo)
      printf '%s\n' unprompted_solo
      ;;
    all)
      printf '%s\n' prompted_solo
      printf '%s\n' unprompted_solo
      ;;
    *)
      printf '%s\n' "${STREAM_TASK_SET}"
      ;;
  esac
}

query_artifacts_for_score_mode() {
  local score_mode="$1"
  local artifact_base="${PROJECTED_ARTIFACT_BASE:-${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/projected_traj_tracin_artifacts/${score_mode}/train_seed_${TRAIN_SEED}}"
  [[ -d "${artifact_base}" ]] || return 0
  find "${artifact_base}" \
    -path "*/shared_query/proj_${PROJECTED_CACHE_DIM}/query_gradient_artifact.npz" \
    -type f \
    | sort
}

run_stream_shard() {
  local slot="$1"
  local score_mode="$2"
  local range="$3"
  local query_paths="$4"
  local gpu="$((slot % GPU_PER_NODE))"
  local unprompted_flag=0
  [[ "${score_mode}" == unprompted_* ]] && unprompted_flag=1

  local artifact_base="${PROJECTED_ARTIFACT_BASE:-${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/projected_traj_tracin_artifacts/${score_mode}/train_seed_${TRAIN_SEED}}"
  local dims_tag
  dims_tag="$(path_tag "${STREAM_PROJ_DIMS}")"
  local stream_root="${STREAM_SCORE_ROOT:-${artifact_base}/stream_scores/cache_${PROJECTED_CACHE_DIM}/proj_${dims_tag}}"
  local rtag
  rtag="$(range_tag "${range}")"
  local out_path="${stream_root}/shards/range_${rtag}/stream_scores.npz"
  mkdir -p "$(dirname "${out_path}")"

  if [[ -f "${out_path}" ]]; then
    echo "[skip] score_mode=${score_mode} range=${range} existing ${out_path}"
    return
  fi

  echo "[stream] score_mode=${score_mode} range=${range} gpu=${gpu} -> ${out_path}"
  local cmd=(
    env
    CUDA_VISIBLE_DEVICES="${gpu}"
    ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}"
    SAMPLE_MODEL_MODE="${score_mode}"
    UNPROMPTED="${unprompted_flag}"
    QUERY="$([[ "${unprompted_flag}" == "1" ]] && printf 'unconditional' || printf 'horse')"
    TRAJ_USE_SAVED_TRAJECTORY=0
    SCORE_INDEX_RANGES="${range}"
    ATTRIBUTION_RANGES="${range}"
    TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE}"
    TRAJ_QUERY_NORMALIZE_EPS="${TRAJ_QUERY_NORMALIZE_EPS}"
    TRAJ_TRACIN_STAGE_MODE=score_stream
    TRAJ_TRACIN_STAGE_ARTIFACT_PATH="${out_path}"
    TRAJ_TRACIN_STREAM_CACHE_DIM="${PROJECTED_CACHE_DIM}"
    TRAJ_TRACIN_STREAM_PROJ_DIMS="${STREAM_PROJ_DIMS}"
    TRAJ_TRACIN_STREAM_QUERY_ARTIFACTS="${query_paths}"
    "${PYTHON_BIN}" "${REFINE_ROOT}/common/run_original_attribution_config.py"
    "${CIFAR2_ROOT}/data_attribution/traj_tracin/CONFIG.py"
  )
  if [[ "${TACC_SLOT_BACKEND:-ibrun}" == "ibrun" && -n "${SLURM_JOB_ID:-}" ]] && command -v ibrun >/dev/null; then
    ibrun -n 1 -o "${slot}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

run_one_score_mode() {
  local score_mode="$1"
  mapfile -t query_paths_array < <(query_artifacts_for_score_mode "${score_mode}")
  if (( ${#query_paths_array[@]} == 0 )); then
    echo "No query artifacts found for score_mode=${score_mode}, proj=${PROJECTED_CACHE_DIM}." >&2
    echo "Run projected_traj_tracin_query_score_array_h100.sh with RUN_SCORE_SWEEP=0 first." >&2
    exit 1
  fi
  local query_paths
  query_paths="$(IFS=:; printf '%s' "${query_paths_array[*]}")"

  echo "Projected Traj-TracIn query-cached stream scoring"
  echo "score_mode=${score_mode}; cache_dim=${PROJECTED_CACHE_DIM}; proj_dims=${STREAM_PROJ_DIMS}; queries=${#query_paths_array[@]}"
  echo "ranges=${STREAM_SCORE_RANGES}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s\n' "${query_paths_array[@]}"
    return
  fi

  pids=()
  for slot in "${!RANGES[@]}"; do
    run_stream_shard "${slot}" "${score_mode}" "${RANGES[$slot]}" "${query_paths}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if (( failed != 0 )); then
    echo "At least one stream shard failed for score_mode=${score_mode}." >&2
    exit 1
  fi

  artifact_base="${PROJECTED_ARTIFACT_BASE:-${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/projected_traj_tracin_artifacts/${score_mode}/train_seed_${TRAIN_SEED}}"
  dims_tag="$(path_tag "${STREAM_PROJ_DIMS}")"
  stream_root="${STREAM_SCORE_ROOT:-${artifact_base}/stream_scores/cache_${PROJECTED_CACHE_DIM}/proj_${dims_tag}}"
  shard_paths=()
  for range in "${RANGES[@]}"; do
    rtag="$(range_tag "${range}")"
    shard_paths+=("${stream_root}/shards/range_${rtag}/stream_scores.npz")
  done
  "${PYTHON_BIN}" "${REFINE_ROOT}/common/merge_stream_score_shards.py" \
    --output "${stream_root}/stream_scores_merged.npz" \
    "${shard_paths[@]}"
}

RANGES=()
while IFS= read -r range; do
  RANGES+=("${range}")
done < <(split_words "${STREAM_SCORE_RANGES}")
if (( ${#RANGES[@]} > GPU_SLOTS )); then
  echo "This helper expects at most GPU_SLOTS=${GPU_SLOTS} ranges, got ${#RANGES[@]}." >&2
  exit 2
fi

while IFS= read -r score_mode; do
  run_one_score_mode "${score_mode}"
done < <(score_modes_for_task_set)
