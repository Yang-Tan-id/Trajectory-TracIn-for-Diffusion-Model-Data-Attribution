#!/usr/bin/env bash
#SBATCH -J x3-s3-traj-one-rtx
#SBATCH -o x3-s3-traj-one-rtx-%j.out
#SBATCH -e x3-s3-traj-one-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-3%4

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/x3/stampede3_das"; do
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

TRAJ_SAMPLE_MODE="${TRAJ_SAMPLE_MODE:-prompted_solo}"
TRAJ_SCORE_MODE="${TRAJ_SCORE_MODE:-prompted_solo}"
TRAJ_QUERY="${TRAJ_QUERY:-horse}"
TRAJ_QUERY_ENV="${TRAJ_QUERY}"
TRAJ_INITIAL_SEED="${TRAJ_INITIAL_SEED:-3}"
TRAJ_UNPROMPTED="${TRAJ_UNPROMPTED:-0}"
if [[ "${TRAJ_UNPROMPTED}" == "1" ]]; then
  TRAJ_SAMPLE_MODE="${TRAJ_SAMPLE_MODE:-unprompted_solo}"
  TRAJ_SCORE_MODE="${TRAJ_SCORE_MODE:-unprompted_solo}"
  TRAJ_QUERY="unprompted"
  TRAJ_QUERY_ENV="unconditional"
fi

JAX_EPOCHS="${JAX_EPOCHS:-200}"
TRAJ_RANGES_TEXT="${TRAJ_RANGES:-1-2500 2501-5000 5001-7500 7501-10000}"
TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-4}"
TRAJ_SAVE_QUERY_NORMALIZED_SCORES="${TRAJ_SAVE_QUERY_NORMALIZED_SCORES:-1}"
TRAJ_QUERY_NORMALIZE_EPS="${TRAJ_QUERY_NORMALIZE_EPS:-1e-8}"
LOG_ROOT="${X3_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/02_traj_tracin_one_query_rtx_small/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

prompt_tag="$(path_tag "${TRAJ_QUERY_ENV}")"
printf -v seed_tag "%06d" "${TRAJ_INITIAL_SEED}"
printf -v ckpt_stem "seed_%s_epoch_%04d" "${TRAIN_SEED}" "${JAX_EPOCHS}"
sample_run_root="${X3_ROOT}/result/${EXPERIMENT_TAG}/sample/x3/prompt_${prompt_tag}/model_${TRAJ_SAMPLE_MODE}__ckpt_${ckpt_stem}"
mkdir -p "${sample_run_root}"
sample_done_file="${sample_run_root}/seed_${seed_tag}/trajectory_xt.npy"
sample_lock_dir="${sample_run_root}/.sample_seed_${seed_tag}.lock"

range_tag() {
  local value="$1"
  value="${value//:/-}"
  local start="${value%-*}"
  local end="${value#*-}"
  printf 'range_%s_%s' "${start}" "${end}"
}

score_file_for_range() {
  local range="$1"
  local variant="${2:-raw}"
  local tag
  tag="$(range_tag "${range}")"
  if [[ "${TRAJ_UNPROMPTED}" == "1" || "${TRAJ_SCORE_MODE}" == unprompted_* ]]; then
    if [[ "${variant}" == "normalized" ]]; then
      printf '%s/result/%s/attribution_score/%s/train_seed_%s/unprompted/initial_seed_%s/traj_tracin_normalized_unprompted_%s/scores.npy' \
        "${X3_ROOT}" "${EXPERIMENT_TAG}" "${TRAJ_SCORE_MODE}" "${TRAIN_SEED}" "${TRAJ_INITIAL_SEED}" "${tag}"
    else
      printf '%s/result/%s/attribution_score/%s/train_seed_%s/unprompted/initial_seed_%s/traj_tracin_unprompted_%s/scores.npy' \
        "${X3_ROOT}" "${EXPERIMENT_TAG}" "${TRAJ_SCORE_MODE}" "${TRAIN_SEED}" "${TRAJ_INITIAL_SEED}" "${tag}"
    fi
  else
    if [[ "${variant}" == "normalized" ]]; then
      printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/traj_tracin_normalized_%s/scores.npy' \
        "${X3_ROOT}" "${EXPERIMENT_TAG}" "${TRAJ_SCORE_MODE}" "${TRAIN_SEED}" "$(path_tag "${TRAJ_QUERY}")" "${TRAJ_INITIAL_SEED}" "${tag}"
    else
      printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/traj_tracin_%s/scores.npy' \
        "${X3_ROOT}" "${EXPERIMENT_TAG}" "${TRAJ_SCORE_MODE}" "${TRAIN_SEED}" "$(path_tag "${TRAJ_QUERY}")" "${TRAJ_INITIAL_SEED}" "${tag}"
    fi
  fi
}

run_range() {
  local slot="$1"
  local gpu="$2"
  local range="$3"
  local score_file normalized_score_file log
  score_file="$(score_file_for_range "${range}")"
  normalized_score_file="$(score_file_for_range "${range}" normalized)"
  log="${LOG_ROOT}/traj_tracin__${TRAJ_SCORE_MODE}__$(path_tag "${TRAJ_QUERY}")__seed_${TRAJ_INITIAL_SEED}__$(range_tag "${range}").log"
  if [[ -f "${score_file}" && ( "${TRAJ_SAVE_QUERY_NORMALIZED_SCORES}" != "1" || -f "${normalized_score_file}" ) ]]; then
    {
      echo "[traj-skip] range=${range} existing_raw=${score_file}"
      echo "[traj-skip] range=${range} existing_query_normalized=${normalized_score_file}"
    } >"${log}"
    return 0
  fi
  echo "[traj-run] range=${range} gpu=${gpu} log=${log}"
  run_gpu_slot "${slot}" env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    GPU_IDS="${gpu}" \
    JAX_NUM_DEVICES=1 \
    PYTHONUNBUFFERED=1 \
    ATTRIBUTION_TQDM_MININTERVAL="${ATTRIBUTION_TQDM_MININTERVAL:-1}" \
    ATTRIBUTION_TQDM_LEAVE="${ATTRIBUTION_TQDM_LEAVE:-1}" \
    SCORE_INDEX_RANGES="${range}" \
    ATTRIBUTION_RANGES="${range}" \
    TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE}" \
    TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE}" \
    TRAJ_SAVE_QUERY_NORMALIZED_SCORES="${TRAJ_SAVE_QUERY_NORMALIZED_SCORES}" \
    TRAJ_QUERY_NORMALIZE_EPS="${TRAJ_QUERY_NORMALIZE_EPS}" \
    X3_ROOT="${X3_ROOT}" \
    REFINE_ROOT="${REFINE_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
    TRAIN_SEED="${TRAIN_SEED}" \
    SAMPLE_MODEL_MODE="${TRAJ_SAMPLE_MODE}" \
    UNPROMPTED_SAMPLE_MODEL_MODE="${TRAJ_SAMPLE_MODE}" \
    ATTRIBUTION_SAMPLE_MODEL_MODE="${TRAJ_SAMPLE_MODE}" \
    ATTRIBUTION_SCORE_MODEL_MODE="${TRAJ_SCORE_MODE}" \
    UNPROMPTED_SCORE_MODEL_MODE="${TRAJ_SCORE_MODE}" \
    QUERY="${TRAJ_QUERY_ENV}" \
    INITIAL_SEED="${TRAJ_INITIAL_SEED}" \
    SAMPLE_SEED="${TRAJ_INITIAL_SEED}" \
    SAMPLE_SEEDS="${TRAJ_INITIAL_SEED}" \
    SAMPLE_DONE_FILE="${sample_done_file}" \
    SAMPLE_LOCK_DIR="${sample_lock_dir}" \
    SAMPLE_LOCK_WAIT_SECONDS="${SAMPLE_LOCK_WAIT_SECONDS:-21600}" \
    UNPROMPTED="${TRAJ_UNPROMPTED}" \
    ALGORITHM=traj_tracin \
    bash "${SCRIPT_DIR}/02_dtrak_endtracin_attribution_task_stampede3.sh" \
    >"${log}" 2>&1
}

mapfile -t ranges < <(printf '%s\n' ${TRAJ_RANGES_TEXT})

echo "Job 02 RTX-small TrajTracIn one query"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; sample_mode=${TRAJ_SAMPLE_MODE}; score_mode=${TRAJ_SCORE_MODE}; query=${TRAJ_QUERY}; seed=${TRAJ_INITIAL_SEED}"
echo "ranges=${TRAJ_RANGES_TEXT}; array_task=${SLURM_ARRAY_TASK_ID:-none}; score_batch_size=${TRAJ_SCORE_BATCH_SIZE}; snapshot_chunk_size=${TRAJ_SNAPSHOT_CHUNK_SIZE}; save_query_normalized=${TRAJ_SAVE_QUERY_NORMALIZED_SCORES}; logs=${LOG_ROOT}"

pids=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  range_index="${TRAJ_RANGE_INDEX:-${SLURM_ARRAY_TASK_ID}}"
  if (( range_index < 0 || range_index >= ${#ranges[@]} )); then
    echo "TRAJ range index ${range_index} is outside [0, ${#ranges[@]})." >&2
    exit 1
  fi
  run_range 0 0 "${ranges[$range_index]}"
else
  for i in "${!ranges[@]}"; do
    run_range 0 0 "${ranges[$i]}"
  done
fi

missing=0
check_ranges=("${ranges[@]}")
if [[ -n "${range_index:-}" ]]; then
  check_ranges=("${ranges[$range_index]}")
fi
for range in "${check_ranges[@]}"; do
  score_file="$(score_file_for_range "${range}")"
  normalized_score_file="$(score_file_for_range "${range}" normalized)"
  if [[ -f "${score_file}" ]]; then
    echo "Found TrajTracIn range artifact: ${score_file}"
  else
    echo "Missing TrajTracIn range artifact: ${score_file}" >&2
    missing=$((missing + 1))
  fi
  if [[ "${TRAJ_SAVE_QUERY_NORMALIZED_SCORES}" == "1" ]]; then
    if [[ -f "${normalized_score_file}" ]]; then
      echo "Found TrajTracIn query-normalized range artifact: ${normalized_score_file}"
    else
      echo "Missing TrajTracIn query-normalized range artifact: ${normalized_score_file}" >&2
      missing=$((missing + 1))
    fi
  fi
done
if (( missing > 0 )); then
  echo "TrajTracIn one-query RTX-small job missing ${missing} range artifacts." >&2
  exit 1
fi

echo "TrajTracIn one-query RTX-small complete."
