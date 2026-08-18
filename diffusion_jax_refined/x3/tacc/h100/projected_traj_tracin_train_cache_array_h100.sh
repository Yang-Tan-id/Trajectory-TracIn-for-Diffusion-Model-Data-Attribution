#!/usr/bin/env bash
#SBATCH -J x3-proj-traj-train
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH -o x3-proj-traj-train-%j.out
#SBATCH -e x3-proj-traj-train-%j.err

set -euo pipefail

SCRIPT_DIR_CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR_CANDIDATE}/projected_traj_tracin_score_sweep.sh" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_CANDIDATE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/x3/tacc/h100/projected_traj_tracin_score_sweep.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/x3/tacc/h100" && pwd)"
else
  echo "Could not locate projected_traj_tracin_score_sweep.sh from ${SCRIPT_DIR_CANDIDATE} or SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}." >&2
  exit 2
fi
X3_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REFINE_ROOT="$(cd "${X3_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${REFINE_ROOT}/.." && pwd)"

# Reuse the same Stampede3 runtime setup as the existing X3 DAS/Traj jobs.
# This activates ${SCRATCH}/conda-envs/trajectory-tracin unless ENV_SETUP or
# CONDA_ENV_PATH overrides it.
source "${X3_ROOT}/stampede3_das/_stampede3_das_lib.sh"
stampede3_das_init

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_SLOTS="${GPU_SLOTS:-16}"
GPU_PER_NODE="${GPU_PER_NODE:-4}"
PROJECTED_TRAIN_SHARD_RANGES="${PROJECTED_TRAIN_SHARD_RANGES:-1-625 626-1250 1251-1875 1876-2500 2501-3125 3126-3750 3751-4375 4376-5000 5001-5625 5626-6250 6251-6875 6876-7500 7501-8125 8126-8750 8751-9375 9376-10000}"
PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-32768}"
TRAIN_SCORE_INDEX_RANGES="${TRAIN_SCORE_INDEX_RANGES:-1-10000}"
TRAIN_CACHE_TASK_SET="${TRAIN_CACHE_TASK_SET:-all}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHON_BIN PROJECTED_TRAIN_SHARD_RANGES PROJECTED_CACHE_DIM TRAIN_SCORE_INDEX_RANGES
export RUN_QUERY_STAGE="${RUN_QUERY_STAGE:-0}" RUN_SCORE_SWEEP="${RUN_SCORE_SWEEP:-0}" RUN_TRAIN_STAGE="${RUN_TRAIN_STAGE:-1}"

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

score_modes_for_task_set() {
  case "${TRAIN_CACHE_TASK_SET}" in
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
      printf '%s\n' "${TRAIN_CACHE_TASK_SET}"
      ;;
  esac
}

run_slot() {
  local slot="$1"
  local score_mode="$2"
  local gpu="$((slot % GPU_PER_NODE))"
  local backend="${TACC_SLOT_BACKEND:-ibrun}"
  local unprompted_flag=0
  [[ "${score_mode}" == unprompted_* ]] && unprompted_flag=1

  local cmd=(
    env
    CUDA_VISIBLE_DEVICES="${gpu}"
    ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}"
    SAMPLE_MODEL_MODE="${score_mode}"
    UNPROMPTED="${unprompted_flag}"
    TRAIN_SHARD_INDEX="${slot}"
    bash "${SCRIPT_DIR}/projected_traj_tracin_score_sweep.sh"
  )
  if [[ "${backend}" == "ibrun" && -n "${SLURM_JOB_ID:-}" ]] && command -v ibrun >/dev/null; then
    ibrun -n 1 -o "${slot}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

dry_train_paths() {
  local score_mode="$1"
  local unprompted_flag=0
  [[ "${score_mode}" == unprompted_* ]] && unprompted_flag=1
  RUN_TRAIN_STAGE=0 RUN_QUERY_STAGE=0 RUN_SCORE_SWEEP=0 \
    ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}" SAMPLE_MODEL_MODE="${score_mode}" \
    UNPROMPTED="${unprompted_flag}" TRAIN_SHARD_INDEX=0 \
    bash "${SCRIPT_DIR}/projected_traj_tracin_score_sweep.sh"
}

run_one_score_mode() {
  local score_mode="$1"
  local dry_output train_artifact train_artifact_root

  echo "Projected Traj-TracIn train cache array"
  echo "score_mode=${score_mode}; cache_dim=${PROJECTED_CACHE_DIM}; train_range=${TRAIN_SCORE_INDEX_RANGES}"
  echo "shards=${PROJECTED_TRAIN_SHARD_RANGES}"
  echo "slots=${#SHARDS[@]}/${GPU_SLOTS}; gpu_per_node=${GPU_PER_NODE}"

  dry_output="$(dry_train_paths "${score_mode}")"
  train_artifact="$(printf '%s\n' "${dry_output}" | awk -F= '/^train_artifact=/{print $2; exit}')"
  train_artifact_root="$(dirname "${train_artifact}")"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] merged train artifact: ${train_artifact}"
    echo "[dry-run] shard root: ${train_artifact_root}/shards"
    return
  fi

  if [[ -f "${train_artifact}" ]]; then
    echo "[skip] merged train artifact already exists: ${train_artifact}"
    return
  fi

  pids=()
  for slot in "${!SHARDS[@]}"; do
    echo "[launch] score_mode=${score_mode} slot=${slot} range=${SHARDS[$slot]}"
    run_slot "${slot}" "${score_mode}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if (( failed != 0 )); then
    echo "At least one train shard failed for score_mode=${score_mode}." >&2
    exit 1
  fi

  echo "[merge] locating shard artifacts for score_mode=${score_mode}"
  shard_paths=()
  for shard in "${SHARDS[@]}"; do
    tag="$(path_tag "${shard//-/_}")"
    shard_path="${train_artifact_root}/shards/range_${tag}/train_datapoint_gradient_artifact.npz"
    if [[ ! -f "${shard_path}" ]]; then
      echo "Missing shard artifact: ${shard_path}" >&2
      exit 1
    fi
    shard_paths+=("${shard_path}")
  done

  echo "[merge] output=${train_artifact}"
  "${PYTHON_BIN}" "${REFINE_ROOT}/common/merge_projected_train_shards.py" \
    --output "${train_artifact}" \
    "${shard_paths[@]}"
}

SHARDS=()
while IFS= read -r shard_range; do
  SHARDS+=("${shard_range}")
done < <(split_words "${PROJECTED_TRAIN_SHARD_RANGES}")
if (( ${#SHARDS[@]} > GPU_SLOTS )); then
  echo "This helper expects at most GPU_SLOTS=${GPU_SLOTS} shards, got ${#SHARDS[@]}." >&2
  exit 2
fi

while IFS= read -r score_mode; do
  run_one_score_mode "${score_mode}"
done < <(score_modes_for_task_set)

echo "Projected train cache array complete."
