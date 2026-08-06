#!/usr/bin/env bash
#SBATCH -J cifar2-full-traj-term
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 48:00:00
#SBATCH -o cifar2-full-traj-term-%j.out
#SBATCH -e cifar2-full-traj-term-%j.err

set -euo pipefail

SCRIPT_DIR_CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR_CANDIDATE}/projected_traj_tracin_score_sweep.sh" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_CANDIDATE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/cifar2/tacc/h100/projected_traj_tracin_score_sweep.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/cifar2/tacc/h100" && pwd)"
else
  echo "Could not locate h100 helper scripts from ${SCRIPT_DIR_CANDIDATE} or SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}." >&2
  exit 2
fi
CIFAR2_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REFINE_ROOT="$(cd "${CIFAR2_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${REFINE_ROOT}/.." && pwd)"

source "${CIFAR2_ROOT}/stampede3_das/_stampede3_das_lib.sh"
stampede3_das_init

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_67}"
TRAIN_SEED="${TRAIN_SEED:-67}"
INITIAL_SEED="${INITIAL_SEED:-7}"
TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-8}"
TRAJ_USE_SAVED_TRAJECTORY="${TRAJ_USE_SAVED_TRAJECTORY:-0}"
TRAJ_SAVE_QUERY_NORMALIZED_SCORES="${TRAJ_SAVE_QUERY_NORMALIZED_SCORES:-0}"
TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS="${TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS:-raw}"
FULL_SCORE_RANGES="${FULL_SCORE_RANGES:-1-2500 2501-5000 5001-7500 7501-10000}"
FULL_QUERY_SPECS="${FULL_QUERY_SPECS:-prompted_solo:horse prompted_solo:automobile prompted_solo:horse_automobile unprompted_solo:unconditional}"
GPU_PER_NODE="${GPU_PER_NODE:-4}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHON_BIN EXPERIMENT_TAG TRAIN_SEED INITIAL_SEED
export TRAJ_SCORE_BATCH_SIZE TRAJ_SNAPSHOT_CHUNK_SIZE TRAJ_USE_SAVED_TRAJECTORY
export TRAJ_SAVE_QUERY_NORMALIZED_SCORES TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS

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

query_component() {
  local mode="$1"
  local query="$2"
  if [[ "${mode}" == unprompted_* ]]; then
    printf '%s' "unprompted"
  else
    printf 'query_%s' "$(path_tag "${query}")"
  fi
}

algorithm_dir() {
  local mode="$1"
  local range="$2"
  local rtag
  rtag="$(range_tag "${range}")"
  if [[ "${mode}" == unprompted_* ]]; then
    printf 'traj_tracin_unprompted_range_%s' "${rtag}"
  else
    printf 'traj_tracin_range_%s' "${rtag}"
  fi
}

run_one() {
  local slot="$1"
  local mode="$2"
  local query="$3"
  local range="$4"
  local gpu="$((slot % GPU_PER_NODE))"
  local unprompted_flag=0
  [[ "${mode}" == unprompted_* ]] && unprompted_flag=1

  local qcomp
  qcomp="$(query_component "${mode}" "${query}")"
  local alg
  alg="$(algorithm_dir "${mode}" "${range}")"
  local out_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/attribution_score/${mode}/train_seed_${TRAIN_SEED}/${qcomp}/initial_seed_${INITIAL_SEED}/${alg}"
  local term_artifact="${out_dir}/full_dim_term_scores.npz"

  if [[ -f "${out_dir}/scores.npy" && -f "${out_dir}/score_indices.npy" && -f "${term_artifact}" ]]; then
    echo "[skip] mode=${mode} query=${query} range=${range} existing ${out_dir}"
    return
  fi

  echo "[full-term] slot=${slot} gpu=${gpu} mode=${mode} query=${query} range=${range} -> ${out_dir}"
  local cmd=(
    env
    CUDA_VISIBLE_DEVICES="${gpu}"
    ATTRIBUTION_SCORE_MODEL_MODE="${mode}"
    SAMPLE_MODEL_MODE="${mode}"
    UNPROMPTED="${unprompted_flag}"
    QUERY="${query}"
    INITIAL_SEED="${INITIAL_SEED}"
    SAMPLE_SEED="${INITIAL_SEED}"
    SCORE_INDEX_RANGES="${range}"
    ATTRIBUTION_RANGES="${range}"
    TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE}"
    TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE}"
    TRAJ_USE_SAVED_TRAJECTORY="${TRAJ_USE_SAVED_TRAJECTORY}"
    TRAJ_SAVE_QUERY_NORMALIZED_SCORES="${TRAJ_SAVE_QUERY_NORMALIZED_SCORES}"
    TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS="${TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS}"
    TRAJ_TRACIN_FULL_TERM_SCORE_ARTIFACT_PATH="${term_artifact}"
    "${PYTHON_BIN}" "${REFINE_ROOT}/common/run_original_attribution_config.py"
    "${CIFAR2_ROOT}/data_attribution/traj_tracin/CONFIG.py"
  )
  if [[ "${TACC_SLOT_BACKEND:-ibrun}" == "ibrun" && -n "${SLURM_JOB_ID:-}" ]] && command -v ibrun >/dev/null; then
    ibrun -n 1 -o "${slot}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

mapfile -t ranges < <(split_words "${FULL_SCORE_RANGES}")
mapfile -t query_specs < <(split_words "${FULL_QUERY_SPECS}")

echo "Full-dim Traj-TracIn term-score run"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; initial_seed=${INITIAL_SEED}"
echo "queries=${FULL_QUERY_SPECS}"
echo "ranges=${FULL_SCORE_RANGES}"
echo "term_score_variants=${TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS}"
echo "score_batch_size=${TRAJ_SCORE_BATCH_SIZE}; snapshot_chunk_size=${TRAJ_SNAPSHOT_CHUNK_SIZE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  slot=0
  for spec in "${query_specs[@]}"; do
    mode="${spec%%:*}"
    query="${spec#*:}"
    for range in "${ranges[@]}"; do
      echo "DRY slot=${slot} mode=${mode} query=${query} range=${range}"
      slot=$((slot + 1))
    done
  done
  exit 0
fi

pids=()
slot=0
for spec in "${query_specs[@]}"; do
  mode="${spec%%:*}"
  query="${spec#*:}"
  for range in "${ranges[@]}"; do
    run_one "${slot}" "${mode}" "${query}" "${range}" &
    pids+=("$!")
    slot=$((slot + 1))
  done
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed != 0 )); then
  echo "At least one full-dim term-score task failed." >&2
  exit 1
fi

echo "Full-dim Traj-TracIn term-score run complete."
