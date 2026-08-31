#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIFAR2_ROOT="${CIFAR2_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REFINE_ROOT="${REFINE_ROOT:-$(cd "${CIFAR2_ROOT}/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${REFINE_ROOT}/.." && pwd)}"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
fi

cd "${CIFAR2_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment_67}"
TRAIN_SEED="${TRAIN_SEED:-67}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"

GPU_IDS_TEXT="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
GPU_IDS_TEXT="${GPU_IDS_TEXT//,/ }"
read -r -a GPUS <<<"${GPU_IDS_TEXT}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPU ids found. Set GPU_IDS=0,1,2,3." >&2
  exit 1
fi

MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS[@]}}"
RANGES_TEXT="${ATTRIBUTION_RANGES:-${SCORE_INDEX_RANGES:-1-625 626-1250 1251-1875 1876-2500 2501-3125 3126-3750 3751-4375 4376-5000 5001-5625 5626-6250 6251-6875 6876-7500 7501-8125 8126-8750 8751-9375 9376-10000}}"
ALGORITHMS_TEXT="${ALGORITHMS:-traj_tracin das}"
PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-0 1 2 3 4 5 6 7}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23}"
SCORE_MODE_SUFFIX="${SCORE_MODE_SUFFIX:-10x100}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EXPERIMENT_TAG TRAIN_SEED
export TRAJ_NUM_SNAPSHOTS="${TRAJ_NUM_SNAPSHOTS:-100}"
export TRAJ_TRAIN_MC_SAMPLES="${TRAJ_TRAIN_MC_SAMPLES:-10}"
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
export TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-4}"
export TRAJ_SAVE_QUERY_NORMALIZED_SCORES="${TRAJ_SAVE_QUERY_NORMALIZED_SCORES:-1}"
export TRAJ_QUERY_NORMALIZE_EPS="${TRAJ_QUERY_NORMALIZE_EPS:-1e-8}"
export DAS_TIMESTEPS="${DAS_TIMESTEPS:-$(seq -s ' ' 0 10 990)}"
export DAS_NUM_MC_NOISE="${DAS_NUM_MC_NOISE:-10}"
export DAS_GRAD_BATCH_SIZE="${DAS_GRAD_BATCH_SIZE:-8}"
export DAS_DAMPING_SWEEP="${DAS_DAMPING_SWEEP:-1}"
export DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES:-0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000}"

LOG_ROOT="${LOG_ROOT:-${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/local_10x100_logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_ROOT}"

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
  value="${value//:/-}"
  printf 'range_%s_%s' "${value%-*}" "${value#*-}"
}

score_mode_with_suffix() {
  local score_mode="$1"
  if [[ -z "${SCORE_MODE_SUFFIX}" ]]; then
    printf '%s' "${score_mode}"
  else
    printf '%s_%s' "${score_mode}" "${SCORE_MODE_SUFFIX}"
  fi
}

task_lines() {
  local seed range
  for seed in ${UNPROMPTED_SEEDS_TEXT}; do
    for range in ${RANGES_TEXT}; do
      printf 'unprompted_solo|unprompted_solo|unconditional|%s|1|%s\n' "${seed}" "${range}"
    done
  done
  for seed in ${PROMPTED_SEEDS_TEXT}; do
    for range in ${RANGES_TEXT}; do
      printf 'prompted_solo|prompted_solo|horse|%s|0|%s\n' "${seed}" "${range}"
      printf 'prompted_solo|prompted_solo|automobile|%s|0|%s\n' "${seed}" "${range}"
      printf 'prompted_multi|prompted_solo|horse,automobile|%s|0|%s\n' "${seed}" "${range}"
    done
  done
}

running=0
pids=()
failed=0

wait_one_if_needed() {
  while (( running >= MAX_PARALLEL )); do
    wait -n || failed=1
    running=$((running - 1))
  done
}

cleanup() {
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

run_task() {
  local algorithm="$1"
  local sample_mode="$2"
  local score_mode="$3"
  local query="$4"
  local seed="$5"
  local unprompted_flag="$6"
  local range="$7"
  local gpu="$8"

  local score_mode_out
  score_mode_out="$(score_mode_with_suffix "${score_mode}")"

  local query_env="${query}"
  if [[ "${unprompted_flag}" == "1" ]]; then
    query_env="unconditional"
  fi

  local prompt_tag seed_tag ckpt_stem sample_root sample_done_file sample_lock_dir
  prompt_tag="$(path_tag "${query_env}")"
  printf -v seed_tag "%06d" "${seed}"
  printf -v ckpt_stem "seed_%s_epoch_%04d" "${TRAIN_SEED}" "${JAX_EPOCHS}"
  sample_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/sample/cifar/prompt_${prompt_tag}/model_${sample_mode}__ckpt_${ckpt_stem}"
  sample_done_file="${sample_root}/seed_${seed_tag}/trajectory_xt.npy"
  sample_lock_dir="${sample_root}/.sample_seed_${seed_tag}.lock"
  mkdir -p "${sample_root}"

  echo "[sample] algorithm=${algorithm} sample_mode=${sample_mode} score_mode=${score_mode_out} query=${query_env} seed=${seed} gpu=${gpu}"
  if [[ -f "${sample_done_file}" ]]; then
    echo "[sample] reuse ${sample_done_file}"
  elif mkdir "${sample_lock_dir}" 2>/dev/null; then
    trap 'rmdir "${sample_lock_dir}" 2>/dev/null || true' RETURN
    echo "[sample] generating ${sample_done_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    GPU_IDS="${gpu}" \
    JAX_NUM_DEVICES=1 \
    SAMPLE_MODEL_MODE="${sample_mode}" \
    ATTRIBUTION_SAMPLE_MODEL_MODE="${sample_mode}" \
    QUERY="${query_env}" \
    INITIAL_SEED="${seed}" \
    SAMPLE_SEED="${seed}" \
    SAMPLE_SEEDS="${seed}" \
    UNPROMPTED="${unprompted_flag}" \
    bash scripts/00_sample_for_attribution.sh
    rmdir "${sample_lock_dir}" 2>/dev/null || true
    trap - RETURN
  else
    echo "[sample] waiting for ${sample_done_file}"
    while [[ ! -f "${sample_done_file}" ]]; do sleep 30; done
  fi

  echo "[run] algorithm=${algorithm} range=${range} score_mode=${score_mode_out} query=${query_env} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  GPU_IDS="${gpu}" \
  JAX_NUM_DEVICES=1 \
  SCORE_INDEX_RANGES="${range}" \
  ATTRIBUTION_RANGES="${range}" \
  SAMPLE_MODEL_MODE="${sample_mode}" \
  UNPROMPTED_SAMPLE_MODEL_MODE="${sample_mode}" \
  ATTRIBUTION_SAMPLE_MODEL_MODE="${sample_mode}" \
  ATTRIBUTION_SCORE_MODEL_MODE="${score_mode_out}" \
  UNPROMPTED_SCORE_MODEL_MODE="${score_mode_out}" \
  QUERY="${query_env}" \
  INITIAL_SEED="${seed}" \
  SAMPLE_SEED="${seed}" \
  SAMPLE_SEEDS="${seed}" \
  UNPROMPTED="${unprompted_flag}" \
  ALGORITHM="${algorithm}" \
  "${PYTHON_BIN}" "${REFINE_ROOT}/common/run_original_attribution_config.py" "${CIFAR2_ROOT}/data_attribution/${algorithm}/CONFIG.py"
}

mapfile -t TASKS < <(task_lines)
echo "CIFAR2 local 10x100 attribution"
echo "repo=${REPO_ROOT}"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; score_suffix=${SCORE_MODE_SUFFIX}"
echo "algorithms=${ALGORITHMS_TEXT}; ranges=${RANGES_TEXT}"
echo "traj=${TRAJ_TRAIN_MC_SAMPLES}mc x ${TRAJ_NUM_SNAPSHOTS} snapshots"
echo "das=${DAS_NUM_MC_NOISE}mc x $(wc -w <<<"${DAS_TIMESTEPS}") timestamps"
echo "gpus=${GPUS[*]}; max_parallel=${MAX_PARALLEL}"
echo "logs=${LOG_ROOT}"

task_id=0
for algorithm in ${ALGORITHMS_TEXT}; do
  for line in "${TASKS[@]}"; do
    IFS='|' read -r sample_mode score_mode query seed unprompted_flag range <<<"${line}"
    gpu="${GPUS[$((task_id % ${#GPUS[@]}))]}"
    log="${LOG_ROOT}/task_${task_id}__${algorithm}__$(score_mode_with_suffix "${score_mode}")__$(path_tag "${query}")__seed_${seed}__$(range_tag "${range}").log"
    echo "[launch] task=${task_id} algorithm=${algorithm} query=${query} seed=${seed} range=${range} gpu=${gpu} -> ${log}"
    wait_one_if_needed
    run_task "${algorithm}" "${sample_mode}" "${score_mode}" "${query}" "${seed}" "${unprompted_flag}" "${range}" "${gpu}" >"${log}" 2>&1 &
    pids+=("$!")
    running=$((running + 1))
    task_id=$((task_id + 1))
  done
done

wait || failed=1
trap - INT TERM

if (( failed != 0 )); then
  echo "At least one task failed. Check logs under ${LOG_ROOT}" >&2
  exit 1
fi

echo "CIFAR2 local 10x100 attribution complete."
