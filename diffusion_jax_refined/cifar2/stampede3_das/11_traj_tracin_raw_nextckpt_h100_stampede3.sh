#!/usr/bin/env bash
#SBATCH -J cifar2-s3-traj-raw-next
#SBATCH -o cifar2-s3-traj-raw-next-%j.out
#SBATCH -e cifar2-s3-traj-raw-next-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00

set -euo pipefail

export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_67}"
export TRAIN_SEED="${TRAIN_SEED:-67}"

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

ATTR_NUM_SLOTS="${ATTR_NUM_SLOTS:-16}"
PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 7)}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 23)}"
TRAJ_RANGES_TEXT="${TRAJ_RANGES:-1-2500 2501-5000 5001-7500 7501-10000}"
TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"
TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-4}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
TRAJ_QUERY_OBJECTIVE_VALUE="${TRAJ_QUERY_OBJECTIVE_VALUE:-trajectory_next_checkpoint_noise_mse}"
TRAJ_PARAMETER_SOURCE_VALUE="${TRAJ_PARAMETER_SOURCE_VALUE:-raw}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/11_traj_tracin_raw_nextckpt/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

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

score_file_for_task() {
  local score_mode="$1"
  local query="$2"
  local seed="$3"
  local unprompted_flag="$4"
  local range="$5"
  local tag algorithm_dir
  tag="$(range_tag "${range}")"
  if [[ "${unprompted_flag}" == "1" || "${score_mode}" == unprompted_* ]]; then
    algorithm_dir="traj_tracin_${TRAJ_QUERY_OBJECTIVE_VALUE}_unprompted_${TRAJ_PARAMETER_SOURCE_VALUE}_${tag}"
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/unprompted/initial_seed_%s/%s/scores.npy' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${score_mode}" "${TRAIN_SEED}" "${seed}" "${algorithm_dir}"
  else
    algorithm_dir="traj_tracin_${TRAJ_QUERY_OBJECTIVE_VALUE}_${TRAJ_PARAMETER_SOURCE_VALUE}_${tag}"
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/%s/scores.npy' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${score_mode}" "${TRAIN_SEED}" "$(path_tag "${query}")" "${seed}" "${algorithm_dir}"
  fi
}

run_one_task() {
  local i="$1"
  local slot="$2"
  local gpu="$3"
  local sample_mode score_mode query seed unprompted_flag range
  IFS='|' read -r sample_mode score_mode query seed unprompted_flag range <<<"${TASKS[$i]}"
  local score_file
  score_file="$(score_file_for_task "${score_mode}" "${query}" "${seed}" "${unprompted_flag}" "${range}")"
  local log="${LOG_ROOT}/task_${i}__traj_raw_next__${score_mode}__$(path_tag "${query}")__seed_${seed}__$(range_tag "${range}").log"
  if [[ -f "${score_file}" ]]; then
    echo "[traj-skip] task=${i} existing=${score_file}" >"${log}"
    return 0
  fi

  local query_env="${query}"
  if [[ "${unprompted_flag}" == "1" ]]; then
    query_env="unconditional"
  fi
  local prompt_tag seed_tag ckpt_stem sample_run_root sample_done_file sample_lock_dir
  prompt_tag="$(path_tag "${query_env}")"
  printf -v seed_tag "%06d" "${seed}"
  printf -v ckpt_stem "seed_%s_epoch_%04d" "${TRAIN_SEED}" "${JAX_EPOCHS}"
  sample_run_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/sample/cifar/prompt_${prompt_tag}/model_${sample_mode}__ckpt_${ckpt_stem}"
  mkdir -p "${sample_run_root}"
  sample_done_file="${sample_run_root}/seed_${seed_tag}/trajectory_xt.npy"
  sample_lock_dir="${sample_run_root}/.sample_seed_${seed_tag}.lock"

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
    TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE_VALUE}" \
    TRAJ_PARAMETER_SOURCE="${TRAJ_PARAMETER_SOURCE_VALUE}" \
    TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE}" \
    TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE}" \
    TRAJ_SAVE_QUERY_NORMALIZED_SCORES=0 \
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
    SAMPLE_DONE_FILE="${sample_done_file}" \
    SAMPLE_LOCK_DIR="${sample_lock_dir}" \
    SAMPLE_LOCK_WAIT_SECONDS="${SAMPLE_LOCK_WAIT_SECONDS:-21600}" \
    UNPROMPTED="${unprompted_flag}" \
    ALGORITHM=traj_tracin \
    bash "${SCRIPT_DIR}/02_dtrak_endtracin_attribution_task_stampede3.sh" \
    >"${log}" 2>&1
}

mapfile -t TASKS < <(task_lines)
total_tasks="${#TASKS[@]}"

echo "Stampede3 Traj-TracIn raw-parameter next-checkpoint attribution"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; objective=${TRAJ_QUERY_OBJECTIVE_VALUE}; parameter_source=${TRAJ_PARAMETER_SOURCE_VALUE}"
echo "total_tasks=${total_tasks}; slots=${ATTR_NUM_SLOTS}; ranges=${TRAJ_RANGES_TEXT}; logs=${LOG_ROOT}"

pids=()
for ((slot = 0; slot < ATTR_NUM_SLOTS; slot++)); do
  gpu=$((slot % 4))
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
    echo "Found TrajTracIn raw-next artifact for task=${i}: ${score_file}"
  else
    echo "Missing TrajTracIn raw-next artifact for task=${i}: ${score_file}" >&2
    missing=$((missing + 1))
  fi
done
if (( missing > 0 )); then
  echo "Job 11 missing ${missing} expected score artifacts." >&2
  exit 1
fi
echo "Job 11 complete."
