#!/usr/bin/env bash
#SBATCH -J cifar2-s3-orig-attr-all
#SBATCH -o cifar2-s3-orig-attr-all-%j.out
#SBATCH -e cifar2-s3-orig-attr-all-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

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

ATTR_NUM_SLOTS="${ATTR_NUM_SLOTS:-16}"
ATTR_ALGORITHMS_TEXT="${ATTR_ALGORITHMS:-dtrak end_tracin}"
PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 7)}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 23)}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/02_dtrak_endtracin_attribution_all/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

task_lines() {
  local algorithm seed
  for algorithm in ${ATTR_ALGORITHMS_TEXT}; do
    for seed in ${UNPROMPTED_SEEDS_TEXT}; do
      printf '%s|unprompted_solo|unprompted_solo|unprompted|%s|1\n' "${algorithm}" "${seed}"
    done
    for seed in ${PROMPTED_SEEDS_TEXT}; do
      printf '%s|prompted_solo|prompted_solo|horse|%s|0\n' "${algorithm}" "${seed}"
      printf '%s|prompted_solo|prompted_solo|automobile|%s|0\n' "${algorithm}" "${seed}"
      printf '%s|prompted_multi|prompted_solo|horse,automobile|%s|0\n' "${algorithm}" "${seed}"
    done
  done
}

expected_score_file() {
  local algorithm="$1"
  local score_mode="$2"
  local query="$3"
  local seed="$4"
  local unprompted_flag="$5"
  if [[ "${unprompted_flag}" == "1" || "${score_mode}" == unprompted_* ]]; then
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/unprompted/initial_seed_%s/%s_unprompted/scores.npy' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${score_mode}" "${TRAIN_SEED}" "${seed}" "${algorithm}"
  else
    printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/%s/scores.npy' \
      "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${score_mode}" "${TRAIN_SEED}" "$(path_tag "${query}")" "${seed}" "${algorithm}"
  fi
}

run_one_task() {
  local i="$1"
  local slot="$2"
  local gpu="$3"
  local algorithm sample_mode score_mode query seed unprompted_flag
  IFS='|' read -r algorithm sample_mode score_mode query seed unprompted_flag <<<"${TASKS[$i]}"
  local query_env="${query}"
  if [[ "${unprompted_flag}" == "1" ]]; then
    query_env="unconditional"
  fi
  local prompt_tag seed_tag ckpt_stem sample_run_root sample_done_file sample_lock_dir log
  prompt_tag="$(path_tag "${query_env}")"
  printf -v seed_tag "%06d" "${seed}"
  printf -v ckpt_stem "seed_%s_epoch_%04d" "${TRAIN_SEED}" "${JAX_EPOCHS}"
  sample_run_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/sample/cifar/prompt_${prompt_tag}/model_${sample_mode}__ckpt_${ckpt_stem}"
  mkdir -p "${sample_run_root}"
  sample_done_file="${sample_run_root}/seed_${seed_tag}/trajectory_xt.npy"
  sample_lock_dir="${sample_run_root}/.sample_seed_${seed_tag}.lock"
  log="${LOG_ROOT}/task_${i}__${algorithm}__${score_mode}__$(path_tag "${query}")__seed_${seed}.log"
  echo "[worker ${slot}] task=${i} algorithm=${algorithm} sample_mode=${sample_mode} score_mode=${score_mode} query=${query} seed=${seed} gpu=${gpu} -> ${log}"
  run_gpu_slot "${slot}" env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    GPU_IDS="${gpu}" \
    JAX_NUM_DEVICES=1 \
    PYTHONUNBUFFERED=1 \
    ATTRIBUTION_TQDM_MININTERVAL="${ATTRIBUTION_TQDM_MININTERVAL:-1}" \
    ATTRIBUTION_TQDM_LEAVE="${ATTRIBUTION_TQDM_LEAVE:-1}" \
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
    ALGORITHM="${algorithm}" \
    DTRAK_TRAIN_EXPECTATION_SAMPLES="${DTRAK_TRAIN_EXPECTATION_SAMPLES:-100}" \
    DTRAK_QUERY_EXPECTATION_SAMPLES="${DTRAK_QUERY_EXPECTATION_SAMPLES:-100}" \
    DTRAK_NUM_SAMPLES="${DTRAK_NUM_SAMPLES:-1}" \
    DTRAK_BATCH_SIZE="${DTRAK_BATCH_SIZE:-8}" \
    END_TRACIN_ENDPOINT_MC_SAMPLES="${END_TRACIN_ENDPOINT_MC_SAMPLES:-100}" \
    END_TRACIN_TRAIN_MC_SAMPLES="${END_TRACIN_TRAIN_MC_SAMPLES:-100}" \
    END_TRACIN_SCORE_BATCH_SIZE="${END_TRACIN_SCORE_BATCH_SIZE:-8}" \
    bash "${SCRIPT_DIR}/02_dtrak_endtracin_attribution_task_stampede3.sh" \
    >"${log}" 2>&1
}

mapfile -t TASKS < <(task_lines)
total_tasks="${#TASKS[@]}"

echo "Job 02 Stampede3 dtrak/end_tracin all-in-one attribution"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; total_tasks=${total_tasks}; slots=${ATTR_NUM_SLOTS}"
echo "algorithms=${ATTR_ALGORITHMS_TEXT}; each slot runs task indices slot, slot+${ATTR_NUM_SLOTS}, slot+2*${ATTR_NUM_SLOTS}, ..."
echo "dtrak_train_expect=${DTRAK_TRAIN_EXPECTATION_SAMPLES:-100}; dtrak_query_expect=${DTRAK_QUERY_EXPECTATION_SAMPLES:-100}; end_tracin_endpoint_mc=${END_TRACIN_ENDPOINT_MC_SAMPLES:-100}; end_tracin_train_mc=${END_TRACIN_TRAIN_MC_SAMPLES:-100}"
echo "logs=${LOG_ROOT}"

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
  IFS='|' read -r algorithm _sample_mode score_mode query seed unprompted_flag <<<"${TASKS[$i]}"
  score_file="$(expected_score_file "${algorithm}" "${score_mode}" "${query}" "${seed}" "${unprompted_flag}")"
  if [[ ! -f "${score_file}" ]]; then
    echo "Missing ${algorithm} score artifact for task=${i}: ${score_file}" >&2
    missing=$((missing + 1))
  else
    echo "Found ${algorithm} score artifact for task=${i}: ${score_file}"
  fi
done
if (( missing > 0 )); then
  echo "Job 02 all-in-one missing ${missing} expected score artifacts." >&2
  exit 1
fi
echo "Job 02 all-in-one complete."
