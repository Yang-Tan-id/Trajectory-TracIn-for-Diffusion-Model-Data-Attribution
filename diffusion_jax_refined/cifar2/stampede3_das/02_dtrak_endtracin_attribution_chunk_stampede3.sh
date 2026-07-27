#!/usr/bin/env bash
#SBATCH -J cifar2-s3-orig-attr
#SBATCH -o cifar2-s3-orig-attr-%j.out
#SBATCH -e cifar2-s3-orig-attr-%j.err
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

ATTR_JOB_INDEX="${ATTR_JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
ATTR_NUM_JOBS="${ATTR_NUM_JOBS:-6}"
ATTR_CHUNK_SIZE="${ATTR_CHUNK_SIZE:-16}"
ATTR_ALGORITHMS_TEXT="${ATTR_ALGORITHMS:-dtrak end_tracin}"
PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 7)}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 23)}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/02_dtrak_endtracin_attribution/${SLURM_JOB_ID:-local}/chunk_${ATTR_JOB_INDEX}"
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

mapfile -t TASKS < <(task_lines)
total_tasks="${#TASKS[@]}"
start=$((ATTR_JOB_INDEX * ATTR_CHUNK_SIZE))
end=$((start + ATTR_CHUNK_SIZE))
if (( end > total_tasks )); then
  end="${total_tasks}"
fi

echo "Job 02 Stampede3 dtrak/end_tracin: sample trajectories and run attribution"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; chunk=${ATTR_JOB_INDEX}/${ATTR_NUM_JOBS}; range=[${start}, ${end}); total_tasks=${total_tasks}"
echo "algorithms=${ATTR_ALGORITHMS_TEXT}; dtrak_train_expect=${DTRAK_TRAIN_EXPECTATION_SAMPLES:-100}; dtrak_query_expect=${DTRAK_QUERY_EXPECTATION_SAMPLES:-100}; end_tracin_endpoint_mc=${END_TRACIN_ENDPOINT_MC_SAMPLES:-100}; end_tracin_train_mc=${END_TRACIN_TRAIN_MC_SAMPLES:-100}"
echo "nodes=4; gpu_tasks=16; logs=${LOG_ROOT}"

if (( start >= total_tasks )); then
  echo "Chunk ${ATTR_JOB_INDEX} has no tasks."
  exit 0
fi

pids=()
slot=0
for ((i = start; i < end; i++)); do
  IFS='|' read -r algorithm sample_mode score_mode query seed unprompted_flag <<<"${TASKS[$i]}"
  query_env="${query}"
  if [[ "${unprompted_flag}" == "1" ]]; then
    query_env="unconditional"
  fi
  prompt_tag="$(path_tag "${query_env}")"
  printf -v seed_tag "%06d" "${seed}"
  printf -v ckpt_stem "seed_%s_epoch_%04d" "${TRAIN_SEED}" "${JAX_EPOCHS}"
  sample_run_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/sample/cifar/prompt_${prompt_tag}/model_${sample_mode}__ckpt_${ckpt_stem}"
  mkdir -p "${sample_run_root}"
  sample_done_file="${sample_run_root}/seed_${seed_tag}/trajectory_xt.npy"
  sample_lock_dir="${sample_run_root}/.sample_seed_${seed_tag}.lock"
  gpu=$((slot % 4))
  log="${LOG_ROOT}/task_${i}__${algorithm}__${score_mode}__$(path_tag "${query}")__seed_${seed}.log"
  echo "Launch task=${i} algorithm=${algorithm} sample_mode=${sample_mode} score_mode=${score_mode} query=${query} seed=${seed} slot=${slot} gpu=${gpu} -> ${log}"
  (
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
      DTRAK_BATCH_SIZE="${DTRAK_BATCH_SIZE:-64}" \
      END_TRACIN_ENDPOINT_MC_SAMPLES="${END_TRACIN_ENDPOINT_MC_SAMPLES:-100}" \
      END_TRACIN_TRAIN_MC_SAMPLES="${END_TRACIN_TRAIN_MC_SAMPLES:-100}" \
      END_TRACIN_SCORE_BATCH_SIZE="${END_TRACIN_SCORE_BATCH_SIZE:-32}" \
      bash "${SCRIPT_DIR}/02_dtrak_endtracin_attribution_task_stampede3.sh"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done

wait_all "${pids[@]}"
missing=0
for ((i = start; i < end; i++)); do
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
  echo "Job 02 chunk ${ATTR_JOB_INDEX} missing ${missing} expected score artifacts." >&2
  exit 1
fi
echo "Job 02 chunk ${ATTR_JOB_INDEX} complete."
