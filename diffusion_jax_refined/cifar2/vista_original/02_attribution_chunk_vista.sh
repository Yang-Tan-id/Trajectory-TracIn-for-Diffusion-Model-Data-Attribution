#!/usr/bin/env bash
#SBATCH --job-name=cifar2-orig-attr-chunk
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=cifar2-orig-attr-chunk-%j.out
#SBATCH --error=cifar2-orig-attr-chunk-%j.err

set -euo pipefail

SCRIPT_DIR="${VISTA_ORIGINAL_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_vista_original_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/vista_original"; do
    if [[ -n "${candidate}" && -f "${candidate}/_vista_original_lib.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_vista_original_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_vista_original_lib.sh
source "${SCRIPT_DIR}/_vista_original_lib.sh"
vista_original_init

ATTR_JOB_INDEX="${ATTR_JOB_INDEX:-0}"
ATTR_NUM_JOBS="${ATTR_NUM_JOBS:-6}"
ATTR_CHUNK_SIZE="${ATTR_CHUNK_SIZE:-64}"
PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 7)}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 23)}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
TRAJ_RANGES=(1-2500 2501-5000 5001-7500 7501-10000)
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_original_logs/02_attribution_chunks/${SLURM_JOB_ID:-local}/chunk_${ATTR_JOB_INDEX}"
mkdir -p "${LOG_ROOT}"

task_lines() {
  local range seed
  for range in "${TRAJ_RANGES[@]}"; do
    for seed in ${PROMPTED_SEEDS_TEXT}; do
      printf 'traj_tracin|%s|prompted_solo|prompted_solo|horse|%s|0\n' "${range}" "${seed}"
      printf 'traj_tracin|%s|prompted_solo|prompted_solo|automobile|%s|0\n' "${range}" "${seed}"
      printf 'traj_tracin|%s|prompted_multi|prompted_solo|horse,automobile|%s|0\n' "${range}" "${seed}"
    done
  done
  for range in "${TRAJ_RANGES[@]}"; do
    for seed in ${UNPROMPTED_SEEDS_TEXT}; do
      printf 'traj_tracin|%s|unprompted_solo|unprompted_solo|unprompted|%s|1\n' "${range}" "${seed}"
    done
  done
  for seed in ${PROMPTED_SEEDS_TEXT}; do
    printf 'das||prompted_solo|prompted_solo|horse|%s|0\n' "${seed}"
    printf 'das||prompted_solo|prompted_solo|automobile|%s|0\n' "${seed}"
    printf 'das||prompted_multi|prompted_solo|horse,automobile|%s|0\n' "${seed}"
  done
  for seed in ${UNPROMPTED_SEEDS_TEXT}; do
    printf 'das||unprompted_solo|unprompted_solo|unprompted|%s|1\n' "${seed}"
  done
  for algorithm in dtrak end_tracin; do
    for seed in ${PROMPTED_SEEDS_TEXT}; do
      printf '%s||prompted_solo|prompted_solo|horse|%s|0\n' "${algorithm}" "${seed}"
      printf '%s||prompted_solo|prompted_solo|automobile|%s|0\n' "${algorithm}" "${seed}"
      printf '%s||prompted_multi|prompted_solo|horse,automobile|%s|0\n' "${algorithm}" "${seed}"
    done
    for seed in ${UNPROMPTED_SEEDS_TEXT}; do
      printf '%s||unprompted_solo|unprompted_solo|unprompted|%s|1\n' "${algorithm}" "${seed}"
    done
  done
}

mapfile -t TASKS < <(task_lines)
total_tasks="${#TASKS[@]}"
start=$((ATTR_JOB_INDEX * ATTR_CHUNK_SIZE))
end=$((start + ATTR_CHUNK_SIZE))
if (( end > total_tasks )); then
  end="${total_tasks}"
fi

echo "Job 02 chunk: sample trajectories and run original attribution"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; chunk=${ATTR_JOB_INDEX}/${ATTR_NUM_JOBS}; range=[${start}, ${end}); total_tasks=${total_tasks}"
echo "priority order: traj_tracin prompted/unprompted ranges, das prompted/unprompted, then dtrak/end_tracin"
echo "logs=${LOG_ROOT}"

if (( start >= total_tasks )); then
  echo "Chunk ${ATTR_JOB_INDEX} has no tasks."
  exit 0
fi

pids=()
slot=0
for ((i = start; i < end; i++)); do
  IFS='|' read -r algorithm range sample_mode score_mode query seed unprompted_flag <<<"${TASKS[$i]}"
  query_env="${query}"
  if [[ "${unprompted_flag}" == "1" ]]; then
    query_env="unconditional"
  fi
  range_tag="${range:-all}"
  prompt_tag="$(path_tag "${query_env}")"
  printf -v seed_tag "%06d" "${seed}"
  printf -v ckpt_stem "seed_%s_epoch_%04d" "${TRAIN_SEED}" "${JAX_EPOCHS}"
  sample_run_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/sample/cifar/prompt_${prompt_tag}/model_${sample_mode}__ckpt_${ckpt_stem}"
  sample_done_file="${sample_run_root}/seed_${seed_tag}/trajectory_xt.npy"
  sample_lock_dir="${sample_run_root}/.sample_seed_${seed_tag}.lock"
  log="${LOG_ROOT}/task_${i}__${algorithm}__${score_mode}__$(path_tag "${query}")__seed_${seed}__range_${range_tag}.log"
  echo "Launch task=${i} algorithm=${algorithm} range=${range_tag} sample_mode=${sample_mode} score_mode=${score_mode} query=${query} seed=${seed} slot=${slot} -> ${log}"
  (
    run_slot "${slot}" env \
      CUDA_VISIBLE_DEVICES=0 \
      GPU_IDS=0 \
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
      ATTRIBUTION_RANGES="${range}" \
      SCORE_INDEX_RANGES="${range}" \
      UNPROMPTED="${unprompted_flag}" \
      ALGORITHM="${algorithm}" \
      bash -c '
        set -euo pipefail
        echo "[sample] sample_mode=${SAMPLE_MODEL_MODE} score_mode=${ATTRIBUTION_SCORE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED}"
        if [[ -f "${SAMPLE_DONE_FILE}" ]]; then
          echo "[sample] reuse existing ${SAMPLE_DONE_FILE}"
        elif mkdir "${SAMPLE_LOCK_DIR}" 2>/dev/null; then
          trap '\''rmdir "${SAMPLE_LOCK_DIR}" 2>/dev/null || true'\'' EXIT
          if [[ -f "${SAMPLE_DONE_FILE}" ]]; then
            echo "[sample] reuse existing ${SAMPLE_DONE_FILE}"
          else
            echo "[sample] generating ${SAMPLE_DONE_FILE}"
            bash scripts/00_sample_for_attribution.sh
          fi
          rmdir "${SAMPLE_LOCK_DIR}" 2>/dev/null || true
          trap - EXIT
        else
          echo "[sample] waiting for locked sample ${SAMPLE_DONE_FILE}"
          waited=0
          while [[ ! -f "${SAMPLE_DONE_FILE}" ]]; do
            if (( waited >= SAMPLE_LOCK_WAIT_SECONDS )); then
              echo "Timed out waiting for sample ${SAMPLE_DONE_FILE}" >&2
              exit 1
            fi
            sleep 30
            waited=$((waited + 30))
          done
          echo "[sample] reuse after wait ${SAMPLE_DONE_FILE}"
        fi
        echo "[original-attribution] algorithm=${ALGORITHM} range=${ATTRIBUTION_RANGES:-all} score_mode=${ATTRIBUTION_SCORE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED}"
        "${PYTHON_BIN}" "${REFINE_ROOT}/common/run_original_attribution_config.py" "${CIFAR2_ROOT}/data_attribution/${ALGORITHM}/CONFIG.py"
      '
  ) >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done

wait_all "${pids[@]}"
echo "Job 02 chunk ${ATTR_JOB_INDEX} complete."
