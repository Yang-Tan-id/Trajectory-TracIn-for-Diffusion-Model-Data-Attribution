#!/usr/bin/env bash
#SBATCH --job-name=cifar2-orig-lds-eval
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=48
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=cifar2-orig-lds-eval-%j.out
#SBATCH --error=cifar2-orig-lds-eval-%j.err

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

ALGORITHMS=(dtrak das end_tracin traj_tracin)
TARGETS=(simple_loss noise_trajectory)
LDS_SEEDS_TEXT="${LDS_SEEDS:-$(seq -s ' ' 0 7)}"
LDS_M="${LDS_M:-64}"
LDS_K="${LDS_K:-5000}"
LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-50}"
LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"
PRED_TAG="${PRED_TAG:-pred_kept_sign_m1}"
DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES:-0.1 0.2 0.5 1 2 5 10 20 50}"
PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 7)}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 23)}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_original_logs/03_lds_eval_report/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

damping_tag() {
  local value="$1"
  value="${value//+/_}"
  value="${value//-/neg_}"
  value="${value//./p}"
  printf '%s' "$(path_tag "${value}")"
}

EVAL_ALGORITHMS=(dtrak end_tracin traj_tracin)
if [[ "${DAS_DAMPING_SWEEP:-0}" == "1" ]]; then
  for lambda in ${DAS_DAMPING_SWEEP_VALUES}; do
    EVAL_ALGORITHMS+=("das_lambda_$(damping_tag "${lambda}")")
  done
else
  EVAL_ALGORITHMS+=(das)
fi

query_specs() {
  local seed
  for seed in ${UNPROMPTED_SEEDS_TEXT}; do
    printf 'unprompted_solo|unprompted|unconditional|%s|1\n' "${seed}"
  done
  for seed in ${PROMPTED_SEEDS_TEXT}; do
    printf 'prompted_solo|horse|horse|%s|0\n' "${seed}"
    printf 'prompted_solo|automobile|automobile|%s|0\n' "${seed}"
    printf 'prompted_solo|horse,automobile|horse,automobile|%s|0\n' "${seed}"
  done
}

mapfile -t SPECS < <(query_specs)

echo "Job 03: LDS eval + aggregate/report"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; lds_seeds=${LDS_SEEDS_TEXT}; lds_m=${LDS_M}; targets=${TARGETS[*]}; algorithms=${ALGORITHMS[*]}; eval_algorithms=${EVAL_ALGORITHMS[*]}"
echo "query tasks=${#SPECS[@]}; logs=${LOG_ROOT}"

pids=()
slot=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r score_mode query query_env seed unprompted_flag <<<"${spec}"
  log="${LOG_ROOT}/$(log_tag "${score_mode}" "${query}" "${seed}").log"
  echo "Launch LDS eval score_mode=${score_mode} query=${query} seed=${seed} slot=${slot} -> ${log}"
  (
    run_slot "${slot}" env \
      CUDA_VISIBLE_DEVICES=0 \
      GPU_IDS=0 \
      JAX_NUM_DEVICES=1 \
      CIFAR2_ROOT="${CIFAR2_ROOT}" \
      REPO_ROOT="${REPO_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      TRAIN_SEED="${TRAIN_SEED}" \
      SAMPLE_MODEL_MODE="${score_mode}" \
      UNPROMPTED_SAMPLE_MODEL_MODE="${score_mode}" \
      ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}" \
      UNPROMPTED_SCORE_MODEL_MODE="${score_mode}" \
      QUERY="${query_env}" \
      INITIAL_SEED="${seed}" \
      LDS_SEEDS_TEXT="${LDS_SEEDS_TEXT}" \
      LDS_M="${LDS_M}" \
      LDS_K="${LDS_K}" \
      LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE}" \
      LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET}" \
      LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN}" \
      LDS_DEVICE=gpu \
      LDS_NUM_DEVICES=1 \
      LDS_SIMPLE_LOSS_NUM_MC=10 \
      DAS_DAMPING_SWEEP="${DAS_DAMPING_SWEEP:-0}" \
      DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES}" \
      UNPROMPTED="${unprompted_flag}" \
      TARGETS_TEXT="${TARGETS[*]}" \
      ALGORITHMS_TEXT="${ALGORITHMS[*]}" \
      bash -c '
        set -euo pipefail
        tag_value() {
          local value="$1"
          value="${value//,/__}"
          value="${value//+/_}"
          value="${value//[^A-Za-z0-9._-]/_}"
          while [[ "${value}" == *"__"* ]]; do value="${value//__/_}"; done
          value="${value#_}"
          value="${value%_}"
          printf "%s" "${value:-unprompted}"
        }
        damping_tag() {
          local value="$1"
          value="${value//+/_}"
          value="${value//-/neg_}"
          value="${value//./p}"
          tag_value "${value}"
        }
        score_dir_for_das_lambda() {
          local lambda="$1"
          local tag
          tag="$(damping_tag "${lambda}")"
          if [[ "${UNPROMPTED}" == "1" ]]; then
            printf "%s/result/%s/attribution_score/%s/train_seed_%s/unprompted/initial_seed_%s/das_unprompted/lambda_%s" \
              "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${SAMPLE_MODEL_MODE}" "${TRAIN_SEED}" "${INITIAL_SEED}" "${tag}"
          else
            printf "%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/das/lambda_%s" \
              "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${SAMPLE_MODEL_MODE}" "${TRAIN_SEED}" "$(tag_value "${QUERY}")" "${INITIAL_SEED}" "${tag}"
          fi
        }
        run_eval_once() {
          local score_algorithm="$1"
          local eval_algorithm="$2"
          local score_dir="${3:-}"
          local model_dir="$4"
          local target="$5"
          local extra_env=()
          if [[ -n "${score_dir}" ]]; then
            extra_env=(ATTRIBUTION_RESULT_DIRS="${score_dir}")
          fi
          if [[ "${UNPROMPTED}" == "1" ]]; then
            env "${extra_env[@]}" LDS_MODEL_DIRS="${model_dir}" "${PYTHON_BIN}" lds/run_eval.py --unprompted --algorithm "${eval_algorithm}" --lds-model-dirs "${model_dir}" --target-function "${target}"
          else
            env "${extra_env[@]}" LDS_MODEL_DIRS="${model_dir}" "${PYTHON_BIN}" lds/run_eval.py --algorithm "${eval_algorithm}" --lds-model-dirs "${model_dir}" --target-function "${target}"
          fi
        }
        for target in ${TARGETS_TEXT}; do
          for algorithm in ${ALGORITHMS_TEXT}; do
            for lds_seed in ${LDS_SEEDS_TEXT:-$(seq -s " " 0 7)}; do
              model_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/${SAMPLE_MODEL_MODE}/train_seed_${TRAIN_SEED}"
              model_pattern="${model_root}/m_${LDS_M}_k_*_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_${lds_seed}"
              mapfile -t model_matches < <(compgen -G "${model_pattern}" | sort)
              if [[ "${#model_matches[@]}" -ne 1 ]]; then
                echo "Expected exactly one LDS model dir for pattern ${model_pattern}, found ${#model_matches[@]}" >&2
                printf "  %s\n" "${model_matches[@]}" >&2
                exit 1
              fi
              model_dir="${model_matches[0]}"
              echo "[lds-eval] mode=${SAMPLE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED} target=${target} algorithm=${algorithm} lds_seed=${lds_seed} model_dir=${model_dir}"
              if [[ "${algorithm}" == "das" && "${DAS_DAMPING_SWEEP}" == "1" ]]; then
                for lambda in ${DAS_DAMPING_SWEEP_VALUES}; do
                  lambda_tag="$(damping_tag "${lambda}")"
                  score_dir="$(score_dir_for_das_lambda "${lambda}")"
                  echo "[lds-eval] DAS sweep lambda=${lambda} eval_algorithm=das_lambda_${lambda_tag} score_dir=${score_dir}"
                  run_eval_once "das" "das_lambda_${lambda_tag}" "${score_dir}" "${model_dir}" "${target}"
                done
              else
                run_eval_once "${algorithm}" "${algorithm}" "" "${model_dir}" "${target}"
              fi
            done
          done
        done
      '
  ) >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done

wait_all "${pids[@]}"

echo "Aggregating LDS evals and writing reports"
for target in "${TARGETS[@]}"; do
  for initial_seed in ${PROMPTED_SEEDS_TEXT}; do
    for query in horse automobile horse,automobile; do
      "${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
        --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/prompted_solo" \
        --queries "query_$(path_tag "${query}")" \
        --target-function "${target}" \
        --lds-m "${LDS_M}" \
        --lds-k "${LDS_K}" \
        --model-glob "m_${LDS_M}_k_*_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_*" \
        --initial-seed "${initial_seed}" \
        --prediction-dir "${PRED_TAG}" \
        --algorithms "${EVAL_ALGORITHMS[@]}" \
        --output-name "aggregate_prompted_m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_${target}_initial_seed_${initial_seed}_seeds_0_7" \
        >"${LOG_ROOT}/aggregate_prompted_$(path_tag "${query}")_${target}_seed_${initial_seed}.log" 2>&1 || true
    done
  done
  for initial_seed in ${UNPROMPTED_SEEDS_TEXT}; do
    "${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
      --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/unprompted_solo" \
      --target-function "${target}" \
      --eval-kind lds_unprompted \
      --queries unprompted \
      --lds-m "${LDS_M}" \
      --lds-k "${LDS_K}" \
      --model-glob "m_${LDS_M}_k_*_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_*" \
      --initial-seed "${initial_seed}" \
      --prediction-dir "${PRED_TAG}" \
      --algorithms "${EVAL_ALGORITHMS[@]}" \
      --output-name "aggregate_unprompted_m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_${target}_initial_seed_${initial_seed}_seeds_0_7" \
      >"${LOG_ROOT}/aggregate_unprompted_${target}_seed_${initial_seed}.log" 2>&1 || true
  done
done

"${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/summarize_lds_eval_report.py" \
  --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval" \
  --output-dir "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/reports" \
  >"${LOG_ROOT}/summary_report.log" 2>&1

echo "Job 03 complete. Report root: ${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/reports"
