#!/usr/bin/env bash
#SBATCH --job-name=cifar2-lds-eval-report
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=21
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=cifar2-lds-eval-report-%j.out
#SBATCH --error=cifar2-lds-eval-report-%j.err

set -euo pipefail

SCRIPT_DIR="${VISTA_PIPELINE_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_vista_pipeline_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/vista"; do
    if [[ -n "${candidate}" && -f "${candidate}/_vista_pipeline_lib.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_vista_pipeline_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_vista_pipeline_lib.sh
source "${SCRIPT_DIR}/_vista_pipeline_lib.sh"
vista_init

ALGORITHMS=(dtrak das end_tracin traj_tracin)
LDS_SEEDS_TEXT="${LDS_SEEDS:-$(seq -s ' ' 1 16)}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-5000}"
LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-50}"
LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION:-noise_trajectory}"
LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"
PRED_TAG="${PRED_TAG:-pred_kept_sign_m1}"
SPECS=(
  "unprompted_solo|unprompted|24" "unprompted_solo|unprompted|48" "unprompted_solo|unprompted|96"
  "unprompted_multi|unprompted|24" "unprompted_multi|unprompted|48" "unprompted_multi|unprompted|96"
  "prompted_solo|horse|24" "prompted_solo|horse|48" "prompted_solo|horse|96"
  "prompted_solo|automobile|24" "prompted_solo|automobile|48" "prompted_solo|automobile|96"
  "prompted_multi|horse|24" "prompted_multi|horse|48" "prompted_multi|horse|96"
  "prompted_multi|automobile|24" "prompted_multi|automobile|48" "prompted_multi|automobile|96"
  "prompted_multi|horse,automobile|24" "prompted_multi|horse,automobile|48" "prompted_multi|horse,automobile|96"
)
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/05_lds_eval_report/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

echo "Job 05: LDS eval + aggregate/report"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; lds_seeds=${LDS_SEEDS_TEXT}; algorithms=${ALGORITHMS[*]}"
echo "query tasks=${#SPECS[@]}; logs=${LOG_ROOT}"

pids=()
slot=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r mode query seed <<<"${spec}"
  log="${LOG_ROOT}/$(log_tag "${mode}" "${query}" "${seed}").log"
  echo "Launch LDS eval mode=${mode} query=${query} seed=${seed} slot=${slot} -> ${log}"
  (
    run_slot "${slot}" env \
      CUDA_VISIBLE_DEVICES=0 \
      GPU_IDS=0 \
      JAX_NUM_DEVICES=1 \
      CIFAR2_ROOT="${CIFAR2_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      TRAIN_SEED="${TRAIN_SEED}" \
      SAMPLE_MODEL_MODE="${mode}" \
      UNPROMPTED_SAMPLE_MODEL_MODE="${mode}" \
      ATTRIBUTION_SCORE_MODEL_MODE="${mode}" \
      UNPROMPTED_SCORE_MODEL_MODE="${mode}" \
      QUERY="$([[ "${query}" == "unprompted" ]] && printf 'unconditional' || printf '%s' "${query}")" \
      INITIAL_SEED="${seed}" \
      LDS_SEEDS_TEXT="${LDS_SEEDS_TEXT}" \
      LDS_M="${LDS_M}" \
      LDS_K="${LDS_K}" \
      LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE}" \
      LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION}" \
      LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET}" \
      LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN}" \
      LDS_DEVICE=gpu \
      LDS_NUM_DEVICES=1 \
      LDS_SIMPLE_LOSS_NUM_MC=10 \
      bash -c '
        set -euo pipefail
        for algorithm in dtrak das end_tracin traj_tracin; do
          for lds_seed in ${LDS_SEEDS_TEXT:-$(seq -s " " 1 16)}; do
            model_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/${SAMPLE_MODEL_MODE}/train_seed_${TRAIN_SEED}/m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_${lds_seed}"
            echo "[lds-eval] mode=${SAMPLE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED} algorithm=${algorithm} lds_seed=${lds_seed} model_dir=${model_dir}"
            if [[ "${SAMPLE_MODEL_MODE}" == unprompted_* ]]; then
              LDS_MODEL_DIRS="${model_dir}" "${PYTHON_BIN}" lds/run_eval.py --unprompted --algorithm "${algorithm}" --lds-model-dirs "${model_dir}" --target-function "${LDS_TARGET_FUNCTION}"
            else
              LDS_MODEL_DIRS="${model_dir}" "${PYTHON_BIN}" lds/run_eval.py --algorithm "${algorithm}" --lds-model-dirs "${model_dir}" --target-function "${LDS_TARGET_FUNCTION}"
            fi
          done
        done
      '
  ) >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done

wait_all "${pids[@]}"

echo "Aggregating LDS evals and writing reports"
for mode in prompted_solo prompted_multi; do
  for initial_seed in 24 48 96; do
    "${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
      --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}" \
      --target-function "${LDS_TARGET_FUNCTION}" \
      --lds-m "${LDS_M}" \
      --lds-k "${LDS_K}" \
      --model-glob "m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_*" \
      --initial-seed "${initial_seed}" \
      --prediction-dir "${PRED_TAG}" \
      --algorithms "${ALGORITHMS[@]}" \
      --output-name "aggregate_${mode}_m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_initial_seed_${initial_seed}_seeds_1_16" \
      >"${LOG_ROOT}/aggregate_${mode}_seed_${initial_seed}.log" 2>&1 || true
  done
done
for mode in unprompted_solo unprompted_multi; do
  for initial_seed in 24 48 96; do
    "${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
      --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${mode}" \
      --target-function "${LDS_TARGET_FUNCTION}" \
      --eval-kind lds_unprompted \
      --queries unprompted \
      --lds-m "${LDS_M}" \
      --lds-k "${LDS_K}" \
      --model-glob "m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_*" \
      --initial-seed "${initial_seed}" \
      --prediction-dir "${PRED_TAG}" \
      --algorithms "${ALGORITHMS[@]}" \
      --output-name "aggregate_${mode}_m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_initial_seed_${initial_seed}_seeds_1_16" \
      >"${LOG_ROOT}/aggregate_${mode}_seed_${initial_seed}.log" 2>&1 || true
  done
done

"${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/summarize_lds_eval_report.py" \
  --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval" \
  --output-dir "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/reports" \
  >"${LOG_ROOT}/summary_report.log" 2>&1

echo "Job 05 complete. Report root: ${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/reports"
