#!/usr/bin/env bash
#SBATCH -J cifar2-s3-das-eval
#SBATCH -o cifar2-s3-das-eval-%j.out
#SBATCH -e cifar2-s3-das-eval-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
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

TARGETS=(simple_loss noise_trajectory)
LDS_SEEDS_TEXT="${LDS_SEEDS:-$(seq -s ' ' 0 7)}"
LDS_M="${LDS_M:-64}"
LDS_K="${LDS_K:-5000}"
LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-50}"
LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"
PRED_TAG="${PRED_TAG:-pred_kept_sign_m1}"
PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 7)}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 23)}"
DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES:-0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000}"
TARGETS_TEXT="${TARGETS[*]}"
export PROMPTED_SEEDS_TEXT UNPROMPTED_SEEDS_TEXT LDS_SEEDS_TEXT TARGETS_TEXT DAS_DAMPING_SWEEP_VALUES
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/03_lds_eval_report/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

EVAL_ALGORITHMS=()
for lambda in ${DAS_DAMPING_SWEEP_VALUES}; do
  EVAL_ALGORITHMS+=("das_lambda_$(damping_tag "${lambda}")")
done

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

echo "Job 03 Stampede3 DAS: LDS eval + aggregate/report"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; query_tasks=${#SPECS[@]}; nodes=4; gpu_slots=16; queries_per_gpu=3"
echo "targets=${TARGETS[*]}; lds_seeds=${LDS_SEEDS_TEXT}; das_lambdas=${DAS_DAMPING_SWEEP_VALUES}"
echo "eval_algorithms=${EVAL_ALGORITHMS[*]}; logs=${LOG_ROOT}"

pids=()
for slot in $(seq 0 15); do
  gpu=$((slot % 4))
  log="${LOG_ROOT}/slot_${slot}.log"
  echo "Launch eval slot=${slot} gpu=${gpu}; handles query indices ${slot}, $((slot + 16)), $((slot + 32)) -> ${log}"
  (
    run_gpu_slot "${slot}" env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      GPU_IDS="${gpu}" \
      JAX_NUM_DEVICES=1 \
      CIFAR2_ROOT="${CIFAR2_ROOT}" \
      REPO_ROOT="${REPO_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      TRAIN_SEED="${TRAIN_SEED}" \
      SLOT_INDEX="${slot}" \
      LDS_M="${LDS_M}" \
      LDS_K="${LDS_K}" \
      LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE}" \
      LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET}" \
      LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN}" \
      LDS_DEVICE=gpu \
      LDS_NUM_DEVICES=1 \
      LDS_SIMPLE_LOSS_NUM_MC=10 \
      bash "${SCRIPT_DIR}/03_das_lds_eval_slot_stampede3.sh"
  ) >"${log}" 2>&1 &
  pids+=("$!")
done

wait_all "${pids[@]}"

echo "Aggregating Stampede3 DAS LDS evals and writing reports"
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
        --output-name "aggregate_stampede3_das_m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_${target}_initial_seed_${initial_seed}_seeds_0_7" \
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
      --output-name "aggregate_stampede3_das_unprompted_m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_${target}_initial_seed_${initial_seed}_seeds_0_7" \
      >"${LOG_ROOT}/aggregate_unprompted_${target}_seed_${initial_seed}.log" 2>&1 || true
  done
done

"${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/summarize_lds_eval_report.py" \
  --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval" \
  --output-dir "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/reports" \
  >"${LOG_ROOT}/summary_report.log" 2>&1

echo "Job 03 complete. Report root: ${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/reports"
