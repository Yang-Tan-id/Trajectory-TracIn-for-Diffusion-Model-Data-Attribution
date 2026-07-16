#!/usr/bin/env bash
#SBATCH --job-name=cifar2-eval-vista
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=cifar2-eval-vista-%j.out
#SBATCH --error=cifar2-eval-vista-%j.err

set -euo pipefail

# Evaluate LDS seeds 1-16 for all three queries and four algorithms, then
# aggregate after all per-seed evaluations succeed.
#
# This job should depend on both:
#   1) 00_train_lds_50pct_vista.sh
#   2) 01_sample_and_attribute_vista.sh
#
# Example:
#   sbatch --dependency=afterok:<train_job_id>:<attr_job_id> \
#     diffusion_jax_refined/cifar2/vista/02_eval_and_aggregate_vista.sh

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-24}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-5000}"
LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 1 16)}"
LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION:-noise_trajectory}"
LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"
TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-trajectory_noise_squared_deviation_normalized}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
MAX_PARALLEL_EVAL_TASKS="${MAX_PARALLEL_EVAL_TASKS:-${SLURM_NTASKS:-16}}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/eval_seed_${INITIAL_SEED}_m${LDS_M}_k${LDS_K}_${SLURM_JOB_ID}"

QUERIES=("horse" "automobile" "horse,automobile")
TRAJ_RANGES=("1-2000" "2001-4000" "4001-6000" "6001-8000" "8001-10000")
EVAL_ALGORITHMS=("traj_tracin" "das" "dtrak" "end_tracin")

unset PYTHONPATH
if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-${SCRATCH}/conda-envs/trajectory-tracin}"
  if [[ -f "${SCRATCH}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh under \${SCRATCH}/miniforge3 or \${HOME}/miniforge3." >&2
    echo "Set ENV_SETUP to a shell snippet that activates your conda env." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_PATH}"
fi

[[ -d "${CIFAR2_ROOT}" ]] || {
  echo "CIFAR2 root not found: ${CIFAR2_ROOT}" >&2
  echo "Submit from the repository root or set REPO_ROOT explicitly." >&2
  exit 1
}
command -v python >/dev/null || { echo "python is unavailable" >&2; exit 1; }

mkdir -p "${LOG_ROOT}"
cd "${CIFAR2_ROOT}"

path_tag() {
  local value="$1"
  value="${value//,/_}"
  value="${value//+/_}"
  value="${value//[^A-Za-z0-9._-]/_}"
  while [[ "${value}" == *"__"* ]]; do value="${value//__/_}"; done
  value="${value#_}"
  value="${value%_}"
  printf '%s' "${value}"
}

traj_algorithm_tag() {
  if [[ "${TRAJ_QUERY_OBJECTIVE}" == "trajectory_noise_squared_deviation" ]]; then
    printf '%s' "traj_tracin"
  else
    local value="${TRAJ_QUERY_OBJECTIVE}"
    value="${value//[^A-Za-z0-9._-]/_}"
    while [[ "${value}" == *"__"* ]]; do value="${value//__/_}"; done
    value="${value#_}"
    value="${value%_}"
    printf 'traj_tracin_%s' "${value}"
  fi
}

prediction_tag() {
  local subset="$1"
  local sign="$2"
  local sign_text="${sign}"
  if [[ "${sign_text}" == *.0 ]]; then
    sign_text="${sign_text%.0}"
  fi
  sign_text="${sign_text//-/m}"
  sign_text="${sign_text//+/p}"
  sign_text="${sign_text//./p}"
  printf 'pred_%s_sign_%s' "${subset}" "${sign_text}"
}

PREDICTION_TAG="$(prediction_tag "${LDS_PREDICTION_SUBSET}" "${LDS_PREDICTION_SIGN}")"

score_dirs() {
  local query="$1"
  local algorithm="$2"
  local root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/attribution_score/${ATTRIBUTION_SCORE_MODEL_MODE:-${SAMPLE_MODEL_MODE:-prompted_solo}}/train_seed_${TRAIN_SEED:-42}/query_$(path_tag "${query}")/initial_seed_${INITIAL_SEED}"
  local range output=""
  if [[ "${algorithm}" == "traj_tracin" ]]; then
    for range in "${TRAJ_RANGES[@]}"; do
      output+="${output:+,}${root}/$(traj_algorithm_tag)_range_${range//-/_}"
    done
  else
    output="${root}/${algorithm}_range_1_10000"
  fi
  printf '%s' "${output}"
}

wait_eval_batch() {
  local failed=0 pid
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || {
    echo "At least one per-seed eval failed; see ${LOG_ROOT}" >&2
    exit 1
  }
}

echo "Vista gh job: CIFAR2 LDS eval + aggregate"
echo "REPO_ROOT             : ${REPO_ROOT}"
echo "CIFAR2_ROOT           : ${CIFAR2_ROOT}"
echo "EXPERIMENT_TAG        : ${EXPERIMENT_TAG}"
echo "INITIAL_SEED          : ${INITIAL_SEED}"
echo "LDS config            : m=${LDS_M}, k=${LDS_K}, seeds=${LDS_SEEDS}"
echo "TRAJ_QUERY_OBJECTIVE  : ${TRAJ_QUERY_OBJECTIVE}"
echo "TRAJ algorithm folder : $(traj_algorithm_tag)"
echo "Prediction            : ${PREDICTION_TAG}"
echo "Logs                  : ${LOG_ROOT}"

pids=()
total_launched=0
for seed in ${LDS_SEEDS}; do
  model_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/m_${LDS_M}_k_${LDS_K}_seed_${seed}"
  [[ -f "${model_dir}/lds_model_config.json" ]] || { echo "Missing ${model_dir}/lds_model_config.json" >&2; exit 1; }
  grep -q '"complete": true' "${model_dir}/lds_model_config.json" || {
    echo "Incomplete LDS model folder: ${model_dir}" >&2
    exit 1
  }

  for query in "${QUERIES[@]}"; do
    query_tag="$(path_tag "${query}")"
    for algorithm in "${EVAL_ALGORITHMS[@]}"; do
      eval_algorithm="${algorithm}"
      if [[ "${algorithm}" == "traj_tracin" ]]; then
        eval_algorithm="$(traj_algorithm_tag)"
      fi
      dirs="$(score_dirs "${query}" "${algorithm}")"
      IFS=',' read -r -a inputs <<<"${dirs}"
      for input in "${inputs[@]}"; do
        [[ -f "${input}/scores.npy" ]] || { echo "Missing attribution scores: ${input}/scores.npy" >&2; exit 1; }
      done
      out_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/${ATTRIBUTION_SCORE_MODEL_MODE:-${SAMPLE_MODEL_MODE:-prompted_solo}}/query_${query_tag}/initial_seed_${INITIAL_SEED}/lds/${eval_algorithm}/${LDS_TARGET_FUNCTION}/${PREDICTION_TAG}/$(basename "${model_dir}")"
      if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${out_dir}" ]]; then
        echo "Refusing to overwrite ${out_dir}" >&2
        exit 1
      fi

      slot="${#pids[@]}"
      echo "Launching eval seed=${seed}, query=${query}, algorithm=${eval_algorithm}, slot=${slot}"
      ibrun -n 1 -o "${slot}" \
        env CUDA_VISIBLE_DEVICES=0 \
          EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
          QUERY="${query}" \
          INITIAL_SEED="${INITIAL_SEED}" \
          ALGORITHMS="${eval_algorithm}" \
          ATTRIBUTION_RESULT_DIRS="${dirs}" \
          LDS_MODEL_DIRS="${model_dir}" \
          LDS_DEVICE=gpu \
          LDS_NUM_DEVICES=1 \
          LDS_SIMPLE_LOSS_NUM_MC=10 \
        bash scripts/04_lds_eval.sh \
          --target-function "${LDS_TARGET_FUNCTION}" \
          --prediction-subset "${LDS_PREDICTION_SUBSET}" \
          --prediction-sign "${LDS_PREDICTION_SIGN}" \
        >"${LOG_ROOT}/eval_${query_tag}_${eval_algorithm}_seed_${seed}.log" 2>&1 &
      pids+=("$!")
      total_launched=$((total_launched + 1))
      if (( ${#pids[@]} >= MAX_PARALLEL_EVAL_TASKS )); then
        wait_eval_batch "${pids[@]}"
        pids=()
      fi
    done
  done
done

if (( ${#pids[@]} > 0 )); then
  wait_eval_batch "${pids[@]}"
fi

echo "All ${total_launched} per-seed LDS evaluations completed."
echo "Aggregating per-seed LDS evaluations"
python "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
  --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval" \
  --target-function "${LDS_TARGET_FUNCTION}" \
  --lds-m "${LDS_M}" \
  --lds-k "${LDS_K}" \
  --initial-seed "${INITIAL_SEED}" \
  --algorithms "$(traj_algorithm_tag)" das dtrak end_tracin \
  --prediction-dir "${PREDICTION_TAG}" \
  --output-name "aggregate_m_${LDS_M}_k_${LDS_K}_${PREDICTION_TAG}_normalized_traj_seed_${INITIAL_SEED}_seeds_${LDS_SEEDS// /_}" \
  >"${LOG_ROOT}/aggregate.log" 2>&1

echo "Aggregate log: ${LOG_ROOT}/aggregate.log"
echo "Eval and aggregate completed successfully."
