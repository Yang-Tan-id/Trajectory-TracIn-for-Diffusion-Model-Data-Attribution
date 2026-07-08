#!/usr/bin/env bash
#SBATCH --job-name=cifar2-lds-eval-seed
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=cifar2-lds-eval-seed-%j.out
#SBATCH --error=cifar2-lds-eval-seed-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-42}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-5000}"
LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 1 16)}"
LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION:-noise_trajectory}"
LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
MAX_PARALLEL_EVAL_TASKS="${MAX_PARALLEL_EVAL_TASKS:-${SLURM_NTASKS:-4}}"

QUERIES=("horse" "automobile" "horse,automobile")
TRAJ_RANGES=("1-2000" "2001-4000" "4001-6000" "6001-8000" "8001-10000")
ENDPOINT_ALGORITHMS=("das" "dtrak" "end_tracin")

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
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/lds_eval_by_seed_${PREDICTION_TAG}_${SLURM_JOB_ID}"

unset PYTHONPATH
if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-${SCRATCH}/conda-envs/trajectory-tracin}"
  # shellcheck disable=SC1090
  source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi
[[ -d "${CIFAR2_ROOT}" ]] || {
  echo "CIFAR2 root not found: ${CIFAR2_ROOT}" >&2
  echo "Submit this job from the repository root or set REPO_ROOT explicitly." >&2
  exit 1
}
command -v python >/dev/null || { echo "python is unavailable" >&2; exit 1; }
mkdir -p "${LOG_ROOT}"
cd "${CIFAR2_ROOT}"

path_tag() {
  local value="${1//,/_}"
  value="${value//[^A-Za-z0-9._-]/_}"
  while [[ "${value}" == *"__"* ]]; do value="${value//__/_}"; done
  printf '%s' "${value#_}" | sed 's/_$//'
}

score_dirs() {
  local query="$1" algorithm="$2" range output=""
  local root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/attribution_score/query_$(path_tag "${query}")/initial_seed_${INITIAL_SEED}"
  if [[ "${algorithm}" == "traj_tracin" ]]; then
    for range in "${TRAJ_RANGES[@]}"; do
      output+="${output:+,}${root}/traj_tracin_range_${range//-/_}"
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
  (( failed == 0 )) || { echo "At least one per-seed eval failed; see ${LOG_ROOT}" >&2; exit 1; }
}

pids=()
total_launched=0
for seed in ${LDS_SEEDS}; do
  model_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/m_${LDS_M}_k_${LDS_K}_seed_${seed}"
  [[ -f "${model_dir}/lds_model_config.json" ]] || { echo "Missing ${model_dir}" >&2; exit 1; }
  grep -q '"complete": true' "${model_dir}/lds_model_config.json" || {
    echo "Incomplete LDS run: ${model_dir}" >&2; exit 1;
  }

  for query in "${QUERIES[@]}"; do
    tag="$(path_tag "${query}")"
    for algorithm in traj_tracin "${ENDPOINT_ALGORITHMS[@]}"; do
      dirs="$(score_dirs "${query}" "${algorithm}")"
      IFS=',' read -r -a inputs <<<"${dirs}"
      for input in "${inputs[@]}"; do
        [[ -f "${input}/scores.npy" ]] || { echo "Missing ${input}/scores.npy" >&2; exit 1; }
      done
      out_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/query_${tag}/initial_seed_${INITIAL_SEED}/lds/${algorithm}/${LDS_TARGET_FUNCTION}/${PREDICTION_TAG}/$(basename "${model_dir}")"
      if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${out_dir}" ]]; then
        echo "Refusing to overwrite ${out_dir}" >&2
        exit 1
      fi

      slot="${#pids[@]}"
      echo "Launching per-seed LDS eval seed=${seed}, query=${query}, algorithm=${algorithm}, prediction=${PREDICTION_TAG}"
      ibrun -n 1 -o "${slot}" \
        env CUDA_VISIBLE_DEVICES="$((slot % 4))" \
          EXPERIMENT_TAG="${EXPERIMENT_TAG}" QUERY="${query}" \
          INITIAL_SEED="${INITIAL_SEED}" ALGORITHMS="${algorithm}" \
          ATTRIBUTION_RESULT_DIRS="${dirs}" LDS_MODEL_DIRS="${model_dir}" \
          LDS_DEVICE=gpu LDS_NUM_DEVICES=1 \
        bash scripts/04_lds_eval.sh --target-function "${LDS_TARGET_FUNCTION}" \
        --prediction-subset "${LDS_PREDICTION_SUBSET}" \
        --prediction-sign "${LDS_PREDICTION_SIGN}" \
        >"${LOG_ROOT}/eval_${tag}_${algorithm}_lds_seed_${seed}.log" 2>&1 &
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
echo "All ${total_launched} per-seed LDS evaluations completed. Logs: ${LOG_ROOT}"

echo "Aggregating per-seed LDS evaluations"
python "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
  --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval" \
  --target-function "${LDS_TARGET_FUNCTION}" \
  --lds-m "${LDS_M}" \
  --lds-k "${LDS_K}" \
  --initial-seed "${INITIAL_SEED}" \
  --algorithms traj_tracin "${ENDPOINT_ALGORITHMS[@]}" \
  --output-name "aggregate_m_${LDS_M}_k_${LDS_K}_${PREDICTION_TAG}_seeds_${LDS_SEEDS// /_}" \
  >"${LOG_ROOT}/aggregate.log" 2>&1
echo "Aggregate log: ${LOG_ROOT}/aggregate.log"
