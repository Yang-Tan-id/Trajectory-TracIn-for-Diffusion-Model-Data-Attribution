#!/usr/bin/env bash
#SBATCH --job-name=cifar2-simple-loss-rtx
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-simple-loss-rtx-%j.out
#SBATCH --error=cifar2-simple-loss-rtx-%j.err

set -euo pipefail

# Run CIFAR2 simple_loss LDS eval on one rtx-small node, then export red/green
# scatter SVGs and a SUMMARY.txt in the same style as simple_loss_scatter_exports_red_green.
#
# Submit from the repository root:
#   sbatch diffusion_jax_refined/cifar2/tacc/12_simple_loss_eval_rtxsmall_1node.sh

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-42}"

LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-8000}"
LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 1 16)}"
LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION:-simple_loss}"
LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"
LDS_SIMPLE_LOSS_NUM_MC="${LDS_SIMPLE_LOSS_NUM_MC:-16}"
LDS_SIMPLE_LOSS_MC_SEED="${LDS_SIMPLE_LOSS_MC_SEED:-0}"

MAX_PARALLEL_EVAL_TASKS="${MAX_PARALLEL_EVAL_TASKS:-${SLURM_NTASKS:-2}}"
GPU_PER_NODE="${GPU_PER_NODE:-2}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

QUERIES=("horse" "automobile" "horse,automobile")
ALGORITHMS=("traj_tracin" "das" "dtrak" "end_tracin")
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/simple_loss_k${LDS_K}_m${LDS_M}_${SLURM_JOB_ID}"
EXPORT_DIR="${EXPORT_DIR:-${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/simple_loss_scatter_exports_red_green_${SLURM_JOB_ID}}"

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
mkdir -p "${LOG_ROOT}" "${EXPORT_DIR}"
cd "${CIFAR2_ROOT}"

path_tag() {
  local value="$1"
  value="${value//,/_}"
  value="${value//[^A-Za-z0-9._-]/_}"
  while [[ "${value}" == *"__"* ]]; do
    value="${value//__/_}"
  done
  value="${value#_}"
  value="${value%_}"
  printf '%s' "${value}"
}

query_tag() {
  local value="$1"
  value="${value//,/__}"
  value="${value//[^A-Za-z0-9._-]/_}"
  printf '%s' "${value}"
}

prediction_tag() {
  local subset="$1"
  local sign="$2"
  local sign_text
  sign_text="$(python - "${sign}" <<'PY'
import sys
x = float(sys.argv[1])
print(str(int(x)) if x.is_integer() else f"{x:g}")
PY
)"
  sign_text="${sign_text//-/m}"
  sign_text="${sign_text//+/p}"
  sign_text="${sign_text//./p}"
  printf 'pred_%s_sign_%s' "${subset}" "${sign_text}"
}

lds_model_dir_for_seed() {
  local seed="$1"
  printf '%s/result/%s/lds_model/m_%s_k_%s_seed_%s' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${LDS_M}" "${LDS_K}" "${seed}"
}

eval_out_dir() {
  local query="$1"
  local algorithm="$2"
  local seed="$3"
  local pred_tag
  pred_tag="$(prediction_tag "${LDS_PREDICTION_SUBSET}" "${LDS_PREDICTION_SIGN}")"
  printf '%s/result/%s/eval/query_%s/initial_seed_%s/lds/%s/%s/%s/m_%s_k_%s_seed_%s' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "$(path_tag "${query}")" "${INITIAL_SEED}" \
    "${algorithm}" "${LDS_TARGET_FUNCTION}" "${pred_tag}" "${LDS_M}" "${LDS_K}" "${seed}"
}

default_score_dirs() {
  local query="$1"
  local algorithm="$2"
  local tag
  tag="$(path_tag "${query}")"
  local root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/attribution_score/query_${tag}/initial_seed_${INITIAL_SEED}"
  if [[ "${algorithm}" == "traj_tracin" ]]; then
    printf '%s/traj_tracin_range_6001_8000' "${root}"
  else
    printf '%s/%s' "${root}" "${algorithm}"
  fi
}

score_dirs_for() {
  local query="$1"
  local algorithm="$2"
  local tag var_name override
  tag="$(path_tag "${query}")"
  var_name="SCORE_DIRS_${algorithm}_${tag}"
  override="${!var_name:-}"
  if [[ -n "${override}" ]]; then
    printf '%s' "${override}"
  else
    default_score_dirs "${query}" "${algorithm}"
  fi
}

wait_batch() {
  local failed=0 pid
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || {
    echo "At least one parallel task failed; see ${LOG_ROOT}" >&2
    exit 1
  }
}

validate_lds_models() {
  echo "Validating LDS models M=${LDS_M}, K=${LDS_K}, seeds=${LDS_SEEDS}"
  local seed model_dir
  for seed in ${LDS_SEEDS}; do
    model_dir="$(lds_model_dir_for_seed "${seed}")"
    [[ -f "${model_dir}/lds_model_config.json" ]] || { echo "Missing ${model_dir}/lds_model_config.json" >&2; exit 1; }
    grep -q '"complete": true' "${model_dir}/lds_model_config.json" || {
      echo "Incomplete LDS model folder: ${model_dir}" >&2
      exit 1
    }
  done
}

validate_scores() {
  echo "Validating attribution score folders"
  local query algorithm dirs input
  for query in "${QUERIES[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
      dirs="$(score_dirs_for "${query}" "${algorithm}")"
      IFS=',' read -r -a inputs <<<"${dirs}"
      for input in "${inputs[@]}"; do
        [[ -f "${input}/scores.npy" ]] || { echo "Missing attribution scores: ${input}/scores.npy" >&2; exit 1; }
      done
    done
  done
}

run_eval() {
  echo "Running simple_loss LDS eval"
  local pids=()
  local query algorithm seed model_dir dirs out_dir slot tag
  for query in "${QUERIES[@]}"; do
    tag="$(query_tag "${query}")"
    for algorithm in "${ALGORITHMS[@]}"; do
      dirs="$(score_dirs_for "${query}" "${algorithm}")"
      for seed in ${LDS_SEEDS}; do
        model_dir="$(lds_model_dir_for_seed "${seed}")"
        out_dir="$(eval_out_dir "${query}" "${algorithm}" "${seed}")"
        if [[ "${ALLOW_OVERWRITE}" != "1" && -f "${out_dir}/lds_results.csv" ]]; then
          echo "Skip existing eval: ${out_dir}"
          continue
        fi
        slot="${#pids[@]}"
        echo "Launch eval query=${query} algorithm=${algorithm} seed=${seed} slot=${slot}"
        ibrun -n 1 -o "${slot}" \
          env CUDA_VISIBLE_DEVICES="$((slot % GPU_PER_NODE))" \
            EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
            QUERY="${query}" \
            INITIAL_SEED="${INITIAL_SEED}" \
            ALGORITHMS="${algorithm}" \
            ATTRIBUTION_RESULT_DIRS="${dirs}" \
            LDS_MODEL_DIRS="${model_dir}" \
            LDS_DEVICE=gpu \
            LDS_NUM_DEVICES=1 \
            LDS_SIMPLE_LOSS_NUM_MC="${LDS_SIMPLE_LOSS_NUM_MC}" \
            LDS_SIMPLE_LOSS_MC_SEED="${LDS_SIMPLE_LOSS_MC_SEED}" \
          bash scripts/04_lds_eval.sh \
            --target-function "${LDS_TARGET_FUNCTION}" \
            --prediction-subset "${LDS_PREDICTION_SUBSET}" \
            --prediction-sign "${LDS_PREDICTION_SIGN}" \
          >"${LOG_ROOT}/eval_${tag}_${algorithm}_seed_${seed}.log" 2>&1 &
        pids+=("$!")
        if (( ${#pids[@]} >= MAX_PARALLEL_EVAL_TASKS )); then
          wait_batch "${pids[@]}"
          pids=()
        fi
      done
    done
  done
  if (( ${#pids[@]} > 0 )); then
    wait_batch "${pids[@]}"
  fi
}

run_export() {
  local pred_tag
  pred_tag="$(prediction_tag "${LDS_PREDICTION_SUBSET}" "${LDS_PREDICTION_SIGN}")"
  python "${REPO_ROOT}/diffusion_jax_refined/common/export_lds_scatter_red_green.py" \
    --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval" \
    --target-function "${LDS_TARGET_FUNCTION}" \
    --prediction-dir "${pred_tag}" \
    --out-dir "${EXPORT_DIR}" \
    --clean \
    >"${LOG_ROOT}/export_red_green.log" 2>&1
  echo "Exported red/green scatter SVGs to ${EXPORT_DIR}"
}

echo "CIFAR2 simple_loss eval on rtx-small"
echo "CIFAR2_ROOT=${CIFAR2_ROOT}"
echo "EXPERIMENT_TAG=${EXPERIMENT_TAG}; INITIAL_SEED=${INITIAL_SEED}"
echo "LDS_M=${LDS_M}; LDS_K=${LDS_K}; LDS_SEEDS=${LDS_SEEDS}"
echo "LDS_SIMPLE_LOSS_NUM_MC=${LDS_SIMPLE_LOSS_NUM_MC}; LDS_SIMPLE_LOSS_MC_SEED=${LDS_SIMPLE_LOSS_MC_SEED}"
echo "MAX_PARALLEL_EVAL_TASKS=${MAX_PARALLEL_EVAL_TASKS}; GPU_PER_NODE=${GPU_PER_NODE}"
echo "Logs: ${LOG_ROOT}"

validate_lds_models
validate_scores
run_eval
run_export

echo "Done. Logs are in ${LOG_ROOT}"
