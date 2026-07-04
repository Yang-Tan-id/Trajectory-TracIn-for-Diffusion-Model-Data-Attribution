#!/usr/bin/env bash
#SBATCH --job-name=cifar2-lds-eval
#SBATCH --partition=h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-lds-eval-%j.out
#SBATCH --error=cifar2-lds-eval-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-42}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-5000}"
LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION:-noise_trajectory}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/lds_eval_${SLURM_JOB_ID}"

QUERIES=("horse" "automobile" "horse,automobile")
TRAJ_RANGES=("1-2000" "2001-4000" "4001-6000" "6001-8000" "8001-10000")
ENDPOINT_ALGORITHMS=("das" "dtrak" "end_tracin")

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
fi
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

LDS_MODEL_DIRS=""
for seed in $(seq 1 16); do
  model_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/m_${LDS_M}_k_${LDS_K}_seed_${seed}"
  [[ -f "${model_dir}/lds_model_config.json" ]] || { echo "Missing ${model_dir}" >&2; exit 1; }
  grep -q '"complete": true' "${model_dir}/lds_model_config.json" || {
    echo "Incomplete LDS run: ${model_dir}" >&2; exit 1;
  }
  LDS_MODEL_DIRS+="${LDS_MODEL_DIRS:+,}${model_dir}"
done

pids=()
for query in "${QUERIES[@]}"; do
  tag="$(path_tag "${query}")"
  eval_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/query_${tag}/initial_seed_${INITIAL_SEED}"
  if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${eval_dir}" ]]; then
    echo "Refusing to overwrite ${eval_dir}" >&2
    exit 1
  fi
  for algorithm in traj_tracin "${ENDPOINT_ALGORITHMS[@]}"; do
    dirs="$(score_dirs "${query}" "${algorithm}")"
    IFS=',' read -r -a inputs <<<"${dirs}"
    for input in "${inputs[@]}"; do
      [[ -f "${input}/scores.npy" ]] || { echo "Missing ${input}/scores.npy" >&2; exit 1; }
    done
    srun --exclusive --exact \
      --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
      env EXPERIMENT_TAG="${EXPERIMENT_TAG}" QUERY="${query}" \
        INITIAL_SEED="${INITIAL_SEED}" ALGORITHMS="${algorithm}" \
        ATTRIBUTION_RESULT_DIRS="${dirs}" LDS_MODEL_DIRS="${LDS_MODEL_DIRS}" \
        LDS_DEVICE=gpu LDS_NUM_DEVICES=1 \
      bash scripts/04_lds_eval.sh --target-function "${LDS_TARGET_FUNCTION}" \
      >"${LOG_ROOT}/eval_${tag}_${algorithm}.log" 2>&1 &
    pids+=("$!")
  done
done

failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo "At least one eval failed; see ${LOG_ROOT}" >&2; exit 1; }
echo "All 12 LDS evaluations completed. Logs: ${LOG_ROOT}"
