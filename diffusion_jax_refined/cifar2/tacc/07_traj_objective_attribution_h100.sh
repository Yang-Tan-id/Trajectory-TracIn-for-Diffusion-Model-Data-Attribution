#!/usr/bin/env bash
#SBATCH --job-name=cifar2-traj-obj
#SBATCH --partition=h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=30:00:00
#SBATCH --output=cifar2-traj-obj-%j.out
#SBATCH --error=cifar2-traj-obj-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-42}"
TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:?Set TRAJ_QUERY_OBJECTIVE=eps_deviation_l1_mean or eps_deviation_l2_sq_mean}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/traj_objective_${TRAJ_QUERY_OBJECTIVE}_${SLURM_JOB_ID}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-${SLURM_NTASKS:-16}}"

QUERIES=("horse" "automobile" "horse,automobile")
TRAJ_RANGES=("1-2000" "2001-4000" "4001-6000" "6001-8000" "8001-10000")

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

traj_algorithm_tag() {
  if [[ "${TRAJ_QUERY_OBJECTIVE}" == "trajectory_noise_squared_deviation" ]]; then
    printf '%s' "traj_tracin"
  else
    local value="${TRAJ_QUERY_OBJECTIVE}"
    value="${value//[^A-Za-z0-9._-]/_}"
    while [[ "${value}" == *"__"* ]]; do
      value="${value//__/_}"
    done
    value="${value#_}"
    value="${value%_}"
    printf 'traj_tracin_%s' "${value}"
  fi
}

sample_dir_for_query() {
  local query="$1"
  local tag
  tag="$(path_tag "${query}")"
  printf '%s/result/%s/eval/sampling/cifar/prompt_%s/model_prompted_jax__ckpt_seed_42_epoch_0200/seed_%06d' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${tag}" "${INITIAL_SEED}"
}

wait_batch() {
  local failed=0 pid
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || { echo "At least one traj objective task failed; see ${LOG_ROOT}" >&2; exit 1; }
}

launch_attribution() {
  local query="$1"
  local range="$2"
  local slot="$3"
  local tag
  tag="$(query_tag "${query}")"
  echo "Attribution objective=${TRAJ_QUERY_OBJECTIVE}, query=${query}, range=${range}"
  ibrun -n 1 -o "${slot}" \
    env CUDA_VISIBLE_DEVICES="$((slot % 4))" \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      QUERY="${query}" \
      INITIAL_SEED="${INITIAL_SEED}" \
      TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE}" \
      ALGORITHMS="traj_tracin" \
      ATTRIBUTION_RANGES="${range}" \
    bash scripts/01_data_attribution.sh >"${LOG_ROOT}/attr_${tag}_${range//-/_}.log" 2>&1 &
}

for query in "${QUERIES[@]}"; do
  sample_dir="$(sample_dir_for_query "${query}")"
  [[ -f "${sample_dir}/trajectory_xt.npy" ]] || { echo "Missing sample trajectory: ${sample_dir}" >&2; exit 1; }
done

pids=()
for query in "${QUERIES[@]}"; do
  for range in "${TRAJ_RANGES[@]}"; do
    output_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/attribution_score/query_$(path_tag "${query}")/initial_seed_${INITIAL_SEED}/$(traj_algorithm_tag)_range_${range//-/_}"
    if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${output_dir}" ]]; then
      echo "Refusing to overwrite ${output_dir}" >&2
      exit 1
    fi
    slot="${#pids[@]}"
    launch_attribution "${query}" "${range}" "${slot}"
    pids+=("$!")
    if (( ${#pids[@]} >= MAX_PARALLEL_TASKS )); then
      wait_batch "${pids[@]}"
      pids=()
    fi
  done
done

if (( ${#pids[@]} > 0 )); then
  wait_batch "${pids[@]}"
fi

echo "Completed traj objective attribution: ${TRAJ_QUERY_OBJECTIVE}"
echo "Score folder tag: $(traj_algorithm_tag)"
echo "Logs: ${LOG_ROOT}"
