#!/usr/bin/env bash
#SBATCH --job-name=cifar2-attr-eval
#SBATCH --partition=h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-attr-eval-%j.out
#SBATCH --error=cifar2-attr-eval-%j.err

set -euo pipefail

# This job must run after 01_train_lds_h100.sh because its final phase reads
# all 16 LDS model folders. Recommended submission:
#
#   lds_job=$(sbatch --parsable -A <allocation> \
#     diffusion_jax_refined/cifar2/tacc/01_train_lds_h100.sh)
#   sbatch -A <allocation> --dependency=afterok:${lds_job} \
#     diffusion_jax_refined/cifar2/tacc/02_sample_attribution_eval_h100.sh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-42}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-5000}"
LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION:-noise_trajectory}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/attr_eval_${SLURM_JOB_ID}"
MAX_PARALLEL=16

# CIFAR multi-label queries are comma-separated. This represents the requested
# "horse + automobile" query in the format accepted by the model.
QUERIES=("horse" "automobile" "horse,automobile")
TRAJ_RANGES=("1-2000" "2001-4000" "4001-6000" "6001-8000" "8001-10000")
ENDPOINT_ALGORITHMS=("das" "dtrak" "end_tracin")

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
fi
command -v python >/dev/null || {
  echo "python is not available; activate the project environment or set ENV_SETUP." >&2
  exit 1
}

mkdir -p "${LOG_ROOT}"
cd "${CIFAR2_ROOT}"

query_tag() {
  local value="$1"
  value="${value//,/__}"
  value="${value//[^A-Za-z0-9._-]/_}"
  printf '%s' "${value}"
}

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

attribution_root_for_query() {
  printf '%s/result/%s/attribution_score/query_%s/initial_seed_%s' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "$(path_tag "$1")" "${INITIAL_SEED}"
}

score_dirs_for_algorithm() {
  local query="$1"
  local algorithm="$2"
  local root range output=""
  root="$(attribution_root_for_query "${query}")"
  if [[ "${algorithm}" == "traj_tracin" ]]; then
    for range in "${TRAJ_RANGES[@]}"; do
      output+="${output:+,}${root}/traj_tracin_range_${range//-/_}"
    done
  else
    output="${root}/${algorithm}_range_1_10000"
  fi
  printf '%s' "${output}"
}

wait_batch() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "A parallel task failed. Check ${LOG_ROOT}." >&2
    exit 1
  fi
}

launch_sample() {
  local query="$1"
  local tag
  tag="$(query_tag "${query}")"
  echo "Sampling query=${query}, initial_seed=${INITIAL_SEED}"
  srun --exclusive --exact \
    --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    env \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      QUERY="${query}" \
      SAMPLE_SEEDS="${INITIAL_SEED}" \
    bash scripts/00_sample.sh >"${LOG_ROOT}/sample_${tag}.log" 2>&1 &
}

launch_attribution() {
  local query="$1"
  local algorithm="$2"
  local range="${3:-}"
  local tag suffix
  tag="$(query_tag "${query}")"
  suffix="${algorithm}"
  if [[ -n "${range}" ]]; then
    suffix="${suffix}_${range//-/_}"
  fi
  echo "Attribution query=${query}, algorithm=${algorithm}, range=${range:-all}"
  srun --exclusive --exact \
    --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    env \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      QUERY="${query}" \
      INITIAL_SEED="${INITIAL_SEED}" \
      ALGORITHMS="${algorithm}" \
      ATTRIBUTION_RANGES="${range}" \
    bash scripts/01_data_attribution.sh >"${LOG_ROOT}/attr_${tag}_${suffix}.log" 2>&1 &
}

launch_eval() {
  local query="$1"
  local algorithm="$2"
  local tag score_dirs
  tag="$(query_tag "${query}")"
  score_dirs="$(score_dirs_for_algorithm "${query}" "${algorithm}")"
  echo "LDS eval query=${query}, algorithm=${algorithm}"
  srun --exclusive --exact \
    --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    env \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      QUERY="${query}" \
      INITIAL_SEED="${INITIAL_SEED}" \
      ALGORITHMS="${algorithm}" \
      ATTRIBUTION_RESULT_DIRS="${score_dirs}" \
      LDS_MODEL_DIRS="${LDS_MODEL_DIRS}" \
      LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION}" \
      LDS_DEVICE=gpu \
      LDS_NUM_DEVICES=1 \
    bash scripts/04_lds_eval.sh \
      --target-function "${LDS_TARGET_FUNCTION}" \
      >"${LOG_ROOT}/eval_${tag}_${algorithm}.log" 2>&1 &
}

if [[ "${ALLOW_OVERWRITE}" != "1" ]]; then
  for query in "${QUERIES[@]}"; do
    tag="$(path_tag "${query}")"
    sample_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/sampling/cifar/prompt_${tag}/model_prompted_jax__ckpt_seed_42_epoch_0200/seed_$(printf '%06d' "${INITIAL_SEED}")"
    attribution_dir="$(attribution_root_for_query "${query}")"
    eval_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/query_${tag}/initial_seed_${INITIAL_SEED}"
    for existing in "${sample_dir}" "${attribution_dir}" "${eval_dir}"; do
      if [[ -e "${existing}" ]]; then
        echo "Refusing to overwrite existing output: ${existing}" >&2
        echo "Use a new experiment/seed, archive the old output, or set ALLOW_OVERWRITE=1." >&2
        exit 1
      fi
    done
  done
fi

echo "Phase 1/3: sampling three queries"
pids=()
for query in "${QUERIES[@]}"; do
  launch_sample "${query}"
  pids+=("$!")
done
wait_batch "${pids[@]}"

echo "Phase 2/3: attribution (24 GPU tasks in batches of at most ${MAX_PARALLEL})"
pids=()
for query in "${QUERIES[@]}"; do
  for range in "${TRAJ_RANGES[@]}"; do
    launch_attribution "${query}" traj_tracin "${range}"
    pids+=("$!")
    if (( ${#pids[@]} == MAX_PARALLEL )); then
      wait_batch "${pids[@]}"
      pids=()
    fi
  done
  for algorithm in "${ENDPOINT_ALGORITHMS[@]}"; do
    launch_attribution "${query}" "${algorithm}"
    pids+=("$!")
    if (( ${#pids[@]} == MAX_PARALLEL )); then
      wait_batch "${pids[@]}"
      pids=()
    fi
  done
done
if (( ${#pids[@]} )); then
  wait_batch "${pids[@]}"
fi

LDS_MODEL_DIRS=""
for seed in $(seq 1 16); do
  model_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/m_${LDS_M}_k_${LDS_K}_seed_${seed}"
  if [[ ! -f "${model_dir}/lds_model_config.json" ]]; then
    echo "Missing LDS model folder or config: ${model_dir}" >&2
    exit 1
  fi
  if ! grep -q '"complete": true' "${model_dir}/lds_model_config.json"; then
    echo "LDS model run is incomplete: ${model_dir}" >&2
    exit 1
  fi
  LDS_MODEL_DIRS+="${LDS_MODEL_DIRS:+,}${model_dir}"
done
export LDS_MODEL_DIRS

for query in "${QUERIES[@]}"; do
  for algorithm in traj_tracin "${ENDPOINT_ALGORITHMS[@]}"; do
    IFS=',' read -r -a expected_score_dirs <<<"$(score_dirs_for_algorithm "${query}" "${algorithm}")"
    for score_dir in "${expected_score_dirs[@]}"; do
      if [[ ! -f "${score_dir}/scores.npy" ]]; then
        echo "Missing expected attribution score file: ${score_dir}/scores.npy" >&2
        exit 1
      fi
    done
  done
done

echo "Phase 3/3: evaluating four algorithms for all three queries"
pids=()
for query in "${QUERIES[@]}"; do
  for algorithm in traj_tracin "${ENDPOINT_ALGORITHMS[@]}"; do
    launch_eval "${query}" "${algorithm}"
    pids+=("$!")
  done
done
wait_batch "${pids[@]}"

echo "Sampling, attribution, and LDS evaluation completed successfully."
echo "Logs: ${LOG_ROOT}"
