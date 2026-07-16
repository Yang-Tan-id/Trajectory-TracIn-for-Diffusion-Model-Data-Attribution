#!/usr/bin/env bash
#SBATCH --job-name=cifar2-attr-norm-vista
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=cifar2-attr-norm-vista-%j.out
#SBATCH --error=cifar2-attr-norm-vista-%j.err

set -euo pipefail

# Phase 1: sample query trajectories for initial seed 24:
#   horse, automobile, and horse+automobile (implemented as horse,automobile).
# Phase 2: run 24 attribution tasks:
#   3 queries x (5 traj_tracin ranges + das + dtrak + end_tracin).
#
# Submit from this repository root on Vista:
#   sbatch diffusion_jax_refined/cifar2/vista/01_sample_and_attribute_vista.sh

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-24}"
SAMPLE_TRAJECTORY_STEPS="${SAMPLE_TRAJECTORY_STEPS:-100}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-trajectory_noise_squared_deviation_normalized}"
TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-16}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
MAX_PARALLEL_ATTR_TASKS="${MAX_PARALLEL_ATTR_TASKS:-${SLURM_NTASKS:-24}}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/sample_attr_seed_${INITIAL_SEED}_${SLURM_JOB_ID}"

QUERIES=("horse" "automobile" "horse,automobile")
TRAJ_RANGES=("1-2000" "2001-4000" "4001-6000" "6001-8000" "8001-10000")
ENDPOINT_ALGORITHMS=("das" "dtrak" "end_tracin")

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

query_log_tag() {
  local value="$1"
  value="${value//,/__}"
  value="${value//+/_}"
  value="${value//[^A-Za-z0-9._-]/_}"
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

sample_dir_for_query() {
  local query="$1"
  printf '%s/result/%s/eval/sampling/cifar/prompt_%s/model_prompted_jax__ckpt_seed_42_epoch_0200/seed_%06d' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "$(path_tag "${query}")" "${INITIAL_SEED}"
}

score_dir_for_traj_range() {
  local query="$1"
  local range="$2"
  printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/%s_range_%s' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${ATTRIBUTION_SCORE_MODEL_MODE:-${SAMPLE_MODEL_MODE:-prompted_solo}}" "${TRAIN_SEED:-42}" "$(path_tag "${query}")" "${INITIAL_SEED}" \
    "$(traj_algorithm_tag)" "${range//-/_}"
}

score_dir_for_endpoint() {
  local query="$1"
  local algorithm="$2"
  printf '%s/result/%s/attribution_score/%s/train_seed_%s/query_%s/initial_seed_%s/%s_range_1_10000' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${ATTRIBUTION_SCORE_MODEL_MODE:-${SAMPLE_MODEL_MODE:-prompted_solo}}" "${TRAIN_SEED:-42}" "$(path_tag "${query}")" "${INITIAL_SEED}" "${algorithm}"
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

validate_sample() {
  local query="$1"
  local sample_dir
  sample_dir="$(sample_dir_for_query "${query}")"
  python - "${sample_dir}" "${query}" "${INITIAL_SEED}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

sample_dir = Path(sys.argv[1])
expected_prompt = sys.argv[2]
expected_seed = int(sys.argv[3])
required = ["trajectory_xt.npy", "trajectory_t.npy", "final_state.npy", "seed_info.json"]
missing = [name for name in required if not (sample_dir / name).is_file()]
manifest = sample_dir.parent / "manifest.json"
if not manifest.is_file():
    missing.append(str(manifest))
if missing:
    raise FileNotFoundError(f"Invalid/incomplete sample {sample_dir}; missing {missing}")
seed_info = json.loads((sample_dir / "seed_info.json").read_text())
manifest_payload = json.loads(manifest.read_text())
if int(seed_info.get("seed", -1)) != expected_seed:
    raise ValueError(f"{sample_dir}: seed_info seed={seed_info.get('seed')} expected={expected_seed}")
if expected_prompt != str(seed_info.get("prompt")):
    raise ValueError(f"{sample_dir}: prompt={seed_info.get('prompt')!r} expected={expected_prompt!r}")
if expected_prompt != str(manifest_payload.get("prompt")):
    raise ValueError(f"{manifest}: prompt={manifest_payload.get('prompt')!r} expected={expected_prompt!r}")
trajectory = np.load(sample_dir / "trajectory_xt.npy", mmap_mode="r")
times = np.load(sample_dir / "trajectory_t.npy", mmap_mode="r")
final_state = np.load(sample_dir / "final_state.npy", mmap_mode="r")
if trajectory.ndim != 5 or final_state.ndim != 4:
    raise ValueError(f"{sample_dir}: bad trajectory/final shapes {trajectory.shape}, {final_state.shape}")
if int(trajectory.shape[0]) != int(times.shape[0]):
    raise ValueError(f"{sample_dir}: trajectory length {trajectory.shape[0]} != times {times.shape[0]}")
print(f"Validated sample: {sample_dir}")
PY
}

echo "Vista gh job: CIFAR2 sample + attribution"
echo "REPO_ROOT             : ${REPO_ROOT}"
echo "CIFAR2_ROOT           : ${CIFAR2_ROOT}"
echo "EXPERIMENT_TAG        : ${EXPERIMENT_TAG}"
echo "INITIAL_SEED          : ${INITIAL_SEED}"
echo "TRAJ_QUERY_OBJECTIVE  : ${TRAJ_QUERY_OBJECTIVE}"
echo "TRAJ algorithm folder : $(traj_algorithm_tag)"
echo "Logs                  : ${LOG_ROOT}"

echo "Phase 1/2: sampling query trajectories"
pids=()
slot=0
for query in "${QUERIES[@]}"; do
  sample_dir="$(sample_dir_for_query "${query}")"
  if [[ "${ALLOW_OVERWRITE}" != "1" && -f "${sample_dir}/trajectory_xt.npy" ]]; then
    echo "Skip existing sample: ${sample_dir}"
    validate_sample "${query}"
    continue
  fi
  log="${LOG_ROOT}/sample_$(query_log_tag "${query}").log"
  echo "Launching sample query=${query} slot=${slot} -> ${log}"
  ibrun -n 1 -o "${slot}" \
    env CUDA_VISIBLE_DEVICES=0 \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      QUERY="${query}" \
      INITIAL_SEED="${INITIAL_SEED}" \
      SAMPLE_SEEDS="${INITIAL_SEED}" \
      SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE}" \
      SAMPLE_TRAJECTORY_STEPS="${SAMPLE_TRAJECTORY_STEPS}" \
    bash scripts/00_sample.sh >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done
if (( ${#pids[@]} > 0 )); then
  wait_batch "${pids[@]}"
fi
for query in "${QUERIES[@]}"; do
  validate_sample "${query}"
done

echo "Phase 2/2: attribution, 24 task layout"
pids=()
for query in "${QUERIES[@]}"; do
  tag="$(query_log_tag "${query}")"
  for range in "${TRAJ_RANGES[@]}"; do
    output_dir="$(score_dir_for_traj_range "${query}" "${range}")"
    if [[ "${ALLOW_OVERWRITE}" != "1" && -f "${output_dir}/scores.npy" ]]; then
      echo "Skip existing traj attribution: ${output_dir}"
      continue
    fi
    slot="${#pids[@]}"
    echo "Launch traj_tracin query=${query} range=${range} slot=${slot}"
    ibrun -n 1 -o "${slot}" \
      env CUDA_VISIBLE_DEVICES=0 \
        EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
        QUERY="${query}" \
        INITIAL_SEED="${INITIAL_SEED}" \
        SAMPLE_SEED="${INITIAL_SEED}" \
        TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE}" \
        TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE}" \
        ALGORITHMS="traj_tracin" \
        ATTRIBUTION_RANGES="${range}" \
      bash scripts/01_data_attribution.sh >"${LOG_ROOT}/attr_${tag}_traj_${range//-/_}.log" 2>&1 &
    pids+=("$!")
  done

  for algorithm in "${ENDPOINT_ALGORITHMS[@]}"; do
    output_dir="$(score_dir_for_endpoint "${query}" "${algorithm}")"
    if [[ "${ALLOW_OVERWRITE}" != "1" && -f "${output_dir}/scores.npy" ]]; then
      echo "Skip existing endpoint attribution: ${output_dir}"
      continue
    fi
    slot="${#pids[@]}"
    echo "Launch ${algorithm} query=${query} slot=${slot}"
    ibrun -n 1 -o "${slot}" \
      env CUDA_VISIBLE_DEVICES=0 \
        EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
        QUERY="${query}" \
        INITIAL_SEED="${INITIAL_SEED}" \
        SAMPLE_SEED="${INITIAL_SEED}" \
        ALGORITHMS="${algorithm}" \
        ATTRIBUTION_RANGES="1-10000" \
      bash scripts/01_data_attribution.sh >"${LOG_ROOT}/attr_${tag}_${algorithm}.log" 2>&1 &
    pids+=("$!")
  done
done

if (( ${#pids[@]} > MAX_PARALLEL_ATTR_TASKS )); then
  echo "Internal error: launched ${#pids[@]} tasks but allocation allows ${MAX_PARALLEL_ATTR_TASKS}." >&2
  exit 1
fi
wait_batch "${pids[@]}"

echo "Sampling and all attribution tasks completed successfully."
