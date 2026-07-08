#!/usr/bin/env bash
#SBATCH --job-name=cifar2-sample-attr
#SBATCH --partition=h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-sample-attr-%j.out
#SBATCH --error=cifar2-sample-attr-%j.err

set -euo pipefail

# Part 1 samples and runs the first 16 attribution tasks. Part 2 runs the
# remaining 8 tasks and then launches LDS evaluation in the same allocation.

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-42}"
ATTR_SHARD="${ATTR_SHARD:?Set ATTR_SHARD=1 for the first 16 tasks or ATTR_SHARD=2 for the remaining 8}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/sample_attr_part${ATTR_SHARD}_${SLURM_JOB_ID}"
MAX_PARALLEL_ATTR_TASKS="${MAX_PARALLEL_ATTR_TASKS:-${SLURM_NTASKS:-16}}"
[[ "${ATTR_SHARD}" == "1" || "${ATTR_SHARD}" == "2" ]] || {
  echo "ATTR_SHARD must be 1 or 2" >&2
  exit 1
}

# CIFAR multi-label queries are comma-separated. This represents the requested
# "horse + automobile" query in the format accepted by the model.
QUERIES=("horse" "automobile" "horse,automobile")
TRAJ_RANGES=("1-2000" "2001-4000" "4001-6000" "6001-8000" "8001-10000")
ENDPOINT_ALGORITHMS_TEXT="${ENDPOINT_ALGORITHMS_TEXT:-das dtrak end_tracin}"
if [[ "${RUN_ENDPOINTS:-1}" == "0" ]]; then
  ENDPOINT_ALGORITHMS=()
else
  read -r -a ENDPOINT_ALGORITHMS <<<"${ENDPOINT_ALGORITHMS_TEXT}"
fi
TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-${QUERY_OBJECTIVE:-trajectory_noise_squared_deviation}}"

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
      output+="${output:+,}${root}/$(traj_algorithm_tag)_range_${range//-/_}"
    done
  else
    output="${root}/${algorithm}_range_1_10000"
  fi
  printf '%s' "${output}"
}

sample_dir_for_query() {
  local query="$1"
  local tag
  tag="$(path_tag "${query}")"
  printf '%s/result/%s/eval/sampling/cifar/prompt_%s/model_prompted_jax__ckpt_seed_42_epoch_0200/seed_%06d' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${tag}" "${INITIAL_SEED}"
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
required = [
    "trajectory_xt.npy",
    "trajectory_t.npy",
    "final_state.npy",
    "seed_info.json",
]
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
    raise ValueError(f"{sample_dir}: seed_info prompt={seed_info.get('prompt')!r} expected={expected_prompt!r}")
if expected_prompt != str(manifest_payload.get("prompt")):
    raise ValueError(f"{manifest}: manifest prompt={manifest_payload.get('prompt')!r} expected={expected_prompt!r}")
if expected_seed not in [int(x) for x in manifest_payload.get("seeds", [])]:
    raise ValueError(f"{manifest}: expected seed {expected_seed} not in manifest seeds={manifest_payload.get('seeds')}")

trajectory = np.load(sample_dir / "trajectory_xt.npy", mmap_mode="r")
times = np.load(sample_dir / "trajectory_t.npy", mmap_mode="r")
final_state = np.load(sample_dir / "final_state.npy", mmap_mode="r")
if trajectory.ndim != 5:
    raise ValueError(f"{sample_dir}: trajectory_xt.npy shape should be (K,B,H,W,C), got {trajectory.shape}")
if final_state.ndim != 4:
    raise ValueError(f"{sample_dir}: final_state.npy shape should be (B,H,W,C), got {final_state.shape}")
if int(trajectory.shape[0]) != int(times.shape[0]):
    raise ValueError(f"{sample_dir}: trajectory length {trajectory.shape[0]} != trajectory_t length {times.shape[0]}")
if int(trajectory.shape[1]) != int(final_state.shape[0]):
    raise ValueError(f"{sample_dir}: trajectory batch {trajectory.shape[1]} != final_state batch {final_state.shape[0]}")
print(f"Validated sample: {sample_dir} | trajectory={tuple(trajectory.shape)} | final={tuple(final_state.shape)}")
PY
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
  local slot="$2"
  local tag
  tag="$(query_tag "${query}")"
  echo "Sampling query=${query}, initial_seed=${INITIAL_SEED}"
  ibrun -n 1 -o "${slot}" \
    env CUDA_VISIBLE_DEVICES="$((slot % 4))" \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      QUERY="${query}" \
      SAMPLE_SEEDS="${INITIAL_SEED}" \
    bash scripts/00_sample.sh >"${LOG_ROOT}/sample_${tag}.log" 2>&1 &
}

launch_attribution() {
  local query="$1"
  local algorithm="$2"
  local range="${3:-}"
  local slot="$4"
  local tag suffix
  tag="$(query_tag "${query}")"
  suffix="${algorithm}"
  if [[ -n "${range}" ]]; then
    suffix="${suffix}_${range//-/_}"
  fi
  echo "Attribution query=${query}, algorithm=${algorithm}, range=${range:-all}"
  ibrun -n 1 -o "${slot}" \
    env CUDA_VISIBLE_DEVICES="$((slot % 4))" \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      QUERY="${query}" \
      INITIAL_SEED="${INITIAL_SEED}" \
      TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE}" \
      ALGORITHMS="${algorithm}" \
      ATTRIBUTION_RANGES="${range}" \
    bash scripts/01_data_attribution.sh >"${LOG_ROOT}/attr_${tag}_${suffix}.log" 2>&1 &
}

if [[ "${ATTR_SHARD}" == "1" && "${ALLOW_OVERWRITE}" != "1" ]]; then
  for query in "${QUERIES[@]}"; do
    sample_dir="$(sample_dir_for_query "${query}")"
    if [[ -e "${sample_dir}" ]]; then
      echo "Refusing to overwrite existing sample: ${sample_dir}" >&2
      exit 1
    fi
  done
fi

if [[ "${ATTR_SHARD}" == "1" ]]; then
  echo "Phase 1/2: sampling three queries"
  pids=()
  sample_slot=0
  for query in "${QUERIES[@]}"; do
    launch_sample "${query}" "${sample_slot}"
    pids+=("$!")
    sample_slot=$((sample_slot + 1))
  done
  wait_batch "${pids[@]}"
else
  echo "Part 2: reusing samples created by attribution part 1"
fi

echo "Validating attribution samples"
for query in "${QUERIES[@]}"; do
  validate_sample "${query}"
done

echo "Attribution part ${ATTR_SHARD}/2"
pids=()
task_index=0
for query in "${QUERIES[@]}"; do
  for range in "${TRAJ_RANGES[@]}"; do
    task_index=$((task_index + 1))
    if [[ "${ATTR_SHARD}" == "1" && "${task_index}" -gt 16 ]] ||
       [[ "${ATTR_SHARD}" == "2" && "${task_index}" -le 16 ]]; then
      continue
    fi
    output_dir="$(attribution_root_for_query "${query}")/$(traj_algorithm_tag)_range_${range//-/_}"
    if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${output_dir}" ]]; then
      echo "Refusing to overwrite ${output_dir}" >&2
      exit 1
    fi
    slot="${#pids[@]}"
    launch_attribution "${query}" traj_tracin "${range}" "${slot}"
    pids+=("$!")
    if (( ${#pids[@]} >= MAX_PARALLEL_ATTR_TASKS )); then
      wait_batch "${pids[@]}"
      pids=()
    fi
  done
  for algorithm in "${ENDPOINT_ALGORITHMS[@]}"; do
    task_index=$((task_index + 1))
    if [[ "${ATTR_SHARD}" == "1" && "${task_index}" -gt 16 ]] ||
       [[ "${ATTR_SHARD}" == "2" && "${task_index}" -le 16 ]]; then
      continue
    fi
    output_dir="$(attribution_root_for_query "${query}")/${algorithm}_range_1_10000"
    if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${output_dir}" ]]; then
      echo "Refusing to overwrite ${output_dir}" >&2
      exit 1
    fi
    slot="${#pids[@]}"
    launch_attribution "${query}" "${algorithm}" "" "${slot}"
    pids+=("$!")
    if (( ${#pids[@]} >= MAX_PARALLEL_ATTR_TASKS )); then
      wait_batch "${pids[@]}"
      pids=()
    fi
  done
done
if (( ${#pids[@]} > 0 )); then
  wait_batch "${pids[@]}"
fi

echo "Attribution part ${ATTR_SHARD}/2 completed successfully (${#pids[@]} tasks)."
echo "Logs: ${LOG_ROOT}"

if [[ "${ATTR_SHARD}" == "2" ]]; then
  echo "Starting LDS evaluation after attribution part 2"
  bash "${CIFAR2_ROOT}/tacc/03_lds_eval_h100.sh"
fi
