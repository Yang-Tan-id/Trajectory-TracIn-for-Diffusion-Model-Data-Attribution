#!/usr/bin/env bash
#SBATCH --job-name=cifar2-traj-n2-k8k
#SBATCH --partition=h100
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=cifar2-traj-n2-k8k-%j.out
#SBATCH --error=cifar2-traj-n2-k8k-%j.err

set -euo pipefail

# One H100 allocation:
#   1) run traj_tracin attribution with the original norm^2 objective;
#   2) split CIFAR2 scores into four 1-based ranges of 2500 examples;
#   3) evaluate against existing k=8000 LDS models;
#   4) aggregate per-seed LDS summaries.
#
# Submit from the repository root:
#   sbatch diffusion_jax_refined/cifar2/tacc/10_traj_norm2_k8000_h100_3node.sh

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
INITIAL_SEED="${INITIAL_SEED:-42}"
TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-trajectory_noise_squared_deviation}"
TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-16}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-8000}"
LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 1 16)}"
LDS_TARGET_FUNCTION="${LDS_TARGET_FUNCTION:-noise_trajectory}"
LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"

MAX_PARALLEL_ATTR_TASKS="${MAX_PARALLEL_ATTR_TASKS:-${SLURM_NTASKS:-12}}"
MAX_PARALLEL_EVAL_TASKS="${MAX_PARALLEL_EVAL_TASKS:-${SLURM_NTASKS:-12}}"
GPU_PER_NODE="${GPU_PER_NODE:-4}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/traj_norm2_k${LDS_K}_m${LDS_M}_${SLURM_JOB_ID}"

QUERIES=("horse" "automobile" "horse,automobile")
TRAJ_RANGES=("1-2500" "2501-5000" "5001-7500" "7501-10000")

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

sample_dir_for_query() {
  local query="$1"
  printf '%s/result/%s/eval/sampling/cifar/prompt_%s/model_prompted_jax__ckpt_seed_42_epoch_0200/seed_%06d' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "$(path_tag "${query}")" "${INITIAL_SEED}"
}

attribution_root_for_query() {
  printf '%s/result/%s/attribution_score/query_%s/initial_seed_%s' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "$(path_tag "$1")" "${INITIAL_SEED}"
}

score_dir_for_range() {
  local query="$1"
  local range="$2"
  printf '%s/%s_range_%s' \
    "$(attribution_root_for_query "${query}")" "$(traj_algorithm_tag)" "${range//-/_}"
}

score_dirs_for_query() {
  local query="$1"
  local range output=""
  for range in "${TRAJ_RANGES[@]}"; do
    output+="${output:+,}$(score_dir_for_range "${query}" "${range}")"
  done
  printf '%s' "${output}"
}

lds_model_dir_for_seed() {
  local seed="$1"
  printf '%s/result/%s/lds_model/m_%s_k_%s_seed_%s' \
    "${CIFAR2_ROOT}" "${EXPERIMENT_TAG}" "${LDS_M}" "${LDS_K}" "${seed}"
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

validate_samples() {
  echo "Validating saved query trajectories..."
  local query
  for query in "${QUERIES[@]}"; do
    validate_sample "${query}"
  done
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

run_attribution() {
  echo "Running traj_tracin attribution objective=${TRAJ_QUERY_OBJECTIVE}; ranges=${TRAJ_RANGES[*]}"
  local pids=()
  local query range slot tag output_dir
  for query in "${QUERIES[@]}"; do
    tag="$(query_tag "${query}")"
    for range in "${TRAJ_RANGES[@]}"; do
      output_dir="$(score_dir_for_range "${query}" "${range}")"
      if [[ "${ALLOW_OVERWRITE}" != "1" && -f "${output_dir}/scores.npy" ]]; then
        echo "Skip existing attribution: ${output_dir}"
        continue
      fi
      if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${output_dir}" && ! -f "${output_dir}/scores.npy" ]]; then
        echo "Refusing to use incomplete attribution folder: ${output_dir}" >&2
        exit 1
      fi
      slot="${#pids[@]}"
      echo "Launch attribution query=${query} range=${range} slot=${slot}"
      ibrun -n 1 -o "${slot}" \
        env CUDA_VISIBLE_DEVICES="$((slot % GPU_PER_NODE))" \
          EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
          QUERY="${query}" \
          INITIAL_SEED="${INITIAL_SEED}" \
          SAMPLE_SEED="${INITIAL_SEED}" \
          TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE}" \
          TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE}" \
          ALGORITHMS="traj_tracin" \
          ATTRIBUTION_RANGES="${range}" \
        bash scripts/01_data_attribution.sh >"${LOG_ROOT}/attr_${tag}_${range//-/_}.log" 2>&1 &
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
}

run_eval() {
  echo "Running k=${LDS_K} LDS eval for $(traj_algorithm_tag)"
  local pids=()
  local pred_tag query tag seed model_dir dirs input out_dir slot
  pred_tag="$(prediction_tag "${LDS_PREDICTION_SUBSET}" "${LDS_PREDICTION_SIGN}")"
  for query in "${QUERIES[@]}"; do
    tag="$(path_tag "${query}")"
    dirs="$(score_dirs_for_query "${query}")"
    IFS=',' read -r -a inputs <<<"${dirs}"
    for input in "${inputs[@]}"; do
      [[ -f "${input}/scores.npy" ]] || { echo "Missing attribution scores: ${input}/scores.npy" >&2; exit 1; }
    done
    for seed in ${LDS_SEEDS}; do
      model_dir="$(lds_model_dir_for_seed "${seed}")"
      out_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval/query_${tag}/initial_seed_${INITIAL_SEED}/lds/$(traj_algorithm_tag)/${LDS_TARGET_FUNCTION}/${pred_tag}/$(basename "${model_dir}")"
      if [[ "${ALLOW_OVERWRITE}" != "1" && -f "${out_dir}/lds_results.csv" ]]; then
        echo "Skip existing eval: ${out_dir}"
        continue
      fi
      slot="${#pids[@]}"
      echo "Launch LDS eval query=${query} seed=${seed} slot=${slot}"
      ibrun -n 1 -o "${slot}" \
        env CUDA_VISIBLE_DEVICES="$((slot % GPU_PER_NODE))" \
          EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
          QUERY="${query}" \
          INITIAL_SEED="${INITIAL_SEED}" \
          ALGORITHMS="$(traj_algorithm_tag)" \
          ATTRIBUTION_RESULT_DIRS="${dirs}" \
          LDS_MODEL_DIRS="${model_dir}" \
          LDS_DEVICE=gpu \
          LDS_NUM_DEVICES=1 \
        bash scripts/04_lds_eval.sh \
          --target-function "${LDS_TARGET_FUNCTION}" \
          --prediction-subset "${LDS_PREDICTION_SUBSET}" \
          --prediction-sign "${LDS_PREDICTION_SIGN}" \
        >"${LOG_ROOT}/eval_${tag}_seed_${seed}.log" 2>&1 &
      pids+=("$!")
      if (( ${#pids[@]} >= MAX_PARALLEL_EVAL_TASKS )); then
        wait_batch "${pids[@]}"
        pids=()
      fi
    done
  done
  if (( ${#pids[@]} > 0 )); then
    wait_batch "${pids[@]}"
  fi
}

run_aggregate() {
  echo "Aggregating per-seed LDS evals"
  local pred_tag
  pred_tag="$(prediction_tag "${LDS_PREDICTION_SUBSET}" "${LDS_PREDICTION_SIGN}")"
  python "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
    --eval-root "${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/eval" \
    --target-function "${LDS_TARGET_FUNCTION}" \
    --lds-m "${LDS_M}" \
    --lds-k "${LDS_K}" \
    --initial-seed "${INITIAL_SEED}" \
    --algorithms "$(traj_algorithm_tag)" \
    --prediction-dir "${pred_tag}" \
    --output-name "aggregate_m_${LDS_M}_k_${LDS_K}_${pred_tag}_traj_norm2_seeds_${LDS_SEEDS// /_}" \
    >"${LOG_ROOT}/aggregate.log" 2>&1
  echo "Aggregate log: ${LOG_ROOT}/aggregate.log"
}

echo "CIFAR2 traj_tracin norm^2 attribution + k=${LDS_K} LDS eval"
echo "CIFAR2_ROOT=${CIFAR2_ROOT}"
echo "EXPERIMENT_TAG=${EXPERIMENT_TAG}; INITIAL_SEED=${INITIAL_SEED}"
echo "TRAJ_QUERY_OBJECTIVE=${TRAJ_QUERY_OBJECTIVE}; algorithm folder=$(traj_algorithm_tag)"
echo "TRAJ_SCORE_BATCH_SIZE=${TRAJ_SCORE_BATCH_SIZE}"
echo "MAX_PARALLEL_ATTR_TASKS=${MAX_PARALLEL_ATTR_TASKS}; MAX_PARALLEL_EVAL_TASKS=${MAX_PARALLEL_EVAL_TASKS}"
echo "Logs: ${LOG_ROOT}"

validate_samples
run_attribution
validate_lds_models
run_eval
run_aggregate

echo "Done. Logs are in ${LOG_ROOT}"
