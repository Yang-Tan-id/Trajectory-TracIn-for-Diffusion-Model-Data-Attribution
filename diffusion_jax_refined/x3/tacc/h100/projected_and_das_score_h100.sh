#!/usr/bin/env bash
#SBATCH -J x3-proj-das-score
#SBATCH -o x3-proj-das-score-%j.out
#SBATCH -e x3-proj-das-score-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
X3_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REFINE_ROOT="$(cd "${X3_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${REFINE_ROOT}/.." && pwd)"

source "${X3_ROOT}/stampede3_das/_stampede3_das_lib.sh"
stampede3_das_init

QUERY_PLAN="${QUERY_PLAN:-${X3_ROOT}/result/${EXPERIMENT_TAG}/query_plan/query48.tsv}"
PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
PROJECTED_ARTIFACT_BASE="${PROJECTED_ARTIFACT_BASE:-${X3_ROOT}/result/${EXPERIMENT_TAG}/projected_traj_tracin_artifacts_next_ckpt}"
STREAM_VARIANTS="${STREAM_SAVE_TERM_SCORE_VARIANTS:-train_l2_normalized}"
STREAM_DIMS="${STREAM_PROJ_DIMS:-4096}"
STREAM_RANGES="${STREAM_SCORE_RANGES:-1-625 626-1250 1251-1875 1876-2500 2501-3125 3126-3750 3751-4375 4376-5000 5001-5625 5626-6250 6251-6875 6876-7500 7501-8125 8126-8750 8751-9375 9376-10000}"
DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES:-0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000}"
LOG_ROOT="${X3_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/projected_and_das_score/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

if [[ ! -f "${QUERY_PLAN}" ]]; then
  echo "Missing query plan: ${QUERY_PLAN}. Run projected_query48_rtx.sh first." >&2
  exit 1
fi

echo "Job score: x3 projected stream scores plus DAS lambda sweep"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; query_plan=${QUERY_PLAN}"
echo "projected_base=${PROJECTED_ARTIFACT_BASE}; dims=${STREAM_DIMS}; variants=${STREAM_VARIANTS}"
echo "das_lambdas=${DAS_DAMPING_SWEEP_VALUES}"
echo "logs=${LOG_ROOT}"

query_filters_for_mode() {
  local mode="$1"
  awk -F'\t' -v mode="${mode}" 'NR > 1 && $2 == mode {
    if ($6 == "1") {
      printf "unprompted/initial_seed_%s/shared_query/proj_4096/query_gradient_artifact.npz ", $4
    } else {
      q=$3
      gsub(/,/, "__", q)
      gsub(/[^A-Za-z0-9._-]+/, "_", q)
      gsub(/_+/, "_", q)
      sub(/^_/, "", q)
      sub(/_$/, "", q)
      printf "query_%s/initial_seed_%s/shared_query/proj_4096/query_gradient_artifact.npz ", q, $4
    }
  }' "${QUERY_PLAN}"
}

run_projected_target() {
  local target="$1"
  local root_suffix="$2"
  local base="${X3_ROOT}/result/${EXPERIMENT_TAG}/${root_suffix}"
  echo "[projected] target=${target} base=${base}"
  for mode in prompted_solo unprompted_solo; do
    filters="$(query_filters_for_mode "${mode}")"
    if [[ -z "${filters}" ]]; then
      echo "[projected] no filters for mode=${mode}; skip"
      continue
    fi
    dims_tag="$(path_tag "${STREAM_DIMS}")"
    variants_tag="$(path_tag "${STREAM_VARIANTS}")"
    mode_stream_root="${base}/stream_term_scores/score_mode_${mode}/cache_${PROJECTED_CACHE_DIM}/proj_${dims_tag}/variants_${variants_tag}"
    STREAM_TASK_SET="${mode}" \
    STREAM_QUERY_FILTERS="${filters}" \
    PROJECTED_ARTIFACT_BASE="${base}" \
    STREAM_SCORE_ROOT="${mode_stream_root}" \
    PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM}" \
    STREAM_PROJ_DIMS="${STREAM_DIMS}" \
    STREAM_SAVE_TERM_SCORE_VARIANTS="${STREAM_VARIANTS}" \
    STREAM_SCORE_RANGES="${STREAM_RANGES}" \
    TRAJ_QUERY_OBJECTIVE="${target}" \
    TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}" \
    PYTHONUNBUFFERED=1 \
    bash "${X3_ROOT}/tacc/h100/projected_traj_tracin_stream_score_array_h100.sh" \
      >"${LOG_ROOT}/projected_${target}_${mode}.log" 2>&1
  done
}

run_das_sweep_from_plan() {
  echo "[das] running lambda sweep for query plan"
  pids=()
  slot=0
  while IFS=$'\t' read -r sample_mode score_mode query init sample_index unprompted source_row_id labels; do
    gpu=$((slot % 4))
    query_env="${query}"
    if [[ "${unprompted}" == "1" ]]; then
      query_env="unconditional"
    fi
    log="${LOG_ROOT}/das_task_${slot}__${score_mode}__seed_${init}__$(path_tag "${query}").log"
    echo "[das] slot=${slot} gpu=${gpu} mode=${score_mode} query=${query} init=${init} -> ${log}"
    (
      run_gpu_slot "${slot}" env CUDA_VISIBLE_DEVICES="${gpu}" GPU_IDS="${gpu}" JAX_NUM_DEVICES=1 PYTHONUNBUFFERED=1 \
        X3_ROOT="${X3_ROOT}" REFINE_ROOT="${REFINE_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
        EXPERIMENT_TAG="${EXPERIMENT_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
        SAMPLE_MODEL_MODE="${sample_mode}" UNPROMPTED_SAMPLE_MODEL_MODE="${sample_mode}" ATTRIBUTION_SAMPLE_MODEL_MODE="${sample_mode}" \
        ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}" UNPROMPTED_SCORE_MODEL_MODE="${score_mode}" \
        QUERY="${query_env}" INITIAL_SEED="${init}" SAMPLE_SEED="${init}" SAMPLE_SEEDS="${init}" ATTRIBUTION_SAMPLE_INDEX="${sample_index}" \
        UNPROMPTED="${unprompted}" ALGORITHM=das DAS_DAMPING_SWEEP=1 DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES}" \
        bash "${X3_ROOT}/stampede3_das/02_das_attribution_task_stampede3.sh"
    ) >"${log}" 2>&1 &
    pids+=("$!")
    slot=$((slot + 1))
    if (( ${#pids[@]} >= 16 )); then
      wait_all "${pids[@]}"
      pids=()
    fi
  done < <(tail -n +2 "${QUERY_PLAN}")
  if (( ${#pids[@]} > 0 )); then
    wait_all "${pids[@]}"
  fi
}

run_projected_target "trajectory_next_checkpoint_noise_mse" "projected_traj_tracin_artifacts_next_ckpt"
run_projected_target "trajectory_next_checkpoint_ref_projection" "projected_traj_tracin_artifacts_refproj"
run_das_sweep_from_plan

echo "Job score complete."
