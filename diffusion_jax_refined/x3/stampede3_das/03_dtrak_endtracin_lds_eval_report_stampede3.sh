#!/usr/bin/env bash
#SBATCH -J x3-s3-orig-eval
#SBATCH -o x3-s3-orig-eval-%j.out
#SBATCH -e x3-s3-orig-eval-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/stampede3_das" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/x3/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/_stampede3_das_lib.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_stampede3_das_lib.sh
source "${SCRIPT_DIR}/_stampede3_das_lib.sh"
stampede3_das_init

TARGETS=(${LDS_TARGETS:-simple_loss noise_trajectory})
EVAL_ALGORITHMS=(${EVAL_ALGORITHMS:-dtrak end_tracin})
LDS_SEEDS_TEXT="${LDS_SEEDS:-$(seq -s ' ' 0 7)}"
LDS_M="${LDS_M:-64}"
LDS_K="${LDS_K:-5000}"
LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-50}"
LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"
PRED_TAG="${PRED_TAG:-pred_kept_sign_m1}"
PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 7)}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 23)}"
TARGETS_TEXT="${TARGETS[*]}"
EVAL_ALGORITHMS_TEXT="${EVAL_ALGORITHMS[*]}"
LDS_SIMPLE_LOSS_TIMESTEPS="${LDS_SIMPLE_LOSS_TIMESTEPS:-$(seq -s, 0 999)}"
LDS_SIMPLE_LOSS_NOISE_SEEDS="${LDS_SIMPLE_LOSS_NOISE_SEEDS:-0}"
LDS_SIMPLE_LOSS_NUM_MC="${LDS_SIMPLE_LOSS_NUM_MC:-1000}"
LDS_SIMPLE_LOSS_MC_SEED="${LDS_SIMPLE_LOSS_MC_SEED:-0}"
export PROMPTED_SEEDS_TEXT UNPROMPTED_SEEDS_TEXT LDS_SEEDS_TEXT TARGETS_TEXT EVAL_ALGORITHMS_TEXT
export LDS_SIMPLE_LOSS_TIMESTEPS LDS_SIMPLE_LOSS_NOISE_SEEDS LDS_SIMPLE_LOSS_NUM_MC LDS_SIMPLE_LOSS_MC_SEED
LOG_ROOT="${X3_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/03_dtrak_endtracin_lds_eval_report/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

query_specs() {
  local seed
  for seed in ${UNPROMPTED_SEEDS_TEXT}; do
    printf 'unprompted_solo|unprompted|unconditional|%s|1\n' "${seed}"
  done
  for seed in ${PROMPTED_SEEDS_TEXT}; do
    printf 'prompted_solo|horse|horse|%s|0\n' "${seed}"
    printf 'prompted_solo|automobile|automobile|%s|0\n' "${seed}"
    printf 'prompted_solo|horse,automobile|horse,automobile|%s|0\n' "${seed}"
  done
}

mapfile -t SPECS < <(query_specs)
if [[ -n "${EVAL_SLOT_ONLY:-}" ]]; then
  SLOT_LIST="${EVAL_SLOT_ONLY}"
else
  SLOT_LIST="$(seq -s ' ' 0 15)"
fi

echo "Job 03 Stampede3 dtrak/end_tracin: LDS eval + aggregate/report"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; query_tasks=${#SPECS[@]}; eval_slots=${SLOT_LIST}; queries_per_slot=3"
echo "targets=${TARGETS[*]}; algorithms=${EVAL_ALGORITHMS[*]}; lds_seeds=${LDS_SEEDS_TEXT}"
echo "simple_loss_terms=${LDS_SIMPLE_LOSS_NUM_MC}; simple_loss_noise_seeds=${LDS_SIMPLE_LOSS_NOISE_SEEDS}; logs=${LOG_ROOT}"
echo "eval_device_mode=${LDS_EVAL_DEVICE_MODE:-gpu_then_cpu}"
echo "eval_slot_shards=${EVAL_SLOT_SHARD_COUNT:-1}; serial_slots=${EVAL_SERIAL_SLOTS:-0}; local_parallel_slots=${EVAL_LOCAL_PARALLEL_SLOTS:-0}; force=${FORCE_LDS_EVAL:-0}"

run_eval_slot_once() {
  local slot="$1"
  local launch_slot="$2"
  local shard_index="$3"
  local shard_count="$4"
  local attempt="$5"
  local device="$6"
  local gpu="${7:-}"
  local -a device_env
  if [[ "${device}" == "gpu" ]]; then
    device_env=(
      CUDA_VISIBLE_DEVICES="${gpu}"
      GPU_IDS="${gpu}"
      LDS_DEVICE="${LDS_GPU_DEVICE:-gpu}"
    )
  else
    device_env=(
      CUDA_VISIBLE_DEVICES=""
      GPU_IDS=""
      JAX_PLATFORMS=cpu
      LDS_DEVICE=cpu
    )
  fi
  echo "[eval-slot] slot=${slot} shard=${shard_index}/${shard_count} attempt=${attempt} device=${device} gpu=${gpu:-none}"
  run_gpu_slot "${launch_slot}" env \
    "${device_env[@]}" \
    JAX_NUM_DEVICES=1 \
    X3_ROOT="${X3_ROOT}" \
    REPO_ROOT="${REPO_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
    TRAIN_SEED="${TRAIN_SEED}" \
    SLOT_INDEX="${slot}" \
    EVAL_SLOT_SHARD_INDEX="${shard_index}" \
    EVAL_SLOT_SHARD_COUNT="${shard_count}" \
    LDS_M="${LDS_M}" \
    LDS_K="${LDS_K}" \
    LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE}" \
    LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET}" \
    LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN}" \
    PRED_TAG="${PRED_TAG}" \
    LDS_NUM_DEVICES=1 \
    LDS_SIMPLE_LOSS_NUM_MC="${LDS_SIMPLE_LOSS_NUM_MC}" \
    LDS_SIMPLE_LOSS_MC_SEED="${LDS_SIMPLE_LOSS_MC_SEED}" \
    LDS_SIMPLE_LOSS_TIMESTEPS="${LDS_SIMPLE_LOSS_TIMESTEPS}" \
    LDS_SIMPLE_LOSS_NOISE_SEEDS="${LDS_SIMPLE_LOSS_NOISE_SEEDS}" \
    FORCE_LDS_EVAL="${FORCE_LDS_EVAL:-0}" \
    bash "${SCRIPT_DIR}/03_dtrak_endtracin_lds_eval_slot_stampede3.sh"
}

run_eval_slot_with_fallback() {
  local slot="$1"
  local launch_slot="$2"
  local shard_index="$3"
  local shard_count="$4"
  local mode="${LDS_EVAL_DEVICE_MODE:-gpu_then_cpu}"
  local gpu=$((launch_slot % 4))
  case "${mode}" in
    gpu)
      run_eval_slot_once "${slot}" "${launch_slot}" "${shard_index}" "${shard_count}" "gpu-only" gpu "${gpu}"
      ;;
    cpu)
      run_eval_slot_once "${slot}" "${launch_slot}" "${shard_index}" "${shard_count}" "cpu-only" cpu
      ;;
    gpu_then_cpu)
      if run_eval_slot_once "${slot}" "${launch_slot}" "${shard_index}" "${shard_count}" "1/2" gpu "${gpu}"; then
        return 0
      fi
      echo "[eval-slot] slot=${slot} shard=${shard_index}/${shard_count} GPU attempt failed; retrying missing work on CPU"
      run_eval_slot_once "${slot}" "${launch_slot}" "${shard_index}" "${shard_count}" "2/2" cpu
      ;;
    *)
      echo "Unknown LDS_EVAL_DEVICE_MODE=${mode}; expected gpu, cpu, or gpu_then_cpu" >&2
      return 2
      ;;
  esac
}

pids=()
launch_offset=0
local_parallel_slots="${EVAL_LOCAL_PARALLEL_SLOTS:-0}"
for slot in ${SLOT_LIST}; do
  shard_count="${EVAL_SLOT_SHARD_COUNT:-1}"
  for ((shard_index = 0; shard_index < shard_count; shard_index++)); do
    if (( shard_count == 1 )); then
      log="${LOG_ROOT}/slot_${slot}.log"
    else
      log="${LOG_ROOT}/slot_${slot}_shard_${shard_index}.log"
    fi
    echo "Launch eval slot=${slot} shard=${shard_index}/${shard_count}; handles query indices ${slot}, $((slot + 16)), $((slot + 32)) -> ${log}"
    if [[ "${EVAL_SERIAL_SLOTS:-0}" == "1" ]]; then
      (
        run_eval_slot_with_fallback "${slot}" "${launch_offset}" "${shard_index}" "${shard_count}"
      ) >"${log}" 2>&1
    else
      (
        run_eval_slot_with_fallback "${slot}" "${launch_offset}" "${shard_index}" "${shard_count}"
      ) >"${log}" 2>&1 &
      pids+=("$!")
      if (( local_parallel_slots > 0 && ${#pids[@]} >= local_parallel_slots )); then
        wait_all "${pids[@]}"
        pids=()
      fi
    fi
    launch_offset=$((launch_offset + 1))
  done
done

if [[ "${EVAL_SERIAL_SLOTS:-0}" != "1" ]]; then
  wait_all "${pids[@]}"
fi

echo "Aggregating Stampede3 dtrak/end_tracin LDS evals and writing reports"
for target in "${TARGETS[@]}"; do
  for initial_seed in ${PROMPTED_SEEDS_TEXT}; do
    for query in horse automobile horse,automobile; do
      "${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
        --eval-root "${X3_ROOT}/result/${EXPERIMENT_TAG}/eval/prompted_solo" \
        --queries "query_$(path_tag "${query}")" \
        --target-function "${target}" \
        --lds-m "${LDS_M}" \
        --lds-k "${LDS_K}" \
        --model-glob "m_${LDS_M}_k_*_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_*" \
        --initial-seed "${initial_seed}" \
        --prediction-dir "${PRED_TAG}" \
        --algorithms "${EVAL_ALGORITHMS[@]}" \
        --output-name "aggregate_stampede3_dtrak_endtracin_m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_${target}_initial_seed_${initial_seed}_seeds_0_7" \
        >"${LOG_ROOT}/aggregate_prompted_$(path_tag "${query}")_${target}_seed_${initial_seed}.log" 2>&1 || true
    done
  done
  for initial_seed in ${UNPROMPTED_SEEDS_TEXT}; do
    "${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/aggregate_lds_by_seed.py" \
      --eval-root "${X3_ROOT}/result/${EXPERIMENT_TAG}/eval/unprompted_solo" \
      --target-function "${target}" \
      --eval-kind lds_unprompted \
      --queries unprompted \
      --lds-m "${LDS_M}" \
      --lds-k "${LDS_K}" \
      --model-glob "m_${LDS_M}_k_*_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_*" \
      --initial-seed "${initial_seed}" \
      --prediction-dir "${PRED_TAG}" \
      --algorithms "${EVAL_ALGORITHMS[@]}" \
      --output-name "aggregate_stampede3_dtrak_endtracin_unprompted_m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_${target}_initial_seed_${initial_seed}_seeds_0_7" \
      >"${LOG_ROOT}/aggregate_unprompted_${target}_seed_${initial_seed}.log" 2>&1 || true
  done
done

"${PYTHON_BIN}" "${REPO_ROOT}/diffusion_jax_refined/common/summarize_lds_eval_report.py" \
  --eval-root "${X3_ROOT}/result/${EXPERIMENT_TAG}/eval" \
  --output-dir "${X3_ROOT}/result/${EXPERIMENT_TAG}/eval/reports" \
  >"${LOG_ROOT}/summary_report.log" 2>&1

echo "Job 03 complete. Report root: ${X3_ROOT}/result/${EXPERIMENT_TAG}/eval/reports"
