#!/usr/bin/env bash
#SBATCH --job-name=cifar2-orig-sample-attr
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=21
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-orig-sample-attr-%j.out
#SBATCH --error=cifar2-orig-sample-attr-%j.err

set -euo pipefail

SCRIPT_DIR="${VISTA_ORIGINAL_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_vista_original_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_vista_original_lib.sh
source "${SCRIPT_DIR}/_vista_original_lib.sh"
vista_original_init

ALGORITHMS_TEXT="${ALGORITHMS:-dtrak das end_tracin traj_tracin}"
SPECS=(
  "unprompted_solo|unprompted|24" "unprompted_solo|unprompted|48" "unprompted_solo|unprompted|96"
  "unprompted_multi|unprompted|24" "unprompted_multi|unprompted|48" "unprompted_multi|unprompted|96"
  "prompted_solo|horse|24" "prompted_solo|horse|48" "prompted_solo|horse|96"
  "prompted_solo|automobile|24" "prompted_solo|automobile|48" "prompted_solo|automobile|96"
  "prompted_multi|horse|24" "prompted_multi|horse|48" "prompted_multi|horse|96"
  "prompted_multi|automobile|24" "prompted_multi|automobile|48" "prompted_multi|automobile|96"
  "prompted_multi|horse,automobile|24" "prompted_multi|horse,automobile|48" "prompted_multi|horse,automobile|96"
)
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_original_logs/02_sample_and_original_attribution/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

echo "Job 02: sample trajectories, then run original monolithic attribution"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; algorithms=${ALGORITHMS_TEXT}"
echo "query tasks=${#SPECS[@]}; logs=${LOG_ROOT}"
echo "This compact/original job writes final scores only; it does not write strict stage artifacts."

pids=()
slot=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r mode query seed <<<"${spec}"
  log="${LOG_ROOT}/$(log_tag "${mode}" "${query}" "${seed}").log"
  echo "Launch original sample+attribution mode=${mode} query=${query} seed=${seed} slot=${slot} -> ${log}"
  unprompted_flag=0
  query_env="${query}"
  if mode_is_unprompted "${mode}"; then
    unprompted_flag=1
    query_env="unconditional"
  fi
  (
    run_slot "${slot}" env \
      CUDA_VISIBLE_DEVICES=0 \
      GPU_IDS=0 \
      JAX_NUM_DEVICES=1 \
      PYTHONUNBUFFERED=1 \
      ATTRIBUTION_TQDM_MININTERVAL="${ATTRIBUTION_TQDM_MININTERVAL:-1}" \
      ATTRIBUTION_TQDM_LEAVE="${ATTRIBUTION_TQDM_LEAVE:-1}" \
      CIFAR2_ROOT="${CIFAR2_ROOT}" \
      REFINE_ROOT="${REFINE_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      TRAIN_SEED="${TRAIN_SEED}" \
      SAMPLE_MODEL_MODE="${mode}" \
      UNPROMPTED_SAMPLE_MODEL_MODE="${mode}" \
      ATTRIBUTION_SAMPLE_MODEL_MODE="${mode}" \
      ATTRIBUTION_SCORE_MODEL_MODE="${mode}" \
      UNPROMPTED_SCORE_MODEL_MODE="${mode}" \
      QUERY="${query_env}" \
      INITIAL_SEED="${seed}" \
      SAMPLE_SEED="${seed}" \
      SAMPLE_SEEDS="${seed}" \
      ALGORITHMS="${ALGORITHMS_TEXT}" \
      UNPROMPTED="${unprompted_flag}" \
      bash -c '
        set -euo pipefail
        echo "[sample] mode=${SAMPLE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED}"
        bash scripts/00_sample_for_attribution.sh
        for algorithm in ${ALGORITHMS}; do
          echo "[original-attribution] mode=${SAMPLE_MODEL_MODE} query=${QUERY} initial_seed=${INITIAL_SEED} algorithm=${algorithm}"
          "${PYTHON_BIN}" "${REFINE_ROOT}/common/run_original_attribution_config.py" "${CIFAR2_ROOT}/data_attribution/${algorithm}/CONFIG.py"
        done
      '
  ) >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done

wait_all "${pids[@]}"
echo "Job 02 complete."
