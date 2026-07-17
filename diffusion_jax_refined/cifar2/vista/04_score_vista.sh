#!/usr/bin/env bash
#SBATCH --job-name=cifar2-score
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=21
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=cifar2-score-%j.out
#SBATCH --error=cifar2-score-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_vista_pipeline_lib.sh
source "${SCRIPT_DIR}/_vista_pipeline_lib.sh"
vista_init

ALGORITHMS=(dtrak das end_tracin traj_tracin)
SPECS=(
  "unprompted_solo|unprompted|24" "unprompted_solo|unprompted|48" "unprompted_solo|unprompted|96"
  "unprompted_multi|unprompted|24" "unprompted_multi|unprompted|48" "unprompted_multi|unprompted|96"
  "prompted_solo|horse|24" "prompted_solo|horse|48" "prompted_solo|horse|96"
  "prompted_solo|automobile|24" "prompted_solo|automobile|48" "prompted_solo|automobile|96"
  "prompted_multi|horse|24" "prompted_multi|horse|48" "prompted_multi|horse|96"
  "prompted_multi|automobile|24" "prompted_multi|automobile|48" "prompted_multi|automobile|96"
  "prompted_multi|horse,automobile|24" "prompted_multi|horse,automobile|48" "prompted_multi|horse,automobile|96"
)
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/04_score/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

echo "Job 04: pure score calculation from 02/03 artifacts"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; algorithms=${ALGORITHMS[*]}"
echo "query tasks=${#SPECS[@]}; logs=${LOG_ROOT}"

pids=()
slot=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r mode query seed <<<"${spec}"
  log="${LOG_ROOT}/$(log_tag "${mode}" "${query}" "${seed}").log"
  echo "Launch score mode=${mode} query=${query} seed=${seed} slot=${slot} -> ${log}"
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
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      TRAIN_SEED="${TRAIN_SEED}" \
      SAMPLE_MODEL_MODE="${mode}" \
      ATTRIBUTION_SCORE_MODEL_MODE="${mode}" \
      QUERY="${query_env}" \
      INITIAL_SEED="${seed}" \
      SAMPLE_SEED="${seed}" \
      UNPROMPTED="${unprompted_flag}" \
      bash -c 'for algorithm in dtrak das end_tracin traj_tracin; do echo "[score-combine] ${SAMPLE_MODEL_MODE} ${QUERY} seed=${INITIAL_SEED} algorithm=${algorithm}"; cd "data_attribution/${algorithm}"; "${PYTHON_BIN}" 03_score.py; cd - >/dev/null; done'
  ) >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done

wait_all "${pids[@]}"
echo "Job 04 complete."
