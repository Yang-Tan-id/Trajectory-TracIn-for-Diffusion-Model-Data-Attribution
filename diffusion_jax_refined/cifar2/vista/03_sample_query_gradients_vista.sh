#!/usr/bin/env bash
#SBATCH --job-name=cifar2-sample-qgrad
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=21
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-sample-qgrad-%j.out
#SBATCH --error=cifar2-sample-qgrad-%j.err

set -euo pipefail

SCRIPT_DIR="${VISTA_PIPELINE_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_vista_pipeline_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/vista"; do
    if [[ -n "${candidate}" && -f "${candidate}/_vista_pipeline_lib.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_vista_pipeline_lib.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_vista_pipeline_lib.sh
source "${SCRIPT_DIR}/_vista_pipeline_lib.sh"
vista_init

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
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/03_sample_query_gradients/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

echo "Job 03: sample + query gradients"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; algorithms=${ALGORITHMS_TEXT}"
echo "query tasks=${#SPECS[@]}; logs=${LOG_ROOT}"

pids=()
slot=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r mode query seed <<<"${spec}"
  log="${LOG_ROOT}/$(log_tag "${mode}" "${query}" "${seed}").log"
  echo "Launch sample+query-gradient mode=${mode} query=${query} seed=${seed} slot=${slot} -> ${log}"
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
      UNPROMPTED_SAMPLE_MODEL_MODE="${mode}" \
      ATTRIBUTION_SCORE_MODEL_MODE="${mode}" \
      UNPROMPTED_SCORE_MODEL_MODE="${mode}" \
      QUERY="${query_env}" \
      INITIAL_SEED="${seed}" \
      SAMPLE_SEED="${seed}" \
      SAMPLE_SEEDS="${seed}" \
      ALGORITHMS="${ALGORITHMS_TEXT}" \
      UNPROMPTED="${unprompted_flag}" \
      bash scripts/02_sample_query_gradient.sh
  ) >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done

wait_all "${pids[@]}"
echo "Job 03 complete."
