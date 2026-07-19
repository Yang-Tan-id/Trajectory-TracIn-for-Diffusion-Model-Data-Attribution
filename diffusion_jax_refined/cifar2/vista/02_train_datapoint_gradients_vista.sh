#!/usr/bin/env bash
#SBATCH --job-name=cifar2-train-grads
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-train-grads-%j.out
#SBATCH --error=cifar2-train-grads-%j.err

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

MODEL_MODES=(prompted_solo unprompted_solo)
ALGORITHMS=(dtrak das end_tracin traj_tracin)
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/02_train_datapoint_gradients/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

echo "Job 02: train datapoint gradients"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; canonical train modes=${MODEL_MODES[*]}; algorithms=${ALGORITHMS[*]}"
echo "prompted_multi reuses prompted_solo train artifacts; unprompted_multi reuses unprompted_solo train artifacts."
echo "logs=${LOG_ROOT}"

pids=()
slot=0
for mode in "${MODEL_MODES[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do
    log="${LOG_ROOT}/${mode}_${algorithm}.log"
    echo "Launch train-gradient mode=${mode} algorithm=${algorithm} slot=${slot} -> ${log}"
    unprompted_flag=0
    if mode_is_unprompted "${mode}"; then
      unprompted_flag=1
    fi
    (
      run_slot "${slot}" env \
        CUDA_VISIBLE_DEVICES=0 \
        GPU_IDS=0 \
        JAX_NUM_DEVICES=1 \
        EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
        TRAIN_SEED="${TRAIN_SEED}" \
        TRAIN_MODE="${mode}" \
        DATAPOINT_MODEL_MODE="${mode}" \
        UNPROMPTED="${unprompted_flag}" \
        bash -c "cd data_attribution/${algorithm} && '${PYTHON_BIN}' 01_train_datapoint_gradient.py"
    ) >"${log}" 2>&1 &
    pids+=("$!")
    slot=$((slot + 1))
  done
done

wait_all "${pids[@]}"
echo "Job 02 complete."
