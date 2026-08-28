#!/usr/bin/env bash
#SBATCH -J cifar2-full-direct-all-h100
#SBATCH -o cifar2-full-direct-all-h100-%j.out
#SBATCH -e cifar2-full-direct-all-h100-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/run_4query_full_traj_direct_local.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
    if [[ -n "${candidate}" && -f "${candidate}/run_4query_full_traj_direct_local.sh" ]]; then
      SCRIPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/run_4query_full_traj_direct_local.sh" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# shellcheck source=_stampede3_das_lib.sh
source "${SCRIPT_DIR}/_stampede3_das_lib.sh"
stampede3_das_init
cd "${REPO_ROOT}"

PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 7)}"
UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s ' ' 0 23)}"

query_specs=()
for seed in ${UNPROMPTED_SEEDS_TEXT}; do
  query_specs+=("unprompted_solo|unprompted_solo|unconditional|${seed}|1")
done
for seed in ${PROMPTED_SEEDS_TEXT}; do
  query_specs+=("prompted_solo|prompted_solo|horse|${seed}|0")
  query_specs+=("prompted_solo|prompted_solo|automobile|${seed}|0")
  query_specs+=("prompted_multi|prompted_solo|horse,automobile|${seed}|0")
done

export STAMPEDE3_DAS_DIR="${SCRIPT_DIR}"
export GPU_SLOTS="${GPU_SLOTS:-16}"
export GPU_PER_NODE="${GPU_PER_NODE:-4}"
export RANGES="${RANGES:-1-2500 2501-5000 5001-7500 7501-10000}"
export QUERY_SPECS="${QUERY_SPECS:-${query_specs[*]}}"
export FULL_DIRECT_LOG_NAME="${FULL_DIRECT_LOG_NAME:-full_traj_direct_allquery_h100/${SLURM_JOB_ID:-local}}"
export FULL_DIRECT_SRUN_WORKER=1

export TRACIN_USE_LR_WEIGHTS="${TRACIN_USE_LR_WEIGHTS:-0}"
export TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-trajectory_next_checkpoint_noise_mse}"
export TRAJ_PARAMETER_SOURCE="${TRAJ_PARAMETER_SOURCE:-raw}"
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-2}"
export TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-4}"
export TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS="${TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS:-1}"
export TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS="${TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS:-10}"
export TRAJ_SAVE_QUERY_NORMALIZED_SCORES="${TRAJ_SAVE_QUERY_NORMALIZED_SCORES:-1}"
export FULL_DIRECT_VARIANTS="${FULL_DIRECT_VARIANTS:-raw q_l2}"

echo "H100 full direct Traj-TracIn all-query wrapper"
echo "job=${SLURM_JOB_ID:-local}; nodes=${SLURM_NODELIST:-local}; gpu_slots=${GPU_SLOTS}; time_limit=48:00:00"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; query_cases=${#query_specs[@]}; ranges=${RANGES}"
echo "lr_weights=${TRACIN_USE_LR_WEIGHTS}; objective=${TRAJ_QUERY_OBJECTIVE}; parameter_source=${TRAJ_PARAMETER_SOURCE}"
echo "aggregate_train_timestamps=${TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS}; aggregate_num_timesteps=${TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS}"
echo "variants=${FULL_DIRECT_VARIANTS}; save_query_normalized=${TRAJ_SAVE_QUERY_NORMALIZED_SCORES}"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L || true
fi

srun --ntasks="${GPU_SLOTS}" --ntasks-per-node="${GPU_PER_NODE}" \
  bash "${SCRIPT_DIR}/run_4query_full_traj_direct_local.sh"

echo "H100 full direct score workers complete; running fast LDS eval/summary once."
FULL_DIRECT_SRUN_WORKER=0 FULL_DIRECT_EVAL_ONLY=1 GPU_SLOTS=1 \
  bash "${SCRIPT_DIR}/run_4query_full_traj_direct_local.sh"
