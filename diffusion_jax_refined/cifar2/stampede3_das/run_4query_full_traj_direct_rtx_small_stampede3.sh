#!/usr/bin/env bash
#SBATCH -J cifar2-full-direct-rtx
#SBATCH -o cifar2-full-direct-rtx-%j.out
#SBATCH -e cifar2-full-direct-rtx-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
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

# Reuse the same Stampede3 runtime setup as the DAS/original attribution jobs.
# This activates the conda env and resolves REPO_ROOT/CIFAR2_ROOT/REFINE_ROOT.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_stampede3_das_lib.sh"
stampede3_das_init
cd "${REPO_ROOT}"

export STAMPEDE3_DAS_DIR="${SCRIPT_DIR}"
export GPU_SLOTS="${GPU_SLOTS:-2}"
export TRACIN_USE_LR_WEIGHTS="${TRACIN_USE_LR_WEIGHTS:-0}"
export TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-trajectory_next_checkpoint_noise_mse}"
export TRAJ_PARAMETER_SOURCE="${TRAJ_PARAMETER_SOURCE:-raw}"
export TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-2}"
export TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-4}"
export TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS="${TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS:-1}"
export TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS="${TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS:-10}"

echo "RTX-small full direct Traj-TracIn wrapper"
echo "job=${SLURM_JOB_ID:-local}; node=${SLURM_NODELIST:-local}; gpu_slots=${GPU_SLOTS}; time_limit=48:00:00"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; lr_weights=${TRACIN_USE_LR_WEIGHTS}"
echo "aggregate_train_timestamps=${TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS}; aggregate_num_timesteps=${TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS}"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L || true
fi

exec bash "${SCRIPT_DIR}/run_4query_full_traj_direct_local.sh"
