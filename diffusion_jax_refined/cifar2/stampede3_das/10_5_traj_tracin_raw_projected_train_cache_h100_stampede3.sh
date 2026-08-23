#!/usr/bin/env bash
#SBATCH -J cifar2-s3-projtrain-raw
#SBATCH -o cifar2-s3-projtrain-raw-%j.out
#SBATCH -e cifar2-s3-projtrain-raw-%j.err
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
    "${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/stampede3_das"; do
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

PROJECTED_TRAIN_SCRIPT="${CIFAR2_ROOT}/tacc/h100/projected_traj_tracin_train_cache_array_h100.sh"
if [[ ! -f "${PROJECTED_TRAIN_SCRIPT}" ]]; then
  echo "Missing projected train cache helper: ${PROJECTED_TRAIN_SCRIPT}" >&2
  exit 2
fi

export TRAJ_PARAMETER_SOURCE="${TRAJ_PARAMETER_SOURCE:-raw}"
export TRACIN_PARAMETER_SOURCE="${TRACIN_PARAMETER_SOURCE:-${TRAJ_PARAMETER_SOURCE}}"
export TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-trajectory_next_checkpoint_noise_mse}"
export PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
export TRAIN_SCORE_INDEX_RANGES="${TRAIN_SCORE_INDEX_RANGES:-1-10000}"
export TRAIN_CACHE_TASK_SET="${TRAIN_CACHE_TASK_SET:-all}"
export GPU_SLOTS="${GPU_SLOTS:-16}"
export GPU_PER_NODE="${GPU_PER_NODE:-4}"
export PROJECTED_ARTIFACT_DIR_NAME="${PROJECTED_ARTIFACT_DIR_NAME:-projected_traj_tracin_artifacts_${TRAJ_PARAMETER_SOURCE}}"

echo "Stampede3 raw projected Traj-TracIn train cache"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; parameter_source=${TRAJ_PARAMETER_SOURCE}; objective=${TRAJ_QUERY_OBJECTIVE}"
echo "projected_cache_dim=${PROJECTED_CACHE_DIM}; train_range=${TRAIN_SCORE_INDEX_RANGES}; task_set=${TRAIN_CACHE_TASK_SET}"
echo "projected_artifact_dir_name=${PROJECTED_ARTIFACT_DIR_NAME}"

bash "${PROJECTED_TRAIN_SCRIPT}"
