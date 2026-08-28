#!/usr/bin/env bash
#SBATCH -J cifar2-s3-eval-raw-next
#SBATCH -o cifar2-s3-eval-raw-next-%j.out
#SBATCH -e cifar2-s3-eval-raw-next-%j.err
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00

set -euo pipefail

export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_67}"
export TRAIN_SEED="${TRAIN_SEED:-67}"

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

export LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-25}"
export LDS_K="${LDS_K:-2500}"
export LDS_M="${LDS_M:-64}"
export LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 0 7)}"
export LDS_TARGETS="${LDS_TARGETS:-endpoint_contarfactual traj_contarfactual simple_loss trajectory_state_mse}"
export EVAL_ALGORITHMS="${EVAL_ALGORITHMS:-traj_tracin_trajectory_next_checkpoint_noise_mse_raw traj_tracin_trajectory_next_checkpoint_noise_mse_raw_q_l2 traj_tracin_trajectory_next_checkpoint_noise_mse_raw_t_l2 traj_tracin_trajectory_next_checkpoint_noise_mse_raw_qt_l2}"
export UNPROMPTED_EVAL_ALGORITHMS="${UNPROMPTED_EVAL_ALGORITHMS:-traj_tracin_trajectory_next_checkpoint_noise_mse_unprompted_raw traj_tracin_trajectory_next_checkpoint_noise_mse_unprompted_raw_q_l2 traj_tracin_trajectory_next_checkpoint_noise_mse_unprompted_raw_t_l2 traj_tracin_trajectory_next_checkpoint_noise_mse_unprompted_raw_qt_l2}"
export PRED_TAG="${PRED_TAG:-pred_kept_sign_m1}"
export LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"
export LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"
export LDS_SIMPLE_LOSS_TIMESTEPS="${LDS_SIMPLE_LOSS_TIMESTEPS:-$(seq -s, 0 999)}"
export LDS_SIMPLE_LOSS_NOISE_SEEDS="${LDS_SIMPLE_LOSS_NOISE_SEEDS:-0}"
export LDS_SIMPLE_LOSS_NUM_MC="${LDS_SIMPLE_LOSS_NUM_MC:-1000}"
export LDS_SIMPLE_LOSS_MC_SEED="${LDS_SIMPLE_LOSS_MC_SEED:-0}"

echo "Stampede3 LDS eval for raw Traj-TracIn next-checkpoint scores"
echo "experiment=${EXPERIMENT_TAG}; targets=${LDS_TARGETS}; algorithms=${EVAL_ALGORITHMS}; unprompted_algorithms=${UNPROMPTED_EVAL_ALGORITHMS}; pct=${LDS_DATASET_PERCENTAGE}; seeds=${LDS_SEEDS}"

bash "${SCRIPT_DIR}/03_dtrak_endtracin_lds_eval_report_stampede3.sh"
