#!/usr/bin/env bash
#SBATCH --job-name=cifar2-lds-1-16
#SBATCH --partition=h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-lds-%j.out
#SBATCH --error=cifar2-lds-%j.err

set -euo pipefail

# Submit from anywhere:
#   sbatch -A <allocation> diffusion_jax_refined/cifar2/tacc/01_train_lds_h100.sh
#
# Activate the project environment before sbatch, or set ENV_SETUP to a shell
# file that activates it:
#   ENV_SETUP=$HOME/envs/trajectory-tracin.sh sbatch -A <allocation> ...

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-5000}"
LDS_EPOCHS="${LDS_EPOCHS:-200}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/lds_${SLURM_JOB_ID}"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
fi
command -v python >/dev/null || {
  echo "python is not available; activate the project environment or set ENV_SETUP." >&2
  exit 1
}

mkdir -p "${LOG_ROOT}"
cd "${CIFAR2_ROOT}"

echo "Training LDS seeds 1-16 with ${LDS_M} subsets/seed and k=${LDS_K}"
echo "Experiment: ${EXPERIMENT_TAG}"
echo "Logs: ${LOG_ROOT}"

if [[ "${ALLOW_OVERWRITE}" != "1" ]]; then
  for seed in $(seq 1 16); do
    model_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/m_${LDS_M}_k_${LDS_K}_seed_${seed}"
    if [[ -e "${model_dir}" ]]; then
      echo "Refusing to overwrite existing LDS output: ${model_dir}" >&2
      echo "Use a new experiment tag, remove/archive the old output, or set ALLOW_OVERWRITE=1." >&2
      exit 1
    fi
  done
fi

pids=()
for seed in $(seq 1 16); do
  log="${LOG_ROOT}/seed_${seed}.log"
  echo "Launching LDS seed ${seed} -> ${log}"
  srun --exclusive --exact \
    --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    env \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      LDS_M="${LDS_M}" \
      LDS_K="${LDS_K}" \
      LDS_SAMPLE_RANDOM_SEED="${seed}" \
      LDS_EPOCHS="${LDS_EPOCHS}" \
      LDS_DEVICE=gpu \
      LDS_NUM_DEVICES=1 \
    bash scripts/03_lds_training.sh >"${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed )); then
  echo "At least one LDS seed failed. Check ${LOG_ROOT}." >&2
  exit 1
fi

echo "All LDS seeds completed successfully."
