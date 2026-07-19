#!/usr/bin/env bash
#SBATCH --job-name=cifar10-lds-only
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=cifar10-lds-only-%j.out
#SBATCH --error=cifar10-lds-only-%j.err

set -euo pipefail

# Retry LDS seeds 1-16 when the baseline checkpoint already exists.
# Uses one ibrun -n 16 (no nested/concurrent ibrun calls).
#
# Submit from the repository root:
#   ALLOW_OVERWRITE=1 sbatch diffusion_jax_refined/cifar10/vista/01_train_lds_only_vista.sh

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR10_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar10"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
TRAIN_SEED="${TRAIN_SEED:-42}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-40000}"
LDS_EPOCHS="${LDS_EPOCHS:-200}"
LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 1 16)}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
LOG_ROOT="${CIFAR10_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/lds_only_${SLURM_JOB_ID}"
BASELINE_CKPT="${CIFAR10_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_jax/seed_${TRAIN_SEED}_epoch_$(printf '%04d' "${JAX_EPOCHS}").ckpt"
WORKER_SCRIPT="${CIFAR10_ROOT}/vista/01_train_lds_only_worker.sh"

unset PYTHONPATH
if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/11227/kg35276/conda-envs/trajectory-tracin}"
  if [[ -f "${SCRATCH}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh under \${SCRATCH}/miniforge3 or \${HOME}/miniforge3." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_PATH}"
fi

[[ -d "${CIFAR10_ROOT}" ]] || {
  echo "CIFAR10 root not found: ${CIFAR10_ROOT}" >&2
  exit 1
}
[[ -f "${BASELINE_CKPT}" ]] || {
  echo "Baseline checkpoint not found: ${BASELINE_CKPT}" >&2
  echo "Run 00_train_baseline_and_lds_vista.sh first, or fix EXPERIMENT_TAG/TRAIN_SEED/JAX_EPOCHS." >&2
  exit 1
}
[[ -f "${WORKER_SCRIPT}" ]] || {
  echo "Worker script not found: ${WORKER_SCRIPT}" >&2
  exit 1
}

mkdir -p "${LOG_ROOT}"
cd "${CIFAR10_ROOT}"

read -r -a seed_list <<<"${LDS_SEEDS}"
if ((${#seed_list[@]} != 16)); then
  echo "This retry script expects exactly 16 LDS seeds (got ${#seed_list[@]})." >&2
  echo "Set LDS_SEEDS=\"1 2 ... 16\" or use the combined launcher." >&2
  exit 1
fi

if [[ "${ALLOW_OVERWRITE}" != "1" ]]; then
  for seed in "${seed_list[@]}"; do
    model_dir="${CIFAR10_ROOT}/result/${EXPERIMENT_TAG}/lds_model/m_${LDS_M}_k_${LDS_K}_seed_${seed}"
    if [[ -e "${model_dir}" ]]; then
      echo "Refusing to overwrite existing LDS output: ${model_dir}" >&2
      echo "Set ALLOW_OVERWRITE=1 to replace incomplete runs." >&2
      exit 1
    fi
  done
fi

echo "LDS-only Vista gh job"
echo "Baseline ckpt : ${BASELINE_CKPT}"
echo "LDS seeds     : ${LDS_SEEDS}"
echo "Logs          : ${LOG_ROOT}"

export CIFAR10_ROOT LOG_ROOT BASELINE_CKPT EXPERIMENT_TAG
export LDS_M LDS_K LDS_EPOCHS
export LDS_SEED_1="${seed_list[0]}"
export LDS_SEED_2="${seed_list[1]}"
export LDS_SEED_3="${seed_list[2]}"
export LDS_SEED_4="${seed_list[3]}"
export LDS_SEED_5="${seed_list[4]}"
export LDS_SEED_6="${seed_list[5]}"
export LDS_SEED_7="${seed_list[6]}"
export LDS_SEED_8="${seed_list[7]}"
export LDS_SEED_9="${seed_list[8]}"
export LDS_SEED_10="${seed_list[9]}"
export LDS_SEED_11="${seed_list[10]}"
export LDS_SEED_12="${seed_list[11]}"
export LDS_SEED_13="${seed_list[12]}"
export LDS_SEED_14="${seed_list[13]}"
export LDS_SEED_15="${seed_list[14]}"
export LDS_SEED_16="${seed_list[15]}"

ibrun -n 16 bash "${WORKER_SCRIPT}"
echo "All requested LDS seeds completed successfully."
