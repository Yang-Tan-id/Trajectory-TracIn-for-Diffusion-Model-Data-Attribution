#!/usr/bin/env bash
#SBATCH --job-name=cifar2-lds-50pct
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=cifar2-lds-50pct-%j.out
#SBATCH --error=cifar2-lds-50pct-%j.err

set -euo pipefail

# Train CIFAR2 LDS models on 50% of the attribution dataset.
# CIFAR2 has 10,000 attribution examples, so 50% means LDS_K=5000.
#
# Submit from this repository root on Vista:
#   sbatch diffusion_jax_refined/cifar2/vista/00_train_lds_50pct_vista.sh

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-5000}"
LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 1 16)}"
LDS_EPOCHS="${LDS_EPOCHS:-200}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/lds_50pct_m${LDS_M}_k${LDS_K}_${SLURM_JOB_ID}"

unset PYTHONPATH
if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-${SCRATCH}/conda-envs/trajectory-tracin}"
  if [[ -f "${SCRATCH}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh under \${SCRATCH}/miniforge3 or \${HOME}/miniforge3." >&2
    echo "Set ENV_SETUP to a shell snippet that activates your conda env." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_PATH}"
fi

[[ -d "${CIFAR2_ROOT}" ]] || {
  echo "CIFAR2 root not found: ${CIFAR2_ROOT}" >&2
  echo "Submit from the repository root or set REPO_ROOT explicitly." >&2
  exit 1
}
command -v python >/dev/null || { echo "python is unavailable" >&2; exit 1; }

mkdir -p "${LOG_ROOT}"
cd "${CIFAR2_ROOT}"

echo "Vista gh job: CIFAR2 LDS 50%"
echo "REPO_ROOT      : ${REPO_ROOT}"
echo "CIFAR2_ROOT    : ${CIFAR2_ROOT}"
echo "EXPERIMENT_TAG : ${EXPERIMENT_TAG}"
echo "LDS config     : m=${LDS_M}, k=${LDS_K}, seeds=${LDS_SEEDS}, epochs=${LDS_EPOCHS}"
echo "Logs           : ${LOG_ROOT}"

if [[ "${ALLOW_OVERWRITE}" != "1" ]]; then
  for seed in ${LDS_SEEDS}; do
    model_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/m_${LDS_M}_k_${LDS_K}_seed_${seed}"
    if [[ -e "${model_dir}" ]]; then
      echo "Refusing to overwrite existing LDS output: ${model_dir}" >&2
      echo "Use a new EXPERIMENT_TAG or set ALLOW_OVERWRITE=1 intentionally." >&2
      exit 1
    fi
  done
fi

pids=()
slot=0
for seed in ${LDS_SEEDS}; do
  log="${LOG_ROOT}/lds_seed_${seed}.log"
  echo "Launching LDS seed ${seed} on Vista slot ${slot} -> ${log}"
  ibrun -n 1 -o "${slot}" \
    env CUDA_VISIBLE_DEVICES=0 \
      EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
      LDS_M="${LDS_M}" \
      LDS_K="${LDS_K}" \
      LDS_SAMPLE_RANDOM_SEED="${seed}" \
      LDS_EPOCHS="${LDS_EPOCHS}" \
      LDS_DEVICE=gpu \
      LDS_NUM_DEVICES=1 \
    bash scripts/03_lds_training.sh >"${log}" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

if (( failed )); then
  echo "At least one LDS seed failed. Check ${LOG_ROOT}." >&2
  exit 1
fi

echo "All CIFAR2 50% LDS seeds completed successfully."
