#!/usr/bin/env bash
#SBATCH --job-name=cifar2-lds-k8000
#SBATCH --partition=h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00
#SBATCH --output=cifar2-lds-k8000-%j.out
#SBATCH --error=cifar2-lds-k8000-%j.err

set -euo pipefail

# Train reusable CIFAR2 LDS subset models with a gentler perturbation than
# k=5000. Defaults are intentionally a pilot: 16 LDS seeds × 20 subsets each.
# For the full run, submit with LDS_M=50.

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR2_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar2"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
LDS_M="${LDS_M:-20}"
LDS_K="${LDS_K:-8000}"
LDS_SEEDS="${LDS_SEEDS:-$(seq -s ' ' 1 16)}"
LDS_EPOCHS="${LDS_EPOCHS:-200}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/tacc_logs/lds_k${LDS_K}_m${LDS_M}_${SLURM_JOB_ID}"

unset PYTHONPATH
if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-${SCRATCH}/conda-envs/trajectory-tracin}"
  # shellcheck disable=SC1090
  source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
fi

[[ -d "${CIFAR2_ROOT}" ]] || {
  echo "CIFAR2 root not found: ${CIFAR2_ROOT}" >&2
  echo "Submit this job from the repository root or set REPO_ROOT explicitly." >&2
  exit 1
}
command -v python >/dev/null || { echo "python is unavailable" >&2; exit 1; }

mkdir -p "${LOG_ROOT}"
cd "${CIFAR2_ROOT}"

echo "Training CIFAR2 LDS pilot/full models"
echo "Experiment: ${EXPERIMENT_TAG}"
echo "LDS_M=${LDS_M}, LDS_K=${LDS_K}, LDS_SEEDS=${LDS_SEEDS}, LDS_EPOCHS=${LDS_EPOCHS}"
echo "Logs: ${LOG_ROOT}"

if [[ "${ALLOW_OVERWRITE}" != "1" ]]; then
  for seed in ${LDS_SEEDS}; do
    model_dir="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/lds_model/m_${LDS_M}_k_${LDS_K}_seed_${seed}"
    if [[ -e "${model_dir}" ]]; then
      echo "Refusing to overwrite existing LDS output: ${model_dir}" >&2
      echo "Use ALLOW_OVERWRITE=1 only if you intentionally want to replace it." >&2
      exit 1
    fi
  done
fi

pids=()
slot=0
for seed in ${LDS_SEEDS}; do
  log="${LOG_ROOT}/seed_${seed}.log"
  echo "Launching LDS seed ${seed} -> ${log}"
  ibrun -n 1 -o "${slot}" \
    env CUDA_VISIBLE_DEVICES="$((slot % 4))" \
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

echo "All requested LDS seeds completed successfully."
