#!/usr/bin/env bash
#SBATCH --job-name=cifar10-baseline-lds
#SBATCH --partition=gh
#SBATCH --account=CCR25021
#SBATCH --nodes=17
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=cifar10-baseline-lds-%j.out
#SBATCH --error=cifar10-baseline-lds-%j.err

set -euo pipefail

# Combined Vista job:
#   slot 0  -> prompted baseline on full CIFAR-10 (train seed 42)
#   slots 1-16 -> LDS seeds 1-16 in parallel (50 subsets of k=40000 each)
#
# Submit from the repository root:
#   sbatch diffusion_jax_refined/cifar10/vista/00_train_baseline_and_lds_vista.sh
#
# Optional overrides:
#   EXPERIMENT_TAG=experiment1_42 ALLOW_OVERWRITE=1 sbatch ...

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CIFAR10_ROOT="${REPO_ROOT}/diffusion_jax_refined/cifar10"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1_42}"
TRAIN_SEED="${TRAIN_SEED:-42}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
LDS_M="${LDS_M:-50}"
LDS_K="${LDS_K:-40000}"
LDS_EPOCHS="${LDS_EPOCHS:-200}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
LOG_ROOT="${CIFAR10_ROOT}/result/${EXPERIMENT_TAG}/vista_logs/baseline_lds_${SLURM_JOB_ID}"
BASELINE_CKPT="${CIFAR10_ROOT}/result/${EXPERIMENT_TAG}/model/prompted_jax/seed_${TRAIN_SEED}_epoch_$(printf '%04d' "${JAX_EPOCHS}").ckpt"

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
    echo "Set ENV_SETUP to a shell snippet that activates your conda env." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_PATH}"
fi

[[ -d "${CIFAR10_ROOT}" ]] || {
  echo "CIFAR10 root not found: ${CIFAR10_ROOT}" >&2
  echo "Submit this job from the repository root or set REPO_ROOT explicitly." >&2
  exit 1
}
command -v python >/dev/null || {
  echo "python is not available; activate the project environment or set ENV_SETUP." >&2
  exit 1
}

mkdir -p "${LOG_ROOT}"
cd "${CIFAR10_ROOT}"

echo "Vista gh job: baseline + LDS"
echo "Experiment     : ${EXPERIMENT_TAG}"
echo "Baseline seed  : ${TRAIN_SEED}"
echo "Baseline ckpt  : ${BASELINE_CKPT}"
echo "LDS config     : m=${LDS_M}, k=${LDS_K}, seeds=1-16, epochs=${LDS_EPOCHS}"
echo "Logs           : ${LOG_ROOT}"

if [[ "${ALLOW_OVERWRITE}" != "1" ]]; then
  if [[ -e "${BASELINE_CKPT}" ]]; then
    echo "Refusing to overwrite existing baseline checkpoint: ${BASELINE_CKPT}" >&2
    echo "Use a new EXPERIMENT_TAG, archive the old output, or set ALLOW_OVERWRITE=1." >&2
    exit 1
  fi
  for seed in $(seq 1 16); do
    model_dir="${CIFAR10_ROOT}/result/${EXPERIMENT_TAG}/lds_model/m_${LDS_M}_k_${LDS_K}_seed_${seed}"
    if [[ -e "${model_dir}" ]]; then
      echo "Refusing to overwrite existing LDS output: ${model_dir}" >&2
      echo "Use a new EXPERIMENT_TAG, archive the old output, or set ALLOW_OVERWRITE=1." >&2
      exit 1
    fi
  done
fi

echo "Phase 1/2: training prompted baseline on full CIFAR-10 (slot 0)"
ibrun -n 1 -o 0 \
  env CUDA_VISIBLE_DEVICES=0 \
    EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
    TRAIN_SEED="${TRAIN_SEED}" \
    JAX_EPOCHS="${JAX_EPOCHS}" \
    JAX_DATA_PARALLEL=0 \
    JAX_NUM_DEVICES=1 \
    JAX_DEVICE=gpu \
  bash scripts/00_train.sh >"${LOG_ROOT}/baseline_seed_${TRAIN_SEED}.log" 2>&1

[[ -f "${BASELINE_CKPT}" ]] || {
  echo "Baseline checkpoint not found after training: ${BASELINE_CKPT}" >&2
  exit 1
}
echo "Baseline checkpoint ready: ${BASELINE_CKPT}"

echo "Phase 2/2: training LDS seeds 1-16 on slots 1-16"
pids=()
for seed in $(seq 1 16); do
  log="${LOG_ROOT}/lds_seed_${seed}.log"
  echo "Launching LDS seed ${seed} on slot ${seed} -> ${log}"
  ibrun -n 1 -o "${seed}" \
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

echo "Baseline and all LDS seeds completed successfully."

