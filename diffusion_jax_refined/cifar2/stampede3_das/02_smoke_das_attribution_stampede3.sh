#!/usr/bin/env bash
#SBATCH -J cifar2-s3-das-smoke
#SBATCH -o cifar2-s3-das-smoke-%j.out
#SBATCH -e cifar2-s3-das-smoke-%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH -t 00:30:00

set -euo pipefail

SCRIPT_DIR="${STAMPEDE3_DAS_DIR:-}"
if [[ -z "${SCRIPT_DIR}" || ! -f "${SCRIPT_DIR}/_stampede3_das_lib.sh" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
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

SMOKE_SAMPLE_MODE="${SMOKE_SAMPLE_MODE:-prompted_solo}"
SMOKE_SCORE_MODE="${SMOKE_SCORE_MODE:-prompted_solo}"
SMOKE_QUERY="${SMOKE_QUERY:-horse}"
SMOKE_QUERY_ENV="${SMOKE_QUERY}"
SMOKE_INITIAL_SEED="${SMOKE_INITIAL_SEED:-3}"
SMOKE_UNPROMPTED="${SMOKE_UNPROMPTED:-0}"
if [[ "${SMOKE_UNPROMPTED}" == "1" ]]; then
  SMOKE_SAMPLE_MODE="${SMOKE_SAMPLE_MODE:-unprompted_solo}"
  SMOKE_SCORE_MODE="${SMOKE_SCORE_MODE:-unprompted_solo}"
  SMOKE_QUERY="unprompted"
  SMOKE_QUERY_ENV="unconditional"
fi

DAS_DAMPING_SWEEP=1
DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES:-0.01}"
SMOKE_SCORE_INDEX_RANGES="${SMOKE_SCORE_INDEX_RANGES:-1-64}"
SMOKE_DAS_PROJ_DIM="${SMOKE_DAS_PROJ_DIM:-512}"
SMOKE_DAS_GRAD_BATCH_SIZE="${SMOKE_DAS_GRAD_BATCH_SIZE:-8}"
JAX_EPOCHS="${JAX_EPOCHS:-200}"
export DAS_DAMPING_SWEEP DAS_DAMPING_SWEEP_VALUES

LOG_ROOT="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/stampede3_das_logs/02_smoke_das_attribution/${SLURM_JOB_ID:-local}"
mkdir -p "${LOG_ROOT}"

prompt_tag="$(path_tag "${SMOKE_QUERY_ENV}")"
printf -v seed_tag "%06d" "${SMOKE_INITIAL_SEED}"
printf -v ckpt_stem "seed_%s_epoch_%04d" "${TRAIN_SEED}" "${JAX_EPOCHS}"
sample_run_root="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/sample/cifar/prompt_${prompt_tag}/model_${SMOKE_SAMPLE_MODE}__ckpt_${ckpt_stem}"
mkdir -p "${sample_run_root}"
sample_done_file="${sample_run_root}/seed_${seed_tag}/trajectory_xt.npy"
sample_lock_dir="${sample_run_root}/.sample_seed_${seed_tag}.lock"
log="${LOG_ROOT}/task_smoke__das__${SMOKE_SCORE_MODE}__$(path_tag "${SMOKE_QUERY}")__seed_${SMOKE_INITIAL_SEED}.log"

echo "Job 02 smoke Stampede3 DAS"
echo "experiment=${EXPERIMENT_TAG}; train_seed=${TRAIN_SEED}; sample_mode=${SMOKE_SAMPLE_MODE}; score_mode=${SMOKE_SCORE_MODE}; query=${SMOKE_QUERY}; seed=${SMOKE_INITIAL_SEED}; lambdas=${DAS_DAMPING_SWEEP_VALUES}; ranges=${SMOKE_SCORE_INDEX_RANGES}; proj_dim=${SMOKE_DAS_PROJ_DIM}"
echo "logs=${LOG_ROOT}"

run_gpu_slot 0 env \
  CUDA_VISIBLE_DEVICES=0 \
  GPU_IDS=0 \
  JAX_NUM_DEVICES=1 \
  PYTHONUNBUFFERED=1 \
  ATTRIBUTION_TQDM_MININTERVAL="${ATTRIBUTION_TQDM_MININTERVAL:-1}" \
  ATTRIBUTION_TQDM_LEAVE="${ATTRIBUTION_TQDM_LEAVE:-1}" \
  SCORE_INDEX_RANGES="${SMOKE_SCORE_INDEX_RANGES}" \
  DAS_PROJ_DIM="${SMOKE_DAS_PROJ_DIM}" \
  DAS_GRAD_BATCH_SIZE="${SMOKE_DAS_GRAD_BATCH_SIZE}" \
  CIFAR2_ROOT="${CIFAR2_ROOT}" \
  REFINE_ROOT="${REFINE_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
  TRAIN_SEED="${TRAIN_SEED}" \
  SAMPLE_MODEL_MODE="${SMOKE_SAMPLE_MODE}" \
  UNPROMPTED_SAMPLE_MODEL_MODE="${SMOKE_SAMPLE_MODE}" \
  ATTRIBUTION_SAMPLE_MODEL_MODE="${SMOKE_SAMPLE_MODE}" \
  ATTRIBUTION_SCORE_MODEL_MODE="${SMOKE_SCORE_MODE}" \
  UNPROMPTED_SCORE_MODEL_MODE="${SMOKE_SCORE_MODE}" \
  QUERY="${SMOKE_QUERY_ENV}" \
  INITIAL_SEED="${SMOKE_INITIAL_SEED}" \
  SAMPLE_SEED="${SMOKE_INITIAL_SEED}" \
  SAMPLE_SEEDS="${SMOKE_INITIAL_SEED}" \
  SAMPLE_DONE_FILE="${sample_done_file}" \
  SAMPLE_LOCK_DIR="${sample_lock_dir}" \
  SAMPLE_LOCK_WAIT_SECONDS="${SAMPLE_LOCK_WAIT_SECONDS:-21600}" \
  UNPROMPTED="${SMOKE_UNPROMPTED}" \
  ALGORITHM=das \
  bash "${SCRIPT_DIR}/02_das_attribution_task_stampede3.sh" \
  >"${log}" 2>&1

missing=0
for lambda in ${DAS_DAMPING_SWEEP_VALUES}; do
  lambda_tag="$(damping_tag "${lambda}")"
  if [[ "${SMOKE_UNPROMPTED}" == "1" || "${SMOKE_SCORE_MODE}" == unprompted_* ]]; then
    score_file="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/attribution_score/${SMOKE_SCORE_MODE}/train_seed_${TRAIN_SEED}/unprompted/initial_seed_${SMOKE_INITIAL_SEED}/das_unprompted/lambda_${lambda_tag}/scores.npy"
  else
    score_file="${CIFAR2_ROOT}/result/${EXPERIMENT_TAG}/attribution_score/${SMOKE_SCORE_MODE}/train_seed_${TRAIN_SEED}/query_$(path_tag "${SMOKE_QUERY}")/initial_seed_${SMOKE_INITIAL_SEED}/das/lambda_${lambda_tag}/scores.npy"
  fi
  score_base_dir="$(dirname "$(dirname "${score_file}")")"
  score_lambda_dir="$(basename "$(dirname "${score_file}")")"
  mapfile -t score_matches < <(compgen -G "${score_base_dir}/${score_lambda_dir}*/scores.npy" | sort)
  if [[ "${#score_matches[@]}" -eq 0 ]]; then
    echo "Missing smoke DAS score artifact for lambda=${lambda}: ${score_file}" >&2
    missing=$((missing + 1))
  else
    echo "Found smoke DAS score artifact: ${score_matches[0]}"
  fi
done

if (( missing > 0 )); then
  echo "Smoke DAS attribution failed: missing ${missing} score artifacts." >&2
  exit 1
fi

echo "Smoke DAS attribution complete."
