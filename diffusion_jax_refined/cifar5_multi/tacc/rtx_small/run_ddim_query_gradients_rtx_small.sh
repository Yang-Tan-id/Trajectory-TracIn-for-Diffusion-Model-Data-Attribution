#!/usr/bin/env bash
#SBATCH -J cifar5-ddim-qgrad
#SBATCH -o cifar5-ddim-qgrad-%j.out
#SBATCH -e cifar5-ddim-qgrad-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_traj_tracin_ddim.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
  fi
fi

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP}"
else
  source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
  conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
fi

cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

NAMESPACE="${DDIM_NAMESPACE:-raw_nextckpt_school_traj_ddim_eta0_10x10}"
SAMPLE_ROOT_NAME="${DDIM_SAMPLE_ROOT_NAME:-sample_ddim_eta0}"

echo "CIFAR5 DDIM eta=0 DAS + TrajTracIn query gradients"
echo "repo=${REPO_ROOT}"
echo "namespace=${NAMESPACE}"
echo "sample_root=${SAMPLE_ROOT_NAME}"
echo "python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

# Phase 1: make the shared DDIM samples and both query-gradient families.
CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" \
  diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_traj_tracin_ddim.py \
  --execute \
  --include-das \
  --only-query-gradient \
  --namespace "${NAMESPACE}" \
  --sample-root-name "${SAMPLE_ROOT_NAME}" \
  --gpus 0,1 \
  --slots 2 \
  --gpu-per-node 2 \
  --max-parallel 2

# Phase 2: reuse the existing 10x10 residual-aware DAS train state and score.
MODEL_ROOT="diffusion_jax_refined/cifar5_multi/result/cifar5_multi_exp1/model/prompted_solo"
DAS_SOURCE="${MODEL_ROOT}/seed_42_train_gradient_ema_r20_das_10x10_residtrain/das"
DAS_TARGET="${MODEL_ROOT}/seed_42_train_gradient_${NAMESPACE}/das"
if [[ ! -f "${DAS_SOURCE}/global_gram_artifact.npz" ]]; then
  echo "Missing reusable DAS global Gram: ${DAS_SOURCE}" >&2
  exit 1
fi
mkdir -p "$(dirname "${DAS_TARGET}")"
if [[ -L "${DAS_TARGET}" ]]; then
  if [[ "$(readlink -f "${DAS_TARGET}")" != "$(readlink -f "${DAS_SOURCE}")" ]]; then
    echo "DAS target symlink points to the wrong source: ${DAS_TARGET}" >&2
    exit 1
  fi
elif [[ -e "${DAS_TARGET}" ]]; then
  echo "Refusing to replace existing DAS target: ${DAS_TARGET}" >&2
  exit 1
else
  ln -s "$(readlink -f "${DAS_SOURCE}")" "${DAS_TARGET}"
fi

DIFFUSION_TRAJECTORY_SAMPLER=ddim_eta0 \
LDS_TRAJECTORY_SAMPLER=ddim_eta0 \
CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" \
  diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_random_prompted_queries.py \
  --execute \
  --experiment cifar5_multi_exp1 \
  --size 10000 \
  --train-seed 42 \
  --epochs 200 \
  --num-queries 20 \
  --random-query-seed 0 \
  --initial-seed-start 1000 \
  --sample-root-name "${SAMPLE_ROOT_NAME}" \
  --artifact-namespace "${NAMESPACE}" \
  --namespace-query-gradient \
  --skip-sampling \
  --skip-query-gradient \
  --skip-traj-tracin \
  --skip-lds-eval \
  --das-batch-score \
  --gpus 0,1 \
  --slots 2 \
  --gpu-per-node 2 \
  --max-parallel 2 \
  --score-index-ranges 1-5000,5001-10000

RESULT_ROOT="diffusion_jax_refined/cifar5_multi/result/cifar5_multi_exp1"
ARCHIVE="${REPO_ROOT}/cifar5_ddim_eta0_query_and_das_scores.tar"
FILE_LIST="$(mktemp)"
trap 'rm -f "${FILE_LIST}"' EXIT

(
  cd "${RESULT_ROOT}"
  printf '%s\n' "${SAMPLE_ROOT_NAME}" > "${FILE_LIST}"
  find sample/cifar -type d \
    -name "seed_*_query_gradient_${NAMESPACE}" \
    -print >> "${FILE_LIST}"
  find attribution_score/prompted_solo -type d \
    -name "${NAMESPACE}" \
    -print >> "${FILE_LIST}"
  tar -cf "${ARCHIVE}" -T "${FILE_LIST}"
)

echo "[done] archive=${ARCHIVE}"
du -h "${ARCHIVE}"
