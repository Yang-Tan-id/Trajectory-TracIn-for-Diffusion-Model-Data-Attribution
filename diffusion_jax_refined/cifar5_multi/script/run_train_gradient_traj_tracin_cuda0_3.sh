#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIFAR5_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CIFAR5_ROOT}/../.." && pwd)"

cd "${REPO_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-cifar5_multi_exp1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export JAX_EPOCHS="${JAX_EPOCHS:-200}"
export CIFAR5_MULTI_SIZE="${CIFAR5_MULTI_SIZE:-10000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JAX_BFLOAT16="${JAX_BFLOAT16:-1}"
export JAX_PREFETCH_SIZE="${JAX_PREFETCH_SIZE:-1}"
export JAX_DATA_PARALLEL=0
export JAX_NUM_DEVICES=1
export LDS_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

export TRAJ_TRACIN_PROJ_DIM="${TRAJ_TRACIN_PROJ_DIM:-4096}"
export PROJECTED_CACHE_DIM="${PROJECTED_CACHE_DIM:-4096}"
export PROJECTED_DIMS="${PROJECTED_DIMS:-4096}"
export TRACIN_USE_SHARED_TRAIN_GRADIENT=1
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_SLOTS="${GPU_SLOTS:-4}"
export GPU_PER_NODE="${GPU_PER_NODE:-${GPU_SLOTS}}"

"${PYTHON_BIN}" "${CIFAR5_ROOT}/script/run_cifar5_multi_attribution_distributed.py" \
  --execute \
  --experiment "${EXPERIMENT_TAG}" \
  --size "${CIFAR5_MULTI_SIZE}" \
  --train-seed "${TRAIN_SEED}" \
  --epochs "${JAX_EPOCHS}" \
  --gpus "${GPU_IDS}" \
  --slots "${GPU_SLOTS}" \
  --gpu-per-node "${GPU_PER_NODE}" \
  --cpus-per-worker "${CPUS_PER_WORKER:-8}" \
  --slot-backend local \
  --only-train-gradient \
  --skip-das \
  ${EXTRA_CIFAR5_ATTR_ARGS:-}
