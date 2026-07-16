#!/usr/bin/env bash

train_gpu_list() {
  local text="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE:-0}}}"
  text="${text//,/ }"
  read -r -a TRAIN_GPUS <<<"${text}"
  if (( ${#TRAIN_GPUS[@]} == 0 )); then
    TRAIN_GPUS=(0)
  fi
}

train_run_prompted_solo() {
  local dataset_root="$1"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE:-0}}"
  echo "prompted_solo: class-conditioned training, gpu=${CUDA_VISIBLE_DEVICES}, seed=${TRAIN_SEED:-42}, experiment=${EXPERIMENT_TAG:-experiment1}"
  python "${dataset_root}/../common/prompted_jax_training.py" "${dataset_root}/dataset_config.py" --algorithm "${ALGORITHM:-shared}"
}

train_run_prompted_multi() {
  local dataset_root="$1"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS:-${CUDA_DEVICE:-0}}}"
  echo "prompted_multi: class-conditioned training, gpus=${CUDA_VISIBLE_DEVICES}, seed=${TRAIN_SEED:-42}, experiment=${EXPERIMENT_TAG:-experiment1}"
  python "${dataset_root}/../common/prompted_jax_training.py" "${dataset_root}/dataset_config.py" --algorithm "${ALGORITHM:-shared}"
}

train_run_unprompted_solo() {
  local dataset_root="$1"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE:-0}}"
  echo "unprompted_solo: unconditional training, gpu=${CUDA_VISIBLE_DEVICES}, seed=${TRAIN_SEED:-42}, experiment=${EXPERIMENT_TAG:-experiment1}"
  python "${dataset_root}/../common/prompted_jax_training.py" "${dataset_root}/dataset_config.py" --algorithm "${ALGORITHM:-shared}" --unconditional
}

train_run_unprompted_multi() {
  local dataset_root="$1"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS:-${CUDA_DEVICE:-0}}}"
  echo "unprompted_multi: unconditional training, gpus=${CUDA_VISIBLE_DEVICES}, seed=${TRAIN_SEED:-42}, experiment=${EXPERIMENT_TAG:-experiment1}"
  python "${dataset_root}/../common/prompted_jax_training.py" "${dataset_root}/dataset_config.py" --algorithm "${ALGORITHM:-shared}" --unconditional
}
