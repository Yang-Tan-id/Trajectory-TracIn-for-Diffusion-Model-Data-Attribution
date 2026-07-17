#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-../diffusion_jax_refined/dataset/cifar2/cifar-10-batches-py}"
DEVICE="${DEVICE:-cuda}"
TIMESTEPS="${TIMESTEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"

python3 -m torch_das.eval_loss \
  --model-dir "${MODEL_DIR:-runs/cifar2/ddpm}" \
  --dataset "$DATASET" \
  --dataset-kind cifar2 \
  --dataset-type train \
  --center-crop \
  --batch-size "$BATCH_SIZE" \
  --num-timesteps "$TIMESTEPS" \
  --max-samples "$MAX_SAMPLES" \
  --output "${LOSS_OUTPUT:-runs/cifar2/error/train_losses.pkl}" \
  --device "$DEVICE"

python3 -m torch_das.make_error_train \
  --losses "${LOSS_OUTPUT:-runs/cifar2/error/train_losses.pkl}" \
  --output "${ERROR_OUTPUT:-runs/cifar2/error/error_train.npy}"
