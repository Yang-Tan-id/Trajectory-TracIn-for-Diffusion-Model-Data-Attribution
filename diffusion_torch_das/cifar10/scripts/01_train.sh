#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-../diffusion_jax_refined/dataset/cifar10/cifar-10-batches-py}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/cifar10/ddpm}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-200}"

python3 -m torch_das.train \
  --dataset "$DATASET" \
  --dataset-kind cifar10 \
  --config configs/cifar10_unet_das.json \
  --output-dir "$OUTPUT_DIR" \
  --center-crop \
  --random-flip \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$EPOCHS" \
  --checkpointing-steps 500 \
  --device "$DEVICE"
