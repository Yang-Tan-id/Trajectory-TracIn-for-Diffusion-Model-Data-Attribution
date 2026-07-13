#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-runs/cifar10/ddpm}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/cifar10/gen}"
DEVICE="${DEVICE:-cuda}"
NUM_IMAGES="${NUM_IMAGES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
STEPS="${STEPS:-50}"

python3 -m torch_das.generate \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --num-images "$NUM_IMAGES" \
  --batch-size "$BATCH_SIZE" \
  --num-inference-steps "$STEPS" \
  --device "$DEVICE"
