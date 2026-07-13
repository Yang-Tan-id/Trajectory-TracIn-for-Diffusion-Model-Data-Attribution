#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-runs/cifar2/ddpm}"
DATASET="${DATASET:-runs/cifar2/gen}"
OUTPUT="${OUTPUT:-runs/cifar2/query_grads.npy}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

args=()
if [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi

python3 -m torch_das.gradients \
  --model-dir "$MODEL_DIR" \
  --dataset "$DATASET" \
  --dataset-kind cifar2 \
  --dataset-type gen \
  --center-crop \
  --num-timesteps "${TIMESTEPS:-10}" \
  --projection-dim "${PROJECTION_DIM:-4096}" \
  --output "$OUTPUT" \
  --device "$DEVICE" \
  "${args[@]}"
