#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-../diffusion_jax_refined/dataset/cifar2/cifar-10-batches-py}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/cifar2/lds/indices}"
NUM_SUBSETS="${NUM_SUBSETS:-64}"
SUBSET_SIZE="${SUBSET_SIZE:-5000}"
SEED="${SEED:-0}"

python3 -m torch_das.make_lds_subsets \
  --dataset "$DATASET" \
  --dataset-kind cifar2 \
  --center-crop \
  --output-dir "$OUTPUT_DIR" \
  --num-subsets "$NUM_SUBSETS" \
  --subset-size "$SUBSET_SIZE" \
  --seed "$SEED"
