#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-../diffusion_jax_refined/dataset/cifar2/cifar-10-batches-py}"
INDEX_DIR="${INDEX_DIR:-runs/cifar2/lds/indices}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/cifar2/lds/models}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-200}"
START="${START:-0}"
END="${END:-63}"
SEEDS="${SEEDS:-0,1,2}"

IFS=',' read -ra seed_list <<< "$SEEDS"
for seed in "${seed_list[@]}"; do
  for subset in $(seq "$START" "$END"); do
    subset_padded="$(printf "%04d" "$subset")"
    echo "training LDS subset=${subset_padded} seed=${seed}"
    python3 -m torch_das.train \
      --dataset "$DATASET" \
      --dataset-kind cifar2 \
      --index-path "$INDEX_DIR/sub-idx-${subset}.pkl" \
      --config configs/cifar2_unet_das.json \
      --output-dir "$OUTPUT_ROOT/subset_${subset_padded}/seed_${seed}" \
      --center-crop \
      --random-flip \
      --batch-size "$BATCH_SIZE" \
      --num-epochs "$EPOCHS" \
      --checkpointing-steps 0 \
      --seed "$seed" \
      --device "$DEVICE"
  done
done
