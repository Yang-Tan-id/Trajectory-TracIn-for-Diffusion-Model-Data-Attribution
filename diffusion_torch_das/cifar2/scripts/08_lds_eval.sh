#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-runs/cifar2/lds/models}"
LOSS_ROOT="${LOSS_ROOT:-runs/cifar2/lds/losses}"
DATASET="${DATASET:-runs/cifar2/gen}"
DATASET_TYPE="${DATASET_TYPE:-gen}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TIMESTEPS="${TIMESTEPS:-1000}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
START="${START:-0}"
END="${END:-63}"
SEEDS="${SEEDS:-0,1,2}"
EVAL_SEEDS="${EVAL_SEEDS:-0}"

IFS=',' read -ra seed_list <<< "$SEEDS"
IFS=',' read -ra eval_seed_list <<< "$EVAL_SEEDS"
for seed in "${seed_list[@]}"; do
  for eval_seed in "${eval_seed_list[@]}"; do
    for subset in $(seq "$START" "$END"); do
      subset_padded="$(printf "%04d" "$subset")"
      echo "evaluating LDS subset=${subset_padded} seed=${seed} eval_seed=${eval_seed}"
      python3 -m torch_das.eval_loss \
        --model-dir "$MODEL_ROOT/subset_${subset_padded}/seed_${seed}" \
        --dataset "$DATASET" \
        --dataset-kind cifar2 \
        --dataset-type "$DATASET_TYPE" \
        --center-crop \
        --batch-size "$BATCH_SIZE" \
        --num-timesteps "$TIMESTEPS" \
        --max-samples "$MAX_SAMPLES" \
        --seed "$eval_seed" \
        --output "$LOSS_ROOT/subset_${subset_padded}/seed_${seed}/eval_seed_${eval_seed}/losses.pkl" \
        --device "$DEVICE"
    done
  done
done
