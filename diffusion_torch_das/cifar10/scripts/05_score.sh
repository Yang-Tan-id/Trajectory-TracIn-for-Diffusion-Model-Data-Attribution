#!/usr/bin/env bash
set -euo pipefail

python3 -m torch_das.score \
  --train-grads "${TRAIN_GRADS:-runs/cifar10/train_grads.npy}" \
  --query-grads "${QUERY_GRADS:-runs/cifar10/query_grads.npy}" \
  --train-shape "${TRAIN_SHAPE:-50000,4096}" \
  --query-shape "${QUERY_SHAPE:-1000,4096}" \
  --output "${OUTPUT:-runs/cifar10/scores.npy}" \
  --method "${METHOD:-ridge}"
