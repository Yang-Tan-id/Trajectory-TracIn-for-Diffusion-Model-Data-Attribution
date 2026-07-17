#!/usr/bin/env bash
set -euo pipefail

python3 -m torch_das.score \
  --train-grads "${TRAIN_GRADS:-runs/cifar2/train_grads.npy}" \
  --query-grads "${QUERY_GRADS:-runs/cifar2/query_grads.npy}" \
  --train-shape "${TRAIN_SHAPE:-10000,4096}" \
  --query-shape "${QUERY_SHAPE:-1000,4096}" \
  --error-train "${ERROR_TRAIN:-runs/cifar2/error/error_train.npy}" \
  --output "${OUTPUT:-runs/cifar2/das1_scores.npy}" \
  --method das1 \
  --ridge "${RIDGE:-0.01}"
