#!/usr/bin/env bash
set -euo pipefail

python3 -m torch_das.score \
  --train-grads "${TRAIN_GRADS:-runs/cifar2/train_grads.npy}" \
  --query-grads "${QUERY_GRADS:-runs/cifar2/query_grads.npy}" \
  --train-shape "${TRAIN_SHAPE:-10000,4096}" \
  --query-shape "${QUERY_SHAPE:-1000,4096}" \
  --output "${OUTPUT:-runs/cifar2/lds/query_train_scores.npy}" \
  --method "${METHOD:-ridge}" \
  --ridge "${RIDGE:-0.01}" \
  --query-reduction none
