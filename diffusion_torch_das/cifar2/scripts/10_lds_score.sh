#!/usr/bin/env bash
set -euo pipefail

python3 -m torch_das.lds_score \
  --scores "${SCORES:-runs/cifar2/lds/query_train_scores.npy}" \
  --subset-index-dir "${INDEX_DIR:-runs/cifar2/lds/indices}" \
  --train-index "${TRAIN_INDEX:-runs/cifar2/lds/indices/idx-train.pkl}" \
  --loss-root "${LOSS_ROOT:-runs/cifar2/lds/losses}" \
  --output "${OUTPUT:-runs/cifar2/lds/lds_results.csv}" \
  --num-subsets "${NUM_SUBSETS:-64}" \
  --seeds "${SEEDS:-0,1,2}" \
  --eval-seeds "${EVAL_SEEDS:-0}"
