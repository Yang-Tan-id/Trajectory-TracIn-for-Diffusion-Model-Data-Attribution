#!/usr/bin/env bash
set -euo pipefail

python3 -m torch_das.das_score_sweep \
  --train-grads "${TRAIN_GRADS:-runs/cifar2/train_grads.npy}" \
  --query-grads "${QUERY_GRADS:-runs/cifar2/query_grads.npy}" \
  --error-train "${ERROR_TRAIN:-runs/cifar2/error/error_train.npy}" \
  --train-shape "${TRAIN_SHAPE:-10000,4096}" \
  --query-shape "${QUERY_SHAPE:-1000,4096}" \
  --subset-index-dir "${INDEX_DIR:-runs/cifar2/lds/indices}" \
  --train-index "${TRAIN_INDEX:-runs/cifar2/lds/indices/idx-train.pkl}" \
  --loss-root "${LOSS_ROOT:-runs/cifar2/lds/losses}" \
  --output-dir "${OUTPUT_DIR:-runs/cifar2/lds/original_das}" \
  --num-subsets "${NUM_SUBSETS:-64}" \
  --seeds "${SEEDS:-0,1,2}" \
  --eval-seeds "${EVAL_SEEDS:-0}" \
  --method "${METHOD:-das1}" \
  --device "${DEVICE:-cuda}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  ${NORMALIZE_INV:+--normalize-inv} \
  ${SAVE_BEST_SCORES:+--save-best-scores}
