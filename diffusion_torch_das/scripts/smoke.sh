#!/usr/bin/env bash
set -euo pipefail

python3 -m torch_das.train \
  --dataset synthetic \
  --config configs/tiny_unet.json \
  --output-dir runs/smoke/ddpm \
  --synthetic-samples 8 \
  --batch-size 4 \
  --num-epochs 1 \
  --num-train-timesteps 20 \
  --device auto

python3 -m torch_das.generate \
  --model-dir runs/smoke/ddpm \
  --output-dir runs/smoke/gen \
  --num-images 4 \
  --batch-size 2 \
  --num-inference-steps 5 \
  --device auto

python3 -m torch_das.eval_loss \
  --model-dir runs/smoke/ddpm \
  --dataset synthetic \
  --synthetic-samples 8 \
  --max-samples 4 \
  --num-timesteps 4 \
  --output runs/smoke/losses.pkl \
  --device auto

python3 -m torch_das.gradients \
  --model-dir runs/smoke/ddpm \
  --dataset synthetic \
  --synthetic-samples 8 \
  --max-samples 4 \
  --num-timesteps 2 \
  --projection-dim 32 \
  --output runs/smoke/train_grads.npy \
  --device auto

python3 -m torch_das.gradients \
  --model-dir runs/smoke/ddpm \
  --dataset runs/smoke/gen \
  --dataset-type gen \
  --max-samples 4 \
  --num-timesteps 2 \
  --projection-dim 32 \
  --output runs/smoke/query_grads.npy \
  --device auto

python3 -m torch_das.score \
  --train-grads runs/smoke/train_grads.npy \
  --query-grads runs/smoke/query_grads.npy \
  --train-shape 4,32 \
  --query-shape 4,32 \
  --output runs/smoke/scores.npy
