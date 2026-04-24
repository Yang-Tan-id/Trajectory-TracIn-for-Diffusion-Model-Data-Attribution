#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher for DM__training_CIFAR10_pixel.py.
# Uses TrainConfig exactly as defined in that file's __main__ block.
#
# Usage:
#   bash train_cifar10_unet.sh
#
# Optional:
#   CONDA_ENV=attribution bash train_cifar10_unet.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash train_cifar10_unet.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_ENV="${CONDA_ENV:-attribution}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Launching training from: ${SCRIPT_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "Using TrainConfig from DM__training_CIFAR10_pixel.py"

conda run -n "${CONDA_ENV}" --no-capture-output "${PYTHON_BIN}" DM__training_CIFAR10_pixel.py

