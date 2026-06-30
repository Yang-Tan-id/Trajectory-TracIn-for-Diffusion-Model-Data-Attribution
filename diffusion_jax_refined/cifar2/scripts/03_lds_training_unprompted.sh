#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${ROOT}/lds/run_training.py" \
  --unprompted \
  --m "${LDS_M:-100}" \
  --k "${LDS_K:-${LDS_SUBSET_SIZE:-5000}}" \
  --sample-random-seed "${LDS_SAMPLE_RANDOM_SEED:-${LDS_SUBSET_SEED:-0}}" \
  "$@"
