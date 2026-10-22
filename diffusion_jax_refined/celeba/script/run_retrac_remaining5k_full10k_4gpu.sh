#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
celeba_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
experiment="${EXPERIMENT_TAG:-attr40_experiment1}"
query_ids="${QUERY_IDS:-$(seq -s, 0 71)}"
gpu_ids="${GPU_IDS:-0,5,6,7}"
data_root="${CELEBA_ATTR40_DATA_ROOT:-$celeba_root/../dataset/celeba/10000_attr40}"
original_indices="${ATTRIBUTION_INDICES_PATH_ORIGINAL:-$data_root/attribution_5k_indices.npy}"
remaining_indices="${REMAINING_INDICES_PATH:-$data_root/attribution_remaining_5k_indices.npy}"
run_tag="${RUN_TAG:-celeba_retrac_remaining5k_q0_71_$(date +%Y%m%d_%H%M%S)}"

"$python_bin" - "$original_indices" "$remaining_indices" <<'PY'
import os
import sys
from pathlib import Path
import numpy as np

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
selected = np.asarray(np.load(source), dtype=np.int64).reshape(-1)
if len(selected) != 5000 or len(np.unique(selected)) != 5000:
    raise ValueError(f"expected 5000 unique original indices: {source}")
if np.any(selected < 0) or np.any(selected >= 10000):
    raise ValueError("original attribution index outside [0, 10000)")
remaining = np.setdiff1d(np.arange(10000, dtype=np.int64), selected)
if len(remaining) != 5000:
    raise ValueError(f"expected 5000 remaining indices, found {len(remaining)}")
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npy")
np.save(temporary, remaining)
temporary.replace(destination)
print(f"[indices] original={source} remaining={destination} count={len(remaining)}")
PY

ATTRIBUTION_INDICES_PATH="$remaining_indices" \
SCORE_OUTPUT_SUFFIX="_remaining5000" \
REPLAY_DROPOUT=0 \
PYTHON_BIN="$python_bin" \
GPU_IDS="$gpu_ids" \
QUERY_IDS="$query_ids" \
EXPERIMENT_TAG="$experiment" \
RUN_TAG="$run_tag" \
bash "$script_dir/run_retrac_1x50_4gpu.sh"

JAX_PLATFORMS=cpu "$python_bin" "$script_dir/merge_retrac_full10k.py" \
  --query-file "$celeba_root/queries_attr40_val25_x4.json" \
  --query-ids "$query_ids" \
  --experiment "$experiment" \
  --replay-tag "train_replay4_batch1" \
  --supplement-suffix "_remaining5000" \
  --full-suffix "_full10000"

echo "[done] Q0-Q71 ReTrac raw-gradient and AdamW-full full-10000 banks ready"
