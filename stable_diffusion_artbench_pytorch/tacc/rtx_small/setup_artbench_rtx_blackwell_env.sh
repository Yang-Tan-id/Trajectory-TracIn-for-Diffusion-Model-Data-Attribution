#!/usr/bin/env bash
set -euo pipefail

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh

source_env="${SOURCE_ENV:-/scratch/11447/yangtan7447/conda-envs/sd_artbench}"
target_env="${TARGET_ENV:-/scratch/11447/yangtan7447/conda-envs/sd_artbench_rtx}"

[[ -x "$source_env/bin/python" ]] || {
  echo "ERROR: source environment is missing: $source_env" >&2
  exit 2
}

if [[ ! -x "$target_env/bin/python" ]]; then
  echo "[clone] $source_env -> $target_env"
  conda create -y -p "$target_env" --clone "$source_env"
else
  echo "[reuse] $target_env"
fi

python_bin="$target_env/bin/python"
echo "[install] PyTorch 2.7.0 + torchvision 0.22.0 + CUDA 12.8"
"$python_bin" -m pip install --no-cache-dir --upgrade \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

"$python_bin" - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("compiled CUDA:", torch.version.cuda)
if not torch.__version__.startswith("2.7.0"):
    raise RuntimeError(f"unexpected torch version: {torch.__version__}")
if torch.version.cuda != "12.8":
    raise RuntimeError(f"expected CUDA 12.8 build, found {torch.version.cuda}")
print("[environment ready] GPU execution is checked inside the RTX allocation")
PY
