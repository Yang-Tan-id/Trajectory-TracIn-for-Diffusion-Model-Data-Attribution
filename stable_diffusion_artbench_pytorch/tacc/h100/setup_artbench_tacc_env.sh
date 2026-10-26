#!/usr/bin/env bash
# One-time Stampede3 environment setup. Run on a login node before submission.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
artbench="$(cd "$script_dir/../.." && pwd)"
env_path="${CONDA_ENV:-/scratch/11447/yangtan7447/conda-envs/sd_artbench}"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
if [[ ! -x "$env_path/bin/python" ]]; then
  conda create -y -p "$env_path" python=3.10 pip
fi

cd "$artbench"
"$env_path/bin/python" -m pip install -r requirements.txt
"$env_path/bin/python" -m pip install -e .
"$env_path/bin/python" - <<'PY'
import accelerate
import diffusers
import numpy
import torch
import transformers

print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
PY
echo "[done] TACC ArtBench environment: $env_path"
