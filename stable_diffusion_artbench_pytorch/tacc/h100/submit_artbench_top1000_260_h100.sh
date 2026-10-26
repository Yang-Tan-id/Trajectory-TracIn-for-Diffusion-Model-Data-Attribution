#!/usr/bin/env bash
# Run this script on a Stampede3 login node after the school-side push.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-$(cd "$script_dir/../../.." && pwd)}"
artbench="$repo/stable_diffusion_artbench_pytorch"
runner="$artbench/tacc/h100/run_artbench_top1000_260_h100.sh"
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/sd_artbench/bin/python}"
hf_home="${HF_HOME:-/scratch/11447/yangtan7447/huggingface}"
manifest="${SCRATCH:?SCRATCH is required}/artbench_top1000_260_transfer_manifest.txt"

[[ -x "$python_bin" ]] || { echo "ERROR: Python missing: $python_bin" >&2; exit 2; }
[[ -f "$runner" ]] || { echo "ERROR: runner missing: $runner" >&2; exit 2; }
export PYTHONPATH="$artbench/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$python_bin" -m sd_artbench.artbench_top1000_260_plan --mode transfer > "$manifest"
[[ "$(wc -l < "$manifest" | tr -d ' ')" == 5005 ]] || {
  echo "ERROR: expected 5005 required project files" >&2; exit 2;
}
missing=0
while IFS= read -r relative; do
  if [[ ! -s "$repo/$relative" ]]; then
    echo "MISSING: $repo/$relative" >&2
    missing=$((missing + 1))
  fi
done < "$manifest"
[[ $missing -eq 0 ]] || { echo "ERROR: $missing staged files missing" >&2; exit 1; }

model_cache="$hf_home/hub/models--lambdalabs--miniSD-diffusers"
[[ -d "$model_cache/snapshots" ]] || {
  echo "ERROR: staged miniSD cache missing: $model_cache" >&2; exit 1;
}
"$python_bin" -m sd_artbench.artbench_top1000_260_plan \
  --mode validate --top-k 1000 --random-seed 2026

export REPO_ROOT="$repo"
export CONDA_ENV="${CONDA_ENV:-/scratch/11447/yangtan7447/conda-envs/sd_artbench}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$artbench/outputs/topk-removal-adamw-q10-59-random-q0-59}"

echo "[submit] staged inputs verified; no school-server connection is attempted"
sbatch --chdir="$repo" "$runner"
