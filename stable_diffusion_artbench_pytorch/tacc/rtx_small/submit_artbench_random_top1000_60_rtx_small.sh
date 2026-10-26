#!/usr/bin/env bash
# Run this on a Stampede3 login node after the ArtBench H100 staging completed.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-$(cd "$script_dir/../../.." && pwd)}"
artbench="$repo/stable_diffusion_artbench_pytorch"
runner="$artbench/tacc/rtx_small/run_artbench_random_top1000_60_rtx_small.sh"
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/sd_artbench/bin/python}"
hf_home="${HF_HOME:-/scratch/11447/yangtan7447/huggingface}"

[[ -x "$python_bin" ]] || { echo "ERROR: Python missing: $python_bin" >&2; exit 2; }
[[ -f "$runner" ]] || { echo "ERROR: runner missing: $runner" >&2; exit 2; }
[[ -s "$artbench/dataset/indices/idx-train.pkl" ]] || {
  echo "ERROR: staged ArtBench training split is missing" >&2; exit 2;
}
[[ -d "$hf_home/hub/models--lambdalabs--miniSD-diffusers/snapshots" ]] || {
  echo "ERROR: staged miniSD cache is missing under $hf_home" >&2; exit 2;
}

export PYTHONPATH="$artbench/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[unit test] ArtBench random Q0-Q59 RTX contract"
"$python_bin" "$artbench/tests/test_artbench_random_top1000_60_rtx_small.py"
"$python_bin" -m sd_artbench.artbench_random_top1000_60_plan \
  --mode validate --top-k 1000 --random-seed 2026

export REPO_ROOT="$repo"
export CONDA_ENV="${CONDA_ENV:-/scratch/11447/yangtan7447/conda-envs/sd_artbench}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$artbench/outputs/topk-removal-adamw-q10-59-random-q0-59}"

echo "[submit] three independent 20-query shards; no school-server connection"
for shard_index in 0 1 2; do
  sbatch --chdir="$repo" "$runner" "$shard_index" 3
done
