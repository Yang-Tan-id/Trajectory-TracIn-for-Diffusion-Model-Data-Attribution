#!/usr/bin/env bash
# Run this script on the school server.  It pushes data to TACC; TACC never
# attempts to resolve or connect back to the school host.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-$(cd "$script_dir/../../.." && pwd)}"
artbench="$repo/stable_diffusion_artbench_pytorch"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/sd_artbench/bin/python}"
tacc="${TACC_REMOTE:-yangtan7447@stampede3.tacc.utexas.edu}"
tacc_repo="${TACC_REPO_ROOT:-/work2/11447/yangtan7447/stampede3/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution}"
school_hf_home="${SCHOOL_HF_HOME:-$HOME/.cache/huggingface}"
tacc_hf_home="${TACC_HF_HOME:-/scratch/11447/yangtan7447/huggingface}"
model_cache="models--lambdalabs--miniSD-diffusers"
manifest="$repo/artbench_top1000_260_transfer_manifest.txt"

[[ -x "$python_bin" ]] || { echo "ERROR: Python missing: $python_bin" >&2; exit 2; }
export PYTHONPATH="$artbench/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" -m sd_artbench.artbench_top1000_260_plan --mode transfer > "$manifest"
[[ "$(wc -l < "$manifest" | tr -d ' ')" == 5005 ]] || {
  echo "ERROR: expected 5005 staged project files" >&2; exit 2;
}
"$python_bin" -m sd_artbench.artbench_top1000_260_plan \
  --mode validate --top-k 1000 --random-seed 2026

[[ -d "$school_hf_home/hub/$model_cache" ]] || {
  echo "ERROR: miniSD Hugging Face snapshot missing: $school_hf_home/hub/$model_cache" >&2
  exit 2
}

echo "[push] create TACC destinations"
ssh "$tacc" "mkdir -p '$tacc_repo' '$tacc_hf_home/hub/$model_cache'"

echo "[push] 5005 exact files: 5000 train images, metadata/index, three score banks"
rsync -aR --partial --info=progress2 \
  --files-from="$manifest" "$repo/" "$tacc:$tacc_repo/"

echo "[push] complete miniSD Hugging Face snapshot"
rsync -a --partial --info=progress2 \
  "$school_hf_home/hub/$model_cache/" \
  "$tacc:$tacc_hf_home/hub/$model_cache/"

echo "[done] school -> TACC staging complete"
echo "[next on TACC] bash stable_diffusion_artbench_pytorch/tacc/h100/submit_artbench_top1000_260_h100.sh"
