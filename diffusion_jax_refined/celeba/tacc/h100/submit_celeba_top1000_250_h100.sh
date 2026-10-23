#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-$(cd "$script_dir/../../../.." && pwd)}"
celeba="$repo/diffusion_jax_refined/celeba"
plan="$celeba/script/celeba_top1000_250_plan.py"
runner="$celeba/tacc/h100/run_celeba_top1000_250_h100.sh"
query_file="$celeba/queries_attr40_val25_x4.json"
python_bin="${PYTHON_BIN:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin/bin/python}"

school_remote="${SCHOOL_REMOTE:-yt7447@ase-a69796}"
school_repo="${SCHOOL_REPO_ROOT:-/home/yt7447/diffusion_model/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution}"
launch_root="${SCRATCH:?SCRATCH is required}/celeba_top1000_250_launch"
mkdir -p "$launch_root"
manifest="$launch_root/transfer_manifest.txt"

[[ -x "$python_bin" ]] || { echo "ERROR: Python missing: $python_bin" >&2; exit 2; }
[[ -f "$plan" && -f "$runner" ]] || {
  echo "ERROR: required repository files are missing" >&2
  exit 2
}

query_relative="diffusion_jax_refined/celeba/queries_attr40_val25_x4.json"
if [[ "${SKIP_SCHOOL_SYNC:-0}" != 1 ]]; then
  echo "[bootstrap] $query_relative"
  mkdir -p "$(dirname "$query_file")"
  rsync -a --partial \
    "$school_remote:${school_repo%/}/$query_relative" \
    "$query_file"
fi
[[ -s "$query_file" ]] || {
  echo "ERROR: query definition is missing: $query_file" >&2
  exit 2
}

"$python_bin" "$plan" --mode transfer --query-file "$query_file" > "$manifest"
[[ "$(wc -l < "$manifest" | tr -d ' ')" == 896 ]] || {
  echo "ERROR: expected 896 required transfer files" >&2
  exit 2
}

echo "[transfer] source=$school_remote:$school_repo"
echo "[transfer] destination=$repo"
echo "[transfer] files=896 (query definitions, scores, score manifests, base checkpoint, dataset, true-f, trajectories)"
if [[ "${SKIP_SCHOOL_SYNC:-0}" != 1 ]]; then
  rsync -aR --partial --info=progress2 \
    --files-from="$manifest" \
    "$school_remote:${school_repo%/}/" \
    "$repo/"
fi

missing=0
while IFS= read -r relative; do
  if [[ ! -s "$repo/$relative" ]]; then
    echo "MISSING: $repo/$relative" >&2
    missing=$((missing + 1))
  fi
done < "$manifest"
[[ $missing -eq 0 ]] || {
  echo "ERROR: $missing staged inputs are missing" >&2
  exit 1
}
echo "[transfer complete] all 896 required files are present"

export REPO_ROOT="$repo"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-attr40_experiment1}"
export CONDA_ENV="${CONDA_ENV:-/scratch/11447/yangtan7447/conda-envs/trajectory-tracin}"

echo "[submit] $runner"
sbatch --chdir="$repo" "$runner"
