#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-$(cd "$script_dir/../../../.." && pwd)}"
batch_script="$repo/diffusion_jax_refined/3dshapes/tacc/spr/run_package_endpoint_q0_99_scores_lds_spr.sh"

stamp="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
launch_dir="${SCRATCH}/shape3d_q0_99_export_launch"
output_dir="${OUTPUT_DIR:-${SCRATCH}/shape3d_q0_99_endpoint_export_skxdev_${stamp}}"
submit_log="$launch_dir/submit_skxdev_${stamp}.log"

mkdir -p "$launch_dir" "$output_dir"

# Stampede3's submit filter does not support an explicit sbatch export option.
# Export through the calling environment; sbatch inherits it by default.
export REPO_ROOT="$repo"
export OUTPUT_DIR="$output_dir"

sbatch \
  -p skx-dev \
  -A "${ALLOCATION:-IRI26004}" \
  -N 1 \
  -n 1 \
  --cpus-per-task="${PACKAGE_CPUS:-4}" \
  -t "${PACKAGE_TIME_LIMIT:-02:00:00}" \
  --chdir="$SCRATCH" \
  -J pack-3d-q100-dev \
  -o "$launch_dir/package-%j.out" \
  -e "$launch_dir/package-%j.err" \
  "$batch_script" | tee "$submit_log"

job="$(
  awk '/Submitted batch job [0-9]+/{print $4}' "$submit_log" |
    tail -n 1
)"
[[ -n "$job" ]] || { echo "unable to parse submitted job id" >&2; exit 1; }

echo "JOB=$job"
echo "OUTPUT_DIR=$output_dir"
echo "STDOUT=$launch_dir/package-${job}.out"
echo "STDERR=$launch_dir/package-${job}.err"
