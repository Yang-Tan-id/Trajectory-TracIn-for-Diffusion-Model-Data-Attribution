#!/usr/bin/env bash
#SBATCH -J 3d-pn4-own-ts
#SBATCH -o 3d-pn4-own-ts-%j.out
#SBATCH -e 3d-pn4-own-ts-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAJECTORY_MODE=own
exec bash "${SCRIPT_DIR}/run_reference_timestamp_shared_probe4_full_rtx_small.sh"
