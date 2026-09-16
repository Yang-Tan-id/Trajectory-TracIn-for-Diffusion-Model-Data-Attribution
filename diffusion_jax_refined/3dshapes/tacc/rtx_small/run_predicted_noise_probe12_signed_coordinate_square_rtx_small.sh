#!/usr/bin/env bash
#SBATCH -J 3d-pn12-sqplay
#SBATCH -o 3d-pn12-sqplay-%j.out
#SBATCH -e 3d-pn12-sqplay-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUM_PROBES=12
source "${SCRIPT_DIR}/run_predicted_noise_probe8_signed_coordinate_square_rtx_small.sh"
