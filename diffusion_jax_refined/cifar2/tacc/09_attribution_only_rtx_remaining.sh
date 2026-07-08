#!/usr/bin/env bash
#SBATCH --job-name=cifar2-attr-rtx
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-attr-rtx-%j.out
#SBATCH --error=cifar2-attr-rtx-%j.err

set -euo pipefail

# Attribution-only RTX-small remaining-task runner. Reuses existing samples;
# does not run sampling and does not launch LDS eval. With
# ATTR_ALGORITHMS="traj_tracin das", defaults run tasks 17..18 on one
# rtx-small node with two GPUs.

ATTR_TASK_START="${ATTR_TASK_START:-17}"
ATTR_TASK_END="${ATTR_TASK_END:-18}"
MAX_PARALLEL_ATTR_TASKS="${MAX_PARALLEL_ATTR_TASKS:-${SLURM_NTASKS:-2}}"
GPU_PER_NODE="${GPU_PER_NODE:-2}"
ATTR_ONLY_LABEL="${ATTR_ONLY_LABEL:-rtx}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=attribution_only_common.sh
source "${SCRIPT_DIR}/attribution_only_common.sh"
