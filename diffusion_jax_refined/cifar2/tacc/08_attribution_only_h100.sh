#!/usr/bin/env bash
#SBATCH --job-name=cifar2-attr-h100
#SBATCH --partition=h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=cifar2-attr-h100-%j.out
#SBATCH --error=cifar2-attr-h100-%j.err

set -euo pipefail

# Attribution-only H100 runner. Reuses existing samples; does not run sampling
# and does not launch LDS eval. With ATTR_ALGORITHMS="traj_tracin das",
# defaults run tasks 1..16 on 16 H100 GPUs.

ATTR_TASK_START="${ATTR_TASK_START:-1}"
ATTR_TASK_END="${ATTR_TASK_END:-16}"
MAX_PARALLEL_ATTR_TASKS="${MAX_PARALLEL_ATTR_TASKS:-${SLURM_NTASKS:-16}}"
GPU_PER_NODE="${GPU_PER_NODE:-4}"
ATTR_ONLY_LABEL="${ATTR_ONLY_LABEL:-h100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=attribution_only_common.sh
source "${SCRIPT_DIR}/attribution_only_common.sh"
