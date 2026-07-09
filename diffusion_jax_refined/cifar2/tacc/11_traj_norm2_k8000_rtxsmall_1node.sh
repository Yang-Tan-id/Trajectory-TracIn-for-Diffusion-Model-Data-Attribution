#!/usr/bin/env bash
#SBATCH --job-name=cifar2-traj-n2-rtx
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=cifar2-traj-n2-rtx-%j.out
#SBATCH --error=cifar2-traj-n2-rtx-%j.err

set -euo pipefail

# rtx-small fallback for the H100 norm^2 traj_tracin + k=8000 LDS workflow.
# This executes the same workflow as 10_traj_norm2_k8000_h100_3node.sh inside
# a smaller allocation, so it is slower but easier to backfill.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MAX_PARALLEL_ATTR_TASKS="${MAX_PARALLEL_ATTR_TASKS:-${SLURM_NTASKS:-4}}"
export MAX_PARALLEL_EVAL_TASKS="${MAX_PARALLEL_EVAL_TASKS:-${SLURM_NTASKS:-4}}"
export GPU_PER_NODE="${GPU_PER_NODE:-4}"

exec bash "${SCRIPT_DIR}/10_traj_norm2_k8000_h100_3node.sh"
