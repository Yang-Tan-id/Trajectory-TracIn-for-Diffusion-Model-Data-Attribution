#!/usr/bin/env bash
#SBATCH --job-name=cifar2-traj-n2-rtx
#SBATCH --partition=rtx-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --time=20:00:00
#SBATCH --output=cifar2-traj-n2-rtx-%j.out
#SBATCH --error=cifar2-traj-n2-rtx-%j.err

set -euo pipefail

# rtx-small fallback for the H100 norm^2 traj_tracin + k=8000 LDS workflow.
# This executes the same workflow as 10_traj_norm2_k8000_h100_3node.sh inside
# a smaller allocation, so it is slower but easier to backfill.

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="${REPO_ROOT}/diffusion_jax_refined/cifar2/tacc"

export MAX_PARALLEL_ATTR_TASKS="${MAX_PARALLEL_ATTR_TASKS:-${SLURM_NTASKS:-2}}"
export MAX_PARALLEL_EVAL_TASKS="${MAX_PARALLEL_EVAL_TASKS:-${SLURM_NTASKS:-2}}"
export GPU_PER_NODE="${GPU_PER_NODE:-2}"

exec bash "${SCRIPT_DIR}/10_traj_norm2_k8000_h100_3node.sh"
