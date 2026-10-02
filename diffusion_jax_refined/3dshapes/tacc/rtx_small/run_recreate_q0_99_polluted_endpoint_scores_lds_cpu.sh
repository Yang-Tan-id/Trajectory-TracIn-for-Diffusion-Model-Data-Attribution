#!/usr/bin/env bash
#SBATCH -J 3d-q0-99-pscore
#SBATCH -o 3d-q0-99-pscore-%j.out
#SBATCH -e 3d-q0-99-pscore-%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/tacc/rtx_small/run_recreate_q100_polluted_endpoint_scores_lds_cpu.sh" ]]; do
  repo="$(dirname "$repo")"
done

export QUERY_IDS="${QUERY_IDS:-$(seq -s, 0 99)}"
export SCORE_SCOPE=q0_99
export SCORE_TRAIN_KINDS="${SCORE_TRAIN_KINDS:-raw,adamw_residual,adamw_full}"

exec bash \
  "$repo/diffusion_jax_refined/3dshapes/tacc/rtx_small/run_recreate_q100_polluted_endpoint_scores_lds_cpu.sh"
