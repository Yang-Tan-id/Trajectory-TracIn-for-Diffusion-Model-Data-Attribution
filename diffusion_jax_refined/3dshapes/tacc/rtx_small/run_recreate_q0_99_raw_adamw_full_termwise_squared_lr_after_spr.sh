#!/usr/bin/env bash
#SBATCH -J 3d-poll-t2lr
#SBATCH -o 3d-poll-t2lr-%j.out
#SBATCH -e 3d-poll-t2lr-%j.err
#SBATCH -p spr
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH -t 12:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
worker_rel="diffusion_jax_refined/3dshapes/tacc/rtx_small/run_recreate_q0_99_polluted_endpoint_scores_lds_cpu.sh"
while [[ "$repo" != / && ! -f "$repo/$worker_rel" ]]; do
  repo="$(dirname "$repo")"
done
worker="$repo/$worker_rel"
[[ -f "$worker" ]] || { echo "Unable to locate $worker_rel" >&2; exit 1; }

export SCORE_TRAIN_KINDS="raw,adamw_full"
export SCORE_DELTA_KINDS="delta_l2normalized"
export SCORE_REDUCTIONS="termwise_squared_lr_after"

echo "[definition] sum_(checkpoint,timestamp) LR_weight * dot(train,query)^2"
echo "[scope] raw parameter gradient + AdamW full; normalized delta; Q0-Q99"

exec bash "$worker"
