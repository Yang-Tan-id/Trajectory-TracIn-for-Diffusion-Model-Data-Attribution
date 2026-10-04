#!/usr/bin/env bash
#SBATCH -J 3d-paper-ret-both
#SBATCH -o 3d-paper-ret-both-%j.out
#SBATCH -e 3d-paper-ret-both-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/tacc/rtx_small/run_paper_retrac_endpoint100x1_q0_99_scores_lds_rtx_small.sh" ]]; do
  repo="$(dirname "$repo")"
done
worker="$repo/diffusion_jax_refined/3dshapes/tacc/rtx_small/run_paper_retrac_endpoint100x1_q0_99_scores_lds_rtx_small.sh"
[[ -f "$worker" ]] || { echo "Unable to locate score worker from $repo" >&2; exit 1; }
artifact="paper_retrac_raw_adamw_full_l2_normalized_four_events_n5000"

echo "[1/2] paper-normalized raw-loss ReTrac"
PAPER_ARTIFACT_NAME="$artifact" \
PAPER_TRAIN_TRANSFORM=raw \
PAPER_NAMESPACE=paper_retrac_exact4_endpoint100x1_q0_99 \
PAPER_SCORE_SCHEME=paper_retrac_exact4_endpoint100x1_q0_99 \
bash "$worker"

echo "[2/2] paper-normalized AdamW-full ReTrac"
PAPER_ARTIFACT_NAME="$artifact" \
PAPER_TRAIN_TRANSFORM=adamw_full \
PAPER_NAMESPACE=paper_retrac_adamw_full_exact4_endpoint100x1_q0_99 \
PAPER_SCORE_SCHEME=paper_retrac_adamw_full_exact4_endpoint100x1_q0_99 \
bash "$worker"

echo "[done] raw and AdamW-full paper-normalized ReTrac scores/LDS"
