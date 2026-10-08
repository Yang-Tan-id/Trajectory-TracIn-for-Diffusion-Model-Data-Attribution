#!/usr/bin/env bash
#SBATCH -J 3d-end-a10-q100
#SBATCH -o 3d-end-a10-q100-%j.out
#SBATCH -e 3d-end-a10-q100-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/score_retrac_endpoint100x1_checkpoint_major.py" ]]; do repo="$(dirname "$repo")"; done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
ids="$(seq -s, 0 99)"
artifact="$shapes/result/experiment1/model/prompted_solo/seed_42_train_gradient/traj_tracin_adamw_dual_aligned10x10/train_datapoint_gradient_artifact.npz"
reduction="${ENDPOINT_REDUCTION:-linear}"
case "$reduction" in
  linear)
    namespace="endpoint_tracin_adamw_full_train10x10_query100x1_q0_99"
    ;;
  termwise_squared_lr_after)
    namespace="endpoint_tracin_adamw_full_train10x10_query100x1_termwise_squared_lr_after_q0_99"
    ;;
  *)
    echo "Unsupported ENDPOINT_REDUCTION=$reduction" >&2
    exit 2
    ;;
esac
logs="$shapes/result/experiment1/logs/endpoint_adamw10x10_query100x1_${reduction}/${SLURM_JOB_ID}"
mkdir -p "$logs"

pids=()
for gpu in 0 1; do
  (CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 python "$shapes/script/score_retrac_endpoint100x1_checkpoint_major.py" \
    --query-file "$query_file" --query-ids "$ids" --experiment experiment1 --train-seed 42 \
    --retrac-event-root "$shapes/result/experiment1/fixed_checkpoint_raw_four_events_n5000" \
    --methods endpoint --endpoint-train-artifact "$artifact" \
    --endpoint-adamw-aligned10x10 --endpoint-adamw-full --endpoint-namespace "$namespace" \
    --endpoint-reduction "$reduction" \
    --query-batch-size 2 --shard-index "$gpu" --shard-count 2) >"$logs/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "score worker failed; inspect $logs" >&2; exit 1; }

JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" --execute \
  --experiment experiment1 --train-seed 42 --query-file "$query_file" --query-ids "$ids" \
  --score-schemes "$namespace" --prediction-sign -1
echo "[done] AdamW-full train10x10 x endpoint query100x1 $reduction Q0-Q99"
