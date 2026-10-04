#!/usr/bin/env bash
#SBATCH -J 3d-paper-ret-sc
#SBATCH -o 3d-paper-ret-sc-%j.out
#SBATCH -e 3d-paper-ret-sc-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/score_retrac_endpoint100x1_checkpoint_major.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin

export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

experiment="${EXPERIMENT_TAG:-experiment1}"
seed="${TRAIN_SEED:-42}"
query_file="$shapes/queries_in_distribution_plus_zero_seed_100_219.json"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
artifact_name="${PAPER_ARTIFACT_NAME:-paper_retrac_full_l2_normalized_four_events_n5000}"
train_transform="${PAPER_TRAIN_TRANSFORM:-raw}"
namespace="${PAPER_NAMESPACE:-paper_retrac_exact4_endpoint100x1_q0_99}"
score_scheme="${PAPER_SCORE_SCHEME:-paper_retrac_exact4_endpoint100x1_q0_99}"
events="$shapes/result/$experiment/$artifact_name"
logs="$shapes/result/$experiment/logs/paper_retrac_endpoint100x1_scores/${SLURM_JOB_ID}"
mkdir -p "$logs"

count="$(find "$events" -type f -name 'event_gradient_epoch_*_shard_*_of_02.npz' | wc -l | tr -d ' ')"
[[ "$count" == 392 ]] || { echo "expected 392 event shards, found $count" >&2; exit 1; }

echo "[definition] official ReTrac normalization order; 100 timestamps x MC1"
echo "[train] normalize each full event gradient before projection"
echo "[query] normalize each full timestep gradient before projection, then average 100"

pids=()
for gpu in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda JAX_NUM_DEVICES=1 \
      python "$shapes/script/score_retrac_endpoint100x1_checkpoint_major.py" \
        --query-file "$query_file" --query-ids "$query_ids" \
        --experiment "$experiment" --train-seed "$seed" \
        --retrac-event-root "$events" --methods retrac --paper-retrac \
        --paper-train-transform "$train_transform" \
        --retrac-namespace "$namespace" \
        --retrac-reduction linear --query-batch-size "${QUERY_BATCH_SIZE:-1}" \
        --shard-index "$gpu" --shard-count 2
  ) >"$logs/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "score worker failed; inspect $logs" >&2; exit 1; }

JAX_PLATFORMS=cpu python "$shapes/script/run_traj_tracin_lds_cached.py" \
  --execute --experiment "$experiment" --train-seed "$seed" \
  --query-file "$query_file" --query-ids "$query_ids" \
  --score-schemes "$score_scheme" \
  --prediction-sign -1

echo "[done] paper-normalized ReTrac Q0-Q99 scores and LDS"
