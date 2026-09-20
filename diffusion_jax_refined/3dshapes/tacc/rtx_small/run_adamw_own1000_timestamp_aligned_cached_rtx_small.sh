#!/usr/bin/env bash
#SBATCH -J 3d-a4-align
#SBATCH -o 3d-a4-align-%j.out
#SBATCH -e 3d-a4-align-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

set -euo pipefail

repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [[ "$repo" != / && ! -f "$repo/diffusion_jax_refined/3dshapes/script/run_adamw_own1000_timestamp_aligned_scores.py" ]]; do
  repo="$(dirname "$repo")"
done
shapes="$repo/diffusion_jax_refined/3dshapes"
driver="$shapes/script/run_adamw_own1000_timestamp_aligned_scores.py"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

experiment="${EXPERIMENT_TAG:-experiment1}"
train_seed="${TRAIN_SEED:-42}"
out_dir="$shapes/result/$experiment/eval/adamw_own1000_exact_timestamp_alignment/run_${SLURM_JOB_ID}"
log_dir="$shapes/result/$experiment/logs/adamw_own1000_exact_timestamp_alignment/${SLURM_JOB_ID}"
mkdir -p "$out_dir" "$log_dir"

echo "[definition] each datapoint/event uses only its exact saved training timestep"
echo "[query] cached own-trajectory raw next-checkpoint bank with all 1000 timestamps"
echo "[train] four saved AdamW event features per checkpoint/datapoint"
echo "[combine] E1-E4 remain separate; FOUR sums the four already-aligned event scores"
echo "[weighting] uniform checkpoints; fixed p1"

pids=()
for shard in 0 1; do
  (
    CUDA_VISIBLE_DEVICES="$shard" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda \
      python "$driver" score-shard \
        --experiment "$experiment" --train-seed "$train_seed" \
        --shard-index "$shard" --shard-count 2 --out-dir "$out_dir"
  ) >"$log_dir/score_gpu_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "aligned scoring failed; inspect $log_dir" >&2; exit 1; }

JAX_PLATFORMS=cpu python "$driver" merge \
  --experiment "$experiment" --train-seed "$train_seed" \
  --shard-count 2 --out-dir "$out_dir"

echo "[done] exact own-trajectory event-timestamp alignment: $out_dir"
