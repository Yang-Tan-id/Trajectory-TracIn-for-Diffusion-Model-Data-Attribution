#!/usr/bin/env bash
#SBATCH -J art-rnd-t1k
#SBATCH -o art-rnd-t1k-%j.out
#SBATCH -e art-rnd-t1k-%j.err
#SBATCH -p rtx-small
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00

set -euo pipefail

shard_index="${1:?shard index is required}"
shard_count="${2:?shard count is required}"
[[ "$shard_index" =~ ^[0-9]+$ && "$shard_count" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: invalid shard arguments: $shard_index/$shard_count" >&2; exit 2;
}
(( shard_index < shard_count )) || {
  echo "ERROR: shard index must be smaller than shard count" >&2; exit 2;
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
marker="stable_diffusion_artbench_pytorch/src/sd_artbench/artbench_random_top1000_60_plan.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do repo="$(dirname "$repo")"; done
[[ -f "$repo/$marker" ]] || { echo "ERROR: repository not found" >&2; exit 2; }

artbench="$repo/stable_diffusion_artbench_pytorch"
plan_module="sd_artbench.artbench_random_top1000_60_plan"
slot_worker="$artbench/tacc/rtx_small/run_artbench_random_top1000_60_slot.sh"
output_root="${OUTPUT_ROOT:-$artbench/outputs/topk-removal-adamw-q10-59-random-q0-59}"

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/scratch/11447/yangtan7447/conda-envs/sd_artbench}"

export REPO_ROOT="$repo"
export OUTPUT_ROOT="$output_root"
export PYTHONPATH="$artbench/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/11447/yangtan7447/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

log_root="$output_root/logs/random_rtx_${SLURM_JOB_ID}_shard_${shard_index}_of_${shard_count}"
mkdir -p "$log_root"
task_file="$log_root/tasks.tsv"
python -m "$plan_module" --mode tasks \
  --shard-index "$shard_index" --shard-count "$shard_count" > "$task_file"
expected=$(( (60 + shard_count - 1 - shard_index) / shard_count ))
actual="$(wc -l < "$task_file" | tr -d ' ')"
[[ "$actual" == "$expected" ]] || {
  echo "ERROR: expected $expected tasks, found $actual" >&2; exit 2;
}

echo "[definition] ArtBench random top-1000 removal; full 5000-image training split"
echo "[definition] deterministic seed=SeedSequence([2026, query]); Q0-Q59"
echo "[definition] shard=$shard_index/$shard_count tasks=$actual; training seed=42"
echo "[parallel] one rtx-small node x 2 GPUs = 2 static resumable slots"
echo "[output] $output_root"
echo "[logs] $log_root"

echo "[preflight 1/3] random-selection and launcher unit tests"
python "$artbench/tests/test_artbench_random_top1000_60_rtx_small.py"

echo "[preflight 2/3] exact dataset and random-selection contracts"
python -m "$plan_module" --mode validate --top-k 1000 --random-seed 2026

echo "[preflight 3/3] allocated RTX GPU and offline miniSD snapshot"
ibrun -n 1 -o 0 env CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

name = "lambdalabs/miniSD-diffusers"
if not torch.cuda.is_available():
    raise RuntimeError("preflight cannot see the allocated RTX GPU")
CLIPTokenizer.from_pretrained(name, subfolder="tokenizer", local_files_only=True)
DDPMScheduler.from_pretrained(name, subfolder="scheduler", local_files_only=True)
CLIPTextModel.from_pretrained(name, subfolder="text_encoder", local_files_only=True)
AutoencoderKL.from_pretrained(name, subfolder="vae", local_files_only=True)
UNet2DConditionModel.from_pretrained(name, subfolder="unet", local_files_only=True)
print(f"[preflight] torch={torch.__version__}; GPU={torch.cuda.get_device_name(0)}; miniSD complete")
PY

run_slot() {
  local slot="$1"
  ibrun -n 1 -o "$slot" env \
    CUDA_VISIBLE_DEVICES="$slot" \
    REPO_ROOT="$repo" OUTPUT_ROOT="$output_root" \
    PYTHONPATH="$PYTHONPATH" HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    bash "$slot_worker" "$slot" 2 "$task_file"
}

echo "[launch] starting two GPU slots"
pids=()
for slot in 0 1; do
  run_slot "$slot" >"$log_root/slot_${slot}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || {
  echo "ERROR: at least one slot failed; inspect $log_root/slot_*.log" >&2
  exit 1
}

python -m "$plan_module" --mode audit --output-root "$output_root" \
  --top-k 1000 --shard-index "$shard_index" --shard-count "$shard_count"
echo "[done] ArtBench random shard $shard_index/$shard_count complete"
