#!/usr/bin/env bash
#SBATCH -J art-t1k-260
#SBATCH -o art-t1k-260-%j.out
#SBATCH -e art-t1k-260-%j.err
#SBATCH -p h100
#SBATCH -A IRI26004
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH -t 48:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$script_dir}}"
marker="stable_diffusion_artbench_pytorch/src/sd_artbench/artbench_top1000_260_plan.py"
while [[ "$repo" != / && ! -f "$repo/$marker" ]]; do repo="$(dirname "$repo")"; done
[[ -f "$repo/$marker" ]] || { echo "ERROR: repository not found" >&2; exit 2; }

artbench="$repo/stable_diffusion_artbench_pytorch"
plan="$artbench/src/sd_artbench/artbench_top1000_260_plan.py"
slot_worker="$artbench/tacc/h100/run_artbench_top1000_260_slot.sh"
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
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24

log_root="$output_root/logs/${SLURM_JOB_ID}"
mkdir -p "$log_root"
task_file="$log_root/tasks.tsv"
python -m sd_artbench.artbench_top1000_260_plan --mode tasks > "$task_file"
[[ "$(wc -l < "$task_file" | tr -d ' ')" == 260 ]] || {
  echo "ERROR: task plan does not contain 260 rows" >&2; exit 2;
}

echo "[definition] full 5000-image candidate universe; remove top-1000 (20%)"
echo "[definition] four attribution methods Q10-Q59 = 200 models"
echo "[definition] deterministic random(seed=[2026,query]) Q0-Q59 = 60 models"
echo "[definition] total=260 independent final-LoRA-only retrains; training seed=42"
echo "[parallel] 4 H100 nodes x 4 GPUs = 16 static resumable slots"
echo "[output] $output_root"
echo "[logs] $log_root"

echo "[preflight 1/4] unit tests"
python "$artbench/tests/test_artbench_top1000_260_h100.py"

echo "[preflight 2/4] exact dataset and score contracts"
python -m sd_artbench.artbench_top1000_260_plan \
  --mode validate --top-k 1000 --random-seed 2026

echo "[preflight 3/4] one allocated H100 and offline miniSD snapshot"
ibrun -n 1 -o 0 env CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

name = "lambdalabs/miniSD-diffusers"
if not torch.cuda.is_available():
    raise RuntimeError("preflight task cannot see an H100")
CLIPTokenizer.from_pretrained(name, subfolder="tokenizer", local_files_only=True)
DDPMScheduler.from_pretrained(name, subfolder="scheduler", local_files_only=True)
CLIPTextModel.from_pretrained(name, subfolder="text_encoder", local_files_only=True)
AutoencoderKL.from_pretrained(name, subfolder="vae", local_files_only=True)
UNet2DConditionModel.from_pretrained(name, subfolder="unet", local_files_only=True)
print(f"[preflight] torch={torch.__version__}; GPU={torch.cuda.get_device_name(0)}; miniSD complete")
PY

echo "[preflight 4/4] representative ranking directions and random contract"
for spec in \
  endpoint_pollute_query_train_l2:10 \
  endpoint_tracin_raw:10 \
  endpoint_tracin_query_train_l2:59 \
  retrac_query_train_l2:59 \
  random_full5000_seeded:0 \
  random_full5000_seeded:59
do
  method="${spec%%:*}"
  query_index="${spec##*:}"
  python -m sd_artbench.topk_removal \
    --method "$method" --query-index "$query_index" --top-k 1000 \
    --random-seed 2026 --output-root "$output_root" --stage validate
done

run_slot() {
  local slot="$1"
  local local_gpu="$((slot % 4))"
  ibrun -n 1 -o "$slot" env \
    CUDA_VISIBLE_DEVICES="$local_gpu" \
    REPO_ROOT="$repo" OUTPUT_ROOT="$output_root" \
    PYTHONPATH="$PYTHONPATH" HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    bash "$slot_worker" "$slot" 16 "$task_file"
}

echo "[launch] starting 16 static slots"
pids=()
for slot in $(seq 0 15); do
  run_slot "$slot" >"$log_root/slot_${slot}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || {
  echo "ERROR: at least one slot failed; inspect $log_root/slot_*.log" >&2
  exit 1
}

python -m sd_artbench.artbench_top1000_260_plan \
  --mode audit --output-root "$output_root" --top-k 1000
echo "[done] all 260 ArtBench top-1000 removal models trained"
