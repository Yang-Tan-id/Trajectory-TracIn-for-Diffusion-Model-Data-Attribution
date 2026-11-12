#!/usr/bin/env bash
set -euo pipefail

project="${ARTBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="$project/src${PYTHONPATH:+:$PYTHONPATH}"
gpus="${GPU_IDS:-0,1,2,3}"
namespace="${DAS_NAMESPACE:-artbench_das_50x1_full5000_dual32768_sm}"
output_root="${DAS_OUTPUT_ROOT:-$project/outputs/attribution/$namespace}"
query_ids="${QUERY_IDS:-$(seq -s, 0 59)}"
lambdas="${DAS_LAMBDAS:-0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000}"
run_lds="${DAS_RUN_LDS:-0}"
proj_dim="${DAS_PROJ_DIM:-32768}"
[[ "$proj_dim" == "32768" ]] || { echo "DAS_PROJ_DIM must be 32768 for the shared ArtBench projection" >&2; exit 2; }

IFS=',' read -r -a gpu_ids <<<"$gpus"
[[ ${#gpu_ids[@]} -eq 4 ]] || { echo "GPU_IDS must contain four GPUs" >&2; exit 2; }
mkdir -p "$output_root/train_shards" "$output_root/query_shards" "$output_root/logs" "$output_root/scores"

required=(
  "$project/dataset/metadata/ArtBench-10.csv"
  "$project/dataset/indices/idx-train.pkl"
  "$project/outputs/artbench2-queries-60/manifest.json"
)
if [[ "$run_lds" == "1" ]]; then
  required+=("$project/outputs/artbench2-lds/evaluation/four_targets_q0_59/true_f_four_targets.csv")
fi
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "MISSING: $path" >&2; exit 1; }
done
compgen -G "$project/outputs/artbench2-lora/checkpoint-epoch-*" >/dev/null || {
  echo "MISSING: $project/outputs/artbench2-lora/checkpoint-epoch-*" >&2
  exit 1
}
"$python_bin" - <<'PY'
import torch, diffusers, transformers
assert torch.cuda.is_available(), "ArtBench DAS requires CUDA"
print(f"[preflight] torch={torch.__version__} GPUs={torch.cuda.device_count()}")
PY

query_artifact="$output_root/query_artifact_q0_q59.npz"
cross_artifact="$output_root/cross_kernel_q0_q59.npz"
timestep_shards=(
  "999,919,839,759,679,599,519,439,359,279,199,119,39"
  "979,899,819,739,659,579,499,419,339,259,179,99,19"
  "959,879,799,719,639,559,479,399,319,239,159,79"
  "939,859,779,699,619,539,459,379,299,219,139,59"
)

echo "[definition] ArtBench DAS: latent residual, LoRA-only exact gradients"
echo "[definition] 50 DDIM timesteps x MC1; full 5000; projection $proj_dim"
echo "[preprocessing] FP16 train features + FP32 dual eigensystem; cached train-query cross-kernel"
echo "[definition] squared contraction; Sherman-Morrison denominator enabled"
echo "[queries] $query_ids"

pids=()
for slot in 0 1 2 3; do
  shard="$output_root/train_shards/train_${slot}.npz"
  [[ -s "$shard" ]] && continue
  CUDA_VISIBLE_DEVICES="${gpu_ids[$slot]}" "$python_bin" -m sd_artbench.artbench_das \
    --mode train --timesteps "${timestep_shards[$slot]}" --proj-dim "$proj_dim" \
    --batch-size "${DAS_TRAIN_BATCH_SIZE:-2}" --output "$shard" \
    >"$output_root/logs/train_gpu_${gpu_ids[$slot]}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
for slot in 0 1 2 3; do
  [[ -s "$output_root/train_shards/train_${slot}.npz" ]] || {
    echo "MISSING TRAIN SHARD: $slot" >&2
    exit 1
  }
done

if [[ ! -s "$query_artifact" ]]; then
  pids=()
  for slot in 0 1 2 3; do
    ids="$($python_bin - "$query_ids" "$slot" <<'PY'
import sys
ids=[int(x) for x in sys.argv[1].split(',')]
print(','.join(map(str,ids[int(sys.argv[2])::4])))
PY
)"
    shard="$output_root/query_shards/query_${slot}.npz"
    [[ -s "$shard" ]] && continue
    CUDA_VISIBLE_DEVICES="${gpu_ids[$slot]}" "$python_bin" -m sd_artbench.artbench_das \
      --mode query --query-ids "$ids" --proj-dim "$proj_dim" \
      --query-batch-size "${DAS_QUERY_BATCH_SIZE:-4}" --output "$shard" \
      >"$output_root/logs/query_gpu_${gpu_ids[$slot]}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$python_bin" -m sd_artbench.artbench_das --mode merge --merge-kind query \
    --inputs "$output_root"/query_shards/query_*.npz --output "$query_artifact"
fi

if [[ ! -s "$cross_artifact" ]]; then
  pids=()
  for slot in 0 1 2 3; do
    shard="$output_root/cross_${slot}.npz"
    CUDA_VISIBLE_DEVICES="${gpu_ids[$slot]}" "$python_bin" -m sd_artbench.artbench_das \
      --mode cross --train-artifact "$output_root/train_shards/train_${slot}.npz" \
      --query-artifact "$query_artifact" --proj-dim "$proj_dim" --output "$shard" \
      >"$output_root/logs/cross_gpu_${gpu_ids[$slot]}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$python_bin" -m sd_artbench.artbench_das --mode merge --merge-kind cross \
    --inputs "$output_root"/cross_*.npz --output "$cross_artifact"
fi

IFS=' ' read -r -a lambda_values <<<"$lambdas"
pids=()
for slot in 0 1 2 3; do
  assigned=""
  for ((index=slot; index<${#lambda_values[@]}; index+=4)); do
    assigned+="${lambda_values[$index]},"
  done
  assigned="${assigned%,}"
  CUDA_VISIBLE_DEVICES="${gpu_ids[$slot]}" "$python_bin" -m sd_artbench.artbench_das \
    --mode score --train-artifacts "$output_root"/train_shards/train_*.npz \
    --cross-artifact "$cross_artifact" --proj-dim "$proj_dim" \
    --lambdas "$assigned" --output-dir "$output_root/scores" \
    >"$output_root/logs/score_gpu_${gpu_ids[$slot]}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

if [[ "$run_lds" != "1" ]]; then
  echo "[done] DAS train/query artifacts and lambda-sweep scores: $output_root"
  echo "[skip] LDS was not requested; set DAS_RUN_LDS=1 after true-f is available"
  exit 0
fi

for value in "${lambda_values[@]}"; do
  "$python_bin" -m sd_artbench.artbench_das --mode lds \
    --score-path "$output_root/scores/lambda_${value}/scores.npz" \
    --lds-lambda "$value" --output "$output_root/lds_lambda_${value}.txt"
done

"$python_bin" - "$output_root" "${lambda_values[*]}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
lambdas = [float(value) for value in sys.argv[2].split()]
records = {}
pattern = re.compile(r"^(\S+)\s+mean=\s*([+-]?[0-9.]+)%")
for value in lambdas:
    tag = f"{value:g}"
    for line in (root / f"lds_lambda_{tag}.txt").read_text().splitlines():
        match = pattern.match(line)
        if match:
            records.setdefault(match.group(1), []).append((float(match.group(2)), value))
lines = ["ARTBENCH DAS 50X1 LAMBDA SWEEP", "=" * 72]
for target, values in records.items():
    best, damping = max(values)
    lines.append(f"{target:28s} BEST MEAN: lambda={damping:g} LDS={best:+.3f}%")
report = root / "lds_lambda_sweep_report.txt"
report.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"[saved] {report}")
PY

echo "[done] $output_root"
