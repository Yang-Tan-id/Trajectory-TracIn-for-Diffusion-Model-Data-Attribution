#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
celeba="$repo/diffusion_jax_refined/celeba"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
experiment="${EXPERIMENT_TAG:-attr40_experiment1}"
train_seed="${TRAIN_SEED:-42}"
query_file="${QUERY_FILE:-$celeba/queries_attr40_val25_x4.json}"
query_ids="${QUERY_IDS:-$(seq -s, 0 71)}"
gpus="${GPU_IDS:-0,1,2,3}"
namespace="${DAS_NAMESPACE:-das_50x1_mc1}"
lambdas="${DAS_DAMPING_SWEEP_VALUES:-0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000}"

IFS=',' read -r -a gpu_ids <<<"$gpus"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four GPU IDs; got: $gpus" >&2
  exit 2
fi

result_root="$celeba/result/$experiment"
artifact_dir="$result_root/model/prompted_solo/seed_${train_seed}_train_gradient/$namespace"
train_artifact="$artifact_dir/train_datapoint_gradient_artifact.npz"
log_root="$result_root/logs/$namespace"
mkdir -p "$artifact_dir" "$log_root"

required=(
  "$query_file"
  "$repo/diffusion_jax_refined/dataset/celeba/10000_attr40/dataset.npz"
  "$repo/diffusion_jax_refined/dataset/celeba/10000_attr40/attribution_5k_indices.npy"
  "$result_root/model/prompted_jax/seed_${train_seed}_epoch_0200.ckpt"
  "$result_root/query100_lds_csv_ddim50/celeba_attr40_query100_lds_true_f_long.csv"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "MISSING: $path" >&2; exit 1; }
done

export PYTHONUNBUFFERED=1
export EXPERIMENT_TAG="$experiment"
export TRAIN_SEED="$train_seed"
export JAX_EPOCHS=200
export DATAPOINT_MODEL_MODE=prompted_solo
export SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SAMPLE_MODEL_MODE=prompted_solo
export ATTRIBUTION_SCORE_MODEL_MODE=prompted_solo
export DAS_NUM_MC_NOISE=1
export DAS_PROJ_DIM=4096
export DAS_DAMPING_SWEEP=1
export DAS_DAMPING_SWEEP_VALUES="$lambdas"
export DAS_SHERMAN_MORRISON_DENOMINATOR=0
export DAS_SCORE_DENOMINATOR_CACHE=0
export JAX_NUM_DEVICES=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Exact 50-step DDIM grid: {19, 39, ..., 999}.  Split disjointly over 4 GPUs.
timestep_shards=(
  "19,39,59,79,99,119,139,159,179,199,219,239,259"
  "279,299,319,339,359,379,399,419,439,459,479,499,519"
  "539,559,579,599,619,639,659,679,699,719,739,759"
  "779,799,819,839,859,879,899,919,939,959,979,999"
)
shards=()

echo "[definition] CelebA DAS: 50 DDIM timesteps x MC1; 5000 fixed attribution points"
echo "[lambda] $lambdas"
echo "[queries] $query_ids"
echo "[python] $python_bin"

if [[ ! -s "$train_artifact" ]]; then
  echo "[phase 1/4] four-GPU DAS train features"
  pids=()
  cd "$celeba/data_attribution/das"
  for slot in 0 1 2 3; do
    shard="$artifact_dir/train_datapoint_gradient_artifact.timestamps_${slot}.npz"
    shards+=("$shard")
    if [[ -s "$shard" ]]; then
      echo "[skip] train shard $slot exists"
      continue
    fi
    (
      export DAS_TIMESTEPS="${timestep_shards[$slot]}"
      export TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH="$shard"
      export DAS_GLOBAL_GRAM_ARTIFACT_PATH="$shard"
      export CUDA_VISIBLE_DEVICES="${gpu_ids[$slot]}"
      export JAX_PLATFORMS=cuda
      "$python_bin" 01_train_datapoint_gradient.py
    ) >"$log_root/train_gpu_${gpu_ids[$slot]}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  "$python_bin" "$repo/diffusion_jax_refined/common/merge_das_term_shards.py" \
    --output "$train_artifact" "${shards[@]}"
else
  echo "[skip] complete DAS train artifact exists: $train_artifact"
fi

"$python_bin" - "$train_artifact" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
expected_t = np.arange(19, 1000, 20, dtype=np.int32)
with np.load(path, allow_pickle=False) as payload:
    assert payload["train_features"].shape[:2] == (50, 5000), payload["train_features"].shape
    assert payload["train_features"].shape[2] == 4096, payload["train_features"].shape
    assert np.array_equal(payload["timesteps"], expected_t), payload["timesteps"]
    assert np.all(payload["mc_indices"] == 0), payload["mc_indices"]
    assert len(np.unique(payload["score_indices"])) == 5000
print("TRAIN ARTIFACT CHECK PASSED: 50x1, 5000 points, projection 4096")
PY

echo "[phase 2/4] query artifacts"
cd "$celeba"
"$python_bin" script/run_das_50x1_queries_and_scores.py \
  --execute \
  --experiment "$experiment" \
  --train-seed "$train_seed" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --gpus "$gpus" \
  --namespace "$namespace" \
  --damping-values "$lambdas" \
  --skip-score \
  --python-bin "$python_bin"

echo "[phase 3/4] lambda-sweep scores"
"$python_bin" script/run_das_50x1_queries_and_scores.py \
  --execute \
  --experiment "$experiment" \
  --train-seed "$train_seed" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --gpus "$gpus" \
  --namespace "$namespace" \
  --damping-values "$lambdas" \
  --skip-query-gradient \
  --python-bin "$python_bin"

echo "[phase 4/4] cached 192-model LDS"
JAX_PLATFORMS=cpu "$python_bin" script/run_das_50x1_cached_lds.py \
  --experiment "$experiment" \
  --train-seed "$train_seed" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --namespace "$namespace" \
  --lambdas "$lambdas" \
  --prediction-sign -1

first_query="${query_ids%%,*}"
last_query="${query_ids##*,}"
report="$result_root/celeba_${namespace}_lambda_sweep_lds_q${first_query}_${last_query}.txt"
echo "[done] $report"
cat "$report"

