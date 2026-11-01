#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$HOME/diffusion_model/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution}"
celeba="$repo/diffusion_jax_refined/celeba"
py="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
query_file="${QUERY_FILE:-$celeba/queries_attr40_val25_x4.json}"
gpu_ids="${GPU_IDS:-4,5,6,7}"
query_ids="${QUERY_IDS:-$(seq -s, 0 99)}"
probe_count=50
probe_seed=2026
target="simple_loss_50x1"
result_root="$celeba/result/attr40_experiment1"
output_dir="${OUTPUT_DIR:-$result_root/analysis/simple_loss_50x1_q0_99}"
true_f_csv="$output_dir/celeba_attr40_query100_lds_true_f_long.csv"
report="$repo/celeba_five_method_lds_q0_99_all.txt"

mkdir -p "$output_dir"
cd "$repo"
export PYTHONUNBUFFERED=1
export LDS_ALLOW_PROMPT_MISMATCH="${LDS_ALLOW_PROMPT_MISMATCH:-1}"

echo "[definition] CelebA simple-loss LDS target: original 50 uniformly spaced timesteps x MC1"
echo "[definition] timesteps=np.linspace(999,0,50,dtype=int32); base probe seed=2026"
echo "[definition] query seed=2026+100003*query_id; probes shared across all 192 models"
echo "[compatibility] using legacy 80-character query paths; prompt mismatch check disabled"
echo "[scope] Q0-Q99; only five listed methods x raw-gradient/AdamW are scored"
echo "[gpus] $gpu_ids"
echo "[output] $output_dir"

"$py" "$celeba/script/run_attr40_query100_sample_lds.py" \
  --execute \
  --skip-sampling \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --gpus "$gpu_ids" \
  --subset-seeds 0,1,2 \
  --target-functions "$target" \
  --predicted-noise-kl-num-directions "$probe_count" \
  --predicted-noise-kl-seed "$probe_seed" \
  --output-dir "$output_dir" \
  --python-bin "$py"

test -s "$true_f_csv" || {
  echo "MISSING TRUE-F CSV: $true_f_csv" >&2
  exit 1
}

"$py" "$celeba/script/evaluate_five_method_simple_loss_50x1.py" \
  --experiment attr40_experiment1 \
  --train-seed 42 \
  --query-file "$query_file" \
  --query-ids 0-99 \
  --true-f-csv "$true_f_csv" \
  --expected-models 192

"$py" "$repo/diffusion_jax_refined/script/export_pint_lds_txt.py" \
  --dataset celeba \
  --repo-root "$repo" \
  --query-ids 0-99 \
  --expected-subsets 192 \
  --targets endpoint_contarfactual,traj_contarfactual,simple_loss_50x1,noise_trajectory \
  --output "$report"

echo
cat "$report"
echo "[done] $report"
