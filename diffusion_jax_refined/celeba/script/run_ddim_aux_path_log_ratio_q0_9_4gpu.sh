#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$HOME/diffusion_model/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution}"
celeba="$repo/diffusion_jax_refined/celeba"
py="${PYTHON_BIN:-$HOME/miniconda3/envs/experiment_dm/bin/python}"
query_file="${QUERY_FILE:-$celeba/queries_attr40_val25_x4.json}"
gpu_ids="${GPU_IDS:-4,5,6,7}"
query_ids="${QUERY_IDS:-$(seq -s, 0 9)}"
result_root="$celeba/result/attr40_experiment1"
output_dir="${OUTPUT_DIR:-$result_root/analysis/ddim_aux_path_log_ratio_eta1_q0_9}"
true_f_csv="$output_dir/celeba_attr40_query100_lds_true_f_long.csv"

raw_pint="celeba_endpoint_pollute_raw_gradient_train1x50_query1x50_timestamp_sum_squared_shape3d_semantics"
raw_tracin="celeba_endpoint_tracin_raw_gradient_train_mean50_query_mean50_linear_shape3d_semantics"
raw_retrac="celeba_paper_retrac_raw_gradient_train_replay4_batch1_query_mean50_of_full_l2_linear_shape3d_semantics"
adamw_pint="celeba_endpoint_pollute_adamw_full_train1x50_query1x50_timestamp_sum_squared_shape3d_semantics"
adamw_tracin="celeba_endpoint_tracin_adamw_full_train_mean50_query_mean50_linear_shape3d_semantics"
adamw_retrac="celeba_paper_retrac_adamw_full_train_replay4_batch1_query_mean50_of_full_l2_linear_shape3d_semantics"

mkdir -p "$output_dir"
cd "$repo"
export PYTHONUNBUFFERED=1
export LDS_DDIM_AUX_ETA=1.0

echo "[definition] CelebA DDIM auxiliary path log-ratio; Q0-Q9 x 192 LDS models"
echo "[definition] deterministic full-model DDIM reference path"
echo "[definition] positive-variance auxiliary kernel eta=1; final zero-variance transition excluded"
echo "[definition] target=sum over 49 transitions of per-coordinate square plus cross term"
echo "[gpus] $gpu_ids"
echo "[output] $output_dir"

"$py" \
  "$celeba/script/run_attr40_query100_sample_lds.py" \
  --execute \
  --skip-sampling \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --gpus "$gpu_ids" \
  --subset-seeds 0,1,2 \
  --target-functions ddim_aux_path_log_ratio \
  --output-dir "$output_dir" \
  --python-bin "$py"

test -s "$true_f_csv" || {
  echo "MISSING TRUE-F CSV: $true_f_csv" >&2
  exit 1
}

echo "[lds] raw-gradient five-method scores"
"$py" "$celeba/script/run_endpoint_cached_lds.py" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --true-f-csv "$true_f_csv" \
  --namespaces "$raw_pint,$raw_tracin,$raw_retrac" \
  --target-functions ddim_aux_path_log_ratio \
  --prediction-sign -1 \
  --output-tag ddim_aux_path_log_ratio_raw_q0_9

echo "[lds] AdamW five-method scores"
"$py" "$celeba/script/run_endpoint_cached_lds.py" \
  --query-file "$query_file" \
  --query-ids "$query_ids" \
  --true-f-csv "$true_f_csv" \
  --namespaces "$adamw_pint,$adamw_tracin,$adamw_retrac" \
  --target-functions ddim_aux_path_log_ratio \
  --prediction-sign 1 \
  --output-tag ddim_aux_path_log_ratio_adamw_q0_9

report="$output_dir/celeba_ddim_aux_path_log_ratio_five_method_q0_9.txt"
"$py" "$repo/diffusion_jax_refined/script/export_pint_lds_txt.py" \
  --dataset celeba \
  --repo-root "$repo" \
  --query-ids 0-9 \
  --targets ddim_aux_path_log_ratio \
  --expected-subsets 192 \
  --output "$report"

echo
cat "$report"
echo "[done] $report"
