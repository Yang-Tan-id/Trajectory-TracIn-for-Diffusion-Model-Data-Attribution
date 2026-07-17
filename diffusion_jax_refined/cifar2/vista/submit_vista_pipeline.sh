#!/usr/bin/env bash
set -euo pipefail

# Submit the six-job Vista CIFAR2 data-attribution pipeline.
# Run from the repository root:
#   bash diffusion_jax_refined/cifar2/vista/submit_vista_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

submit_job() {
  local output job_id
  output="$("$@")"
  printf '%s\n' "${output}" >&2
  job_id="$(printf '%s\n' "${output}" | grep -Eo '^[0-9]+(;[0-9]+)?$' | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "Could not parse job id from sbatch output above." >&2
    exit 1
  fi
  printf '%s' "${job_id%%;*}"
}

train_job="$(submit_job sbatch --parsable ./00_train_four_models_vista.sh)"
lds_job="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./01_train_lds_models_vista.sh)"
train_grad_job="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./02_train_datapoint_gradients_vista.sh)"
sample_qgrad_job="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./03_sample_query_gradients_vista.sh)"
score_job="$(submit_job sbatch --parsable --dependency=afterok:${lds_job}:${train_grad_job}:${sample_qgrad_job} ./04_score_vista.sh)"
eval_job="$(submit_job sbatch --parsable --dependency=afterok:${score_job} ./05_lds_eval_report_vista.sh)"

echo "Submitted Vista CIFAR2 refined data-attribution pipeline"
echo "  00 train base checkpoints  : ${train_job}"
echo "  01 train LDS models        : ${lds_job} (afterok:${train_job})"
echo "  02 train datapoint grads   : ${train_grad_job} (afterok:${train_job})"
echo "  03 sample + query grads    : ${sample_qgrad_job} (afterok:${train_job})"
echo "  04 score                   : ${score_job} (afterok:${lds_job}:${train_grad_job}:${sample_qgrad_job})"
echo "  05 LDS eval + report       : ${eval_job} (afterok:${score_job})"
echo
echo "Progress commands:"
echo "  squeue -u \"${USER}\" -j ${train_job},${lds_job},${train_grad_job},${sample_qgrad_job},${score_job},${eval_job}"
echo "  sacct -j ${train_job},${lds_job},${train_grad_job},${sample_qgrad_job},${score_job},${eval_job} --format=JobID,JobName%32,State,Elapsed,Timelimit,NodeList%24"
echo "  find diffusion_jax_refined/cifar2/result/experiment_67/vista_logs -type f -name '*.log' | sort"
