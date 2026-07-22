#!/usr/bin/env bash
set -euo pipefail

# Submit the compact/original Vista CIFAR2 data-attribution pipeline.
# Run from the repository root:
#   bash diffusion_jax_refined/cifar2/vista_original/submit_vista_original_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export REPO_ROOT
export VISTA_ORIGINAL_DIR="${SCRIPT_DIR}"
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

train_job="$(submit_job sbatch --parsable ./00_train_base_models_vista.sh)"
lds_job="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./01_train_lds_models_vista.sh)"
attr_job_0="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./02a_attribution_priority_vista.sh)"
attr_job_1="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./02b_attribution_priority_vista.sh)"
attr_job_2="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./02c_attribution_priority_vista.sh)"
attr_job_3="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./02d_attribution_priority_vista.sh)"
attr_job_4="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./02e_attribution_priority_vista.sh)"
attr_job_5="$(submit_job sbatch --parsable --dependency=afterok:${train_job} ./02f_attribution_priority_vista.sh)"
attr_jobs="${attr_job_0}:${attr_job_1}:${attr_job_2}:${attr_job_3}:${attr_job_4}:${attr_job_5}"
attr_jobs_csv="${attr_job_0},${attr_job_1},${attr_job_2},${attr_job_3},${attr_job_4},${attr_job_5}"
eval_job="$(submit_job sbatch --parsable --dependency=afterok:${lds_job}:${attr_jobs} ./03_lds_eval_report_vista.sh)"

echo "Submitted Vista CIFAR2 compact/original data-attribution pipeline"
echo "  00 train base checkpoints       : ${train_job}"
echo "  01 train LDS models             : ${lds_job} (afterok:${train_job})"
echo "  02 attribution chunk 0          : ${attr_job_0} (afterok:${train_job})"
echo "  02 attribution chunk 1          : ${attr_job_1} (afterok:${train_job})"
echo "  02 attribution chunk 2          : ${attr_job_2} (afterok:${train_job})"
echo "  02 attribution chunk 3          : ${attr_job_3} (afterok:${train_job})"
echo "  02 attribution chunk 4          : ${attr_job_4} (afterok:${train_job})"
echo "  02 attribution chunk 5          : ${attr_job_5} (afterok:${train_job})"
echo "  03 LDS eval + report            : ${eval_job} (afterok:${lds_job}:${attr_jobs})"
echo
echo "Progress commands:"
echo "  squeue -u \"${USER}\" -j ${train_job},${lds_job},${attr_jobs_csv},${eval_job}"
echo "  sacct -j ${train_job},${lds_job},${attr_jobs_csv},${eval_job} --format=JobID,JobName%32,State,Elapsed,Timelimit,NodeList%24"
echo "  find diffusion_jax_refined/cifar2/result/experiment_67/vista_original_logs -type f -name '*.log' | sort"
