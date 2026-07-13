#!/usr/bin/env bash
set -euo pipefail

# Submit the full Vista pipeline with explicit Slurm dependencies.
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

train_job="$(submit_job sbatch --parsable ./00_train_lds_50pct_vista.sh)"
attr_job="$(submit_job sbatch --parsable ./01_sample_and_attribute_vista.sh)"
eval_job="$(submit_job sbatch --parsable --dependency=afterok:${train_job}:${attr_job} ./02_eval_and_aggregate_vista.sh)"

echo "Submitted Vista CIFAR2 pipeline"
echo "  LDS train        : ${train_job}"
echo "  sample+attribute : ${attr_job}"
echo "  eval+aggregate   : ${eval_job} (afterok:${train_job}:${attr_job})"
echo
echo "Progress commands:"
echo "  squeue -u \"${USER}\" -j ${train_job},${attr_job},${eval_job}"
echo "  sacct -j ${train_job},${attr_job},${eval_job} --format=JobID,JobName%28,State,Elapsed,Timelimit,NodeList%24"
echo "  tail -f cifar2-lds-50pct-${train_job}.out"
echo "  tail -f cifar2-attr-norm-vista-${attr_job}.out"
echo "  tail -f cifar2-eval-vista-${eval_job}.out"
