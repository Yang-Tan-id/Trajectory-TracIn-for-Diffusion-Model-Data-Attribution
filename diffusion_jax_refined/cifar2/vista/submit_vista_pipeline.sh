#!/usr/bin/env bash
set -euo pipefail

# Submit the full Vista pipeline with explicit Slurm dependencies.
# Run from the repository root:
#   bash diffusion_jax_refined/cifar2/vista/submit_vista_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

train_job="$(
  sbatch "${SCRIPT_DIR}/00_train_lds_50pct_vista.sh" | awk '{print $4}'
)"
attr_job="$(
  sbatch "${SCRIPT_DIR}/01_sample_and_attribute_vista.sh" | awk '{print $4}'
)"
eval_job="$(
  sbatch --dependency=afterok:${train_job}:${attr_job} \
    "${SCRIPT_DIR}/02_eval_and_aggregate_vista.sh" | awk '{print $4}'
)"

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
