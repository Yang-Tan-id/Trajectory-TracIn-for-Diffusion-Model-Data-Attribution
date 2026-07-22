#!/usr/bin/env bash
set -euo pipefail

# Submit the Stampede3 H100 CIFAR2 DAS-only pipeline.
#
# Stampede3 h100 counts Slurm array elements toward the submit limit, so the
# 3-task DAS attribution array must be submitted in a later stage if train/LDS
# jobs are still queued.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export REPO_ROOT
export STAMPEDE3_DAS_DIR="${SCRIPT_DIR}"
cd "${SCRIPT_DIR}"

# shellcheck source=_stampede3_das_lib.sh
source "${SCRIPT_DIR}/_stampede3_das_lib.sh"

SBATCH_ACCOUNT_ARGS=()
if [[ -n "${STAMPEDE3_ACCOUNT:-${ACCOUNT:-}}" ]]; then
  SBATCH_ACCOUNT_ARGS=(-A "${STAMPEDE3_ACCOUNT:-${ACCOUNT:-}}")
fi

stage="${STAMPEDE3_SUBMIT_STAGE:-base_lds}"

case "${stage}" in
  base_lds)
    train_job="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" ./00_train_base_models_stampede3.sh)"
    lds_job="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" --dependency=afterok:${train_job} ./01_train_lds_models_stampede3.sh)"

    echo "Submitted Stampede3 H100 CIFAR2 DAS base/LDS stage"
    echo "  account/project                   : ${STAMPEDE3_ACCOUNT:-${ACCOUNT:-default}}"
    echo "  00 train prompted/unprompted base : ${train_job}"
    echo "  01 train LDS models               : ${lds_job} (afterok:${train_job})"
    echo
    echo "After 00 finishes and submit slots are free, submit attribution with:"
    echo "  STAMPEDE3_ACCOUNT=${STAMPEDE3_ACCOUNT:-${ACCOUNT:-}} EXPERIMENT_TAG=${EXPERIMENT_TAG:-experiment_67} TRAIN_SEED=${TRAIN_SEED:-67} STAMPEDE3_SUBMIT_STAGE=attr TRAIN_JOB_ID=${train_job} bash diffusion_jax_refined/cifar2/stampede3_das/submit_stampede3_das_pipeline.sh"
    echo
    echo "Progress commands:"
    echo "  squeue -u \"${USER}\" -j ${train_job},${lds_job}"
    echo "  sacct -j ${train_job},${lds_job} --format=JobID,JobName%32,State,Elapsed,Timelimit,NodeList%24"
    ;;
  attr)
    train_job="${TRAIN_JOB_ID:?Set TRAIN_JOB_ID to the completed/submitted 00 train job id.}"
    attr_job="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" --dependency=afterok:${train_job} ./02_das_attribution_array_stampede3.sh)"
    echo "Submitted Stampede3 H100 CIFAR2 DAS attribution stage"
    echo "  account/project               : ${STAMPEDE3_ACCOUNT:-${ACCOUNT:-default}}"
    echo "  02 DAS attribution array 0-2  : ${attr_job} (afterok:${train_job})"
    echo
    echo "After 01 LDS and 02 attribution finish, submit eval with:"
    echo "  STAMPEDE3_ACCOUNT=${STAMPEDE3_ACCOUNT:-${ACCOUNT:-}} EXPERIMENT_TAG=${EXPERIMENT_TAG:-experiment_67} TRAIN_SEED=${TRAIN_SEED:-67} STAMPEDE3_SUBMIT_STAGE=eval LDS_JOB_ID=<01_job_id> ATTR_JOB_ID=${attr_job} bash diffusion_jax_refined/cifar2/stampede3_das/submit_stampede3_das_pipeline.sh"
    echo
    echo "Progress commands:"
    echo "  squeue -u \"${USER}\" -j ${attr_job}"
    echo "  sacct -j ${attr_job} --format=JobID,JobName%32,State,Elapsed,Timelimit,NodeList%24"
    ;;
  eval)
    lds_job="${LDS_JOB_ID:?Set LDS_JOB_ID to the 01 LDS job id.}"
    attr_job="${ATTR_JOB_ID:?Set ATTR_JOB_ID to the 02 attribution array job id.}"
    eval_job="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" --dependency=afterok:${lds_job}:${attr_job} ./03_das_lds_eval_report_stampede3.sh)"
    echo "Submitted Stampede3 H100 CIFAR2 DAS eval stage"
    echo "  account/project           : ${STAMPEDE3_ACCOUNT:-${ACCOUNT:-default}}"
    echo "  03 DAS LDS eval + report  : ${eval_job} (afterok:${lds_job}:${attr_job})"
    echo
    echo "Progress commands:"
    echo "  squeue -u \"${USER}\" -j ${eval_job}"
    echo "  sacct -j ${eval_job} --format=JobID,JobName%32,State,Elapsed,Timelimit,NodeList%24"
    ;;
  *)
    echo "Unknown STAMPEDE3_SUBMIT_STAGE=${stage}. Expected base_lds, attr, or eval." >&2
    exit 2
    ;;
esac

echo "Logs:"
echo "  find diffusion_jax_refined/cifar2/result/${EXPERIMENT_TAG:-experiment_67}/stampede3_das_logs -type f -name '*.log' | sort"
