#!/usr/bin/env bash
set -euo pipefail

# Submit the Stampede3 H100 X3 DAS-only pipeline.
#
# Stampede3 h100 has tight job and node limits, so submit this workflow in
# stages. The DAS attribution stage is split into three explicit chunk jobs for
# clearer retry/log handling.

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

    echo "Submitted Stampede3 H100 X3 DAS base/LDS stage"
    echo "  account/project                   : ${STAMPEDE3_ACCOUNT:-${ACCOUNT:-default}}"
    echo "  00 train prompted/unprompted base : ${train_job}"
    echo "  01 train LDS models               : ${lds_job} (afterok:${train_job})"
    echo
    echo "After 00 finishes and submit slots are free, submit attribution with:"
    echo "  STAMPEDE3_ACCOUNT=${STAMPEDE3_ACCOUNT:-${ACCOUNT:-}} EXPERIMENT_TAG=${EXPERIMENT_TAG:-experiment_67} TRAIN_SEED=${TRAIN_SEED:-67} STAMPEDE3_SUBMIT_STAGE=attr bash diffusion_jax_refined/x3/stampede3_das/submit_stampede3_das_pipeline.sh"
    echo
    echo "Progress commands:"
    echo "  squeue -u \"${USER}\" -j ${train_job},${lds_job}"
    echo "  sacct -j ${train_job},${lds_job} --format=JobID,JobName%32,State,Elapsed,Timelimit,NodeList%24"
    ;;
  attr)
    dependency_args=()
    if [[ -n "${TRAIN_JOB_ID:-}" ]]; then
      dependency_args=(--dependency=afterok:${TRAIN_JOB_ID})
    fi
    attr_job_a="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" "${dependency_args[@]}" ./02a_das_attribution_stampede3.sh)"
    attr_job_b="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" "${dependency_args[@]}" ./02b_das_attribution_stampede3.sh)"
    attr_job_c="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" "${dependency_args[@]}" ./02c_das_attribution_stampede3.sh)"
    attr_jobs="${attr_job_a}:${attr_job_b}:${attr_job_c}"
    echo "Submitted Stampede3 H100 X3 DAS attribution stage"
    echo "  account/project              : ${STAMPEDE3_ACCOUNT:-${ACCOUNT:-default}}"
    echo "  02a DAS attribution chunk 0  : ${attr_job_a}${TRAIN_JOB_ID:+ (afterok:${TRAIN_JOB_ID})}"
    echo "  02b DAS attribution chunk 1  : ${attr_job_b}${TRAIN_JOB_ID:+ (afterok:${TRAIN_JOB_ID})}"
    echo "  02c DAS attribution chunk 2  : ${attr_job_c}${TRAIN_JOB_ID:+ (afterok:${TRAIN_JOB_ID})}"
    echo
    echo "After 01 LDS and 02 attribution finish, submit eval with:"
    echo "  STAMPEDE3_ACCOUNT=${STAMPEDE3_ACCOUNT:-${ACCOUNT:-}} EXPERIMENT_TAG=${EXPERIMENT_TAG:-experiment_67} TRAIN_SEED=${TRAIN_SEED:-67} STAMPEDE3_SUBMIT_STAGE=eval LDS_JOB_ID=<01_job_id> ATTR_JOB_IDS=${attr_jobs} bash diffusion_jax_refined/x3/stampede3_das/submit_stampede3_das_pipeline.sh"
    echo
    echo "Progress commands:"
    echo "  squeue -u \"${USER}\" -j ${attr_job_a},${attr_job_b},${attr_job_c}"
    echo "  sacct -j ${attr_job_a},${attr_job_b},${attr_job_c} --format=JobID,JobName%32,State,Elapsed,Timelimit,NodeList%24"
    ;;
  eval)
    lds_job="${LDS_JOB_ID:?Set LDS_JOB_ID to the 01 LDS job id.}"
    attr_jobs="${ATTR_JOB_IDS:-${ATTR_JOB_ID:-}}"
    if [[ -z "${attr_jobs}" ]]; then
      echo "Set ATTR_JOB_IDS to the colon-separated 02a:02b:02c attribution job ids." >&2
      exit 2
    fi
    eval_job="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" --dependency=afterok:${lds_job}:${attr_jobs} ./03_das_lds_eval_report_stampede3.sh)"
    echo "Submitted Stampede3 H100 X3 DAS eval stage"
    echo "  account/project           : ${STAMPEDE3_ACCOUNT:-${ACCOUNT:-default}}"
    echo "  03 DAS LDS eval + report  : ${eval_job} (afterok:${lds_job}:${attr_jobs})"
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
echo "  find diffusion_jax_refined/x3/result/${EXPERIMENT_TAG:-experiment_67}/stampede3_das_logs -type f -name '*.log' | sort"
