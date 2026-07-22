#!/usr/bin/env bash
set -euo pipefail

# Submit the Stampede3 H100 CIFAR2 DAS-only pipeline.
# Uses 4 submitted jobs to fit Stampede3 h100 Max Submit=4:
#   00 base train, 01 LDS, 02 array chunks 0-2, 03 eval.

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

train_job="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" ./00_train_base_models_stampede3.sh)"
lds_job="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" --dependency=afterok:${train_job} ./01_train_lds_models_stampede3.sh)"
attr_job="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" --dependency=afterok:${train_job} ./02_das_attribution_array_stampede3.sh)"
eval_job="$(submit_job sbatch --parsable "${SBATCH_ACCOUNT_ARGS[@]}" --dependency=afterok:${lds_job}:${attr_job} ./03_das_lds_eval_report_stampede3.sh)"

echo "Submitted Stampede3 H100 CIFAR2 DAS-only pipeline"
echo "  account/project                 : ${STAMPEDE3_ACCOUNT:-${ACCOUNT:-default}}"
echo "  00 train prompted/unprompted base : ${train_job}"
echo "  01 train LDS models              : ${lds_job} (afterok:${train_job})"
echo "  02 DAS attribution array 0-2     : ${attr_job} (afterok:${train_job})"
echo "  03 DAS LDS eval + report         : ${eval_job} (afterok:${lds_job}:${attr_job})"
echo
echo "Progress commands:"
echo "  squeue -u \"${USER}\" -j ${train_job},${lds_job},${attr_job},${eval_job}"
echo "  sacct -j ${train_job},${lds_job},${attr_job},${eval_job} --format=JobID,JobName%32,State,Elapsed,Timelimit,NodeList%24"
echo "  find diffusion_jax_refined/cifar2/result/${EXPERIMENT_TAG:-experiment_67}/stampede3_das_logs -type f -name '*.log' | sort"
