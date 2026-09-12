#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ACCOUNT_ARGS=()
if [[ -n "${TACC_ACCOUNT:-${ACCOUNT:-}}" ]]; then
  ACCOUNT_ARGS=(-A "${TACC_ACCOUNT:-${ACCOUNT}}")
fi

base_job="$(sbatch --parsable "${ACCOUNT_ARGS[@]}" train_base_rtx_small.sh)"

echo "Submitted base job: ${base_job}"
echo "The base job will submit LDS seeds 0, 1, and 2 one at a time after each stage succeeds."
echo "Monitor with: squeue -u ${USER}"
