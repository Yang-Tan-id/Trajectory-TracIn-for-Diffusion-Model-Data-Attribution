#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ACCOUNT_ARGS=()
if [[ -n "${TACC_ACCOUNT:-${ACCOUNT:-}}" ]]; then
  ACCOUNT_ARGS=(-A "${TACC_ACCOUNT:-${ACCOUNT}}")
fi

base_job="$(sbatch --parsable "${ACCOUNT_ARGS[@]}" train_base_rtx_small.sh)"
lds_job="$(sbatch --parsable "${ACCOUNT_ARGS[@]}" --dependency="afterok:${base_job}" train_lds_rtx_small_array.sh)"

echo "Submitted base job: ${base_job}"
echo "Submitted LDS array: ${lds_job} (starts after base succeeds)"
echo "Monitor with: squeue -u ${USER}"
