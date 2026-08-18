#!/usr/bin/env bash
set -euo pipefail

# Clean scoring-heavy NewRef/FutureMix outputs on TACC.
# Default is dry-run. Set DELETE=1 to remove matched paths.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CIFAR2_ROOT="${CIFAR2_ROOT:-${REPO_ROOT}/diffusion_jax_refined/cifar2}"
RESULT_ROOT="${RESULT_ROOT:-${CIFAR2_ROOT}/result}"
DELETE="${DELETE:-0}"

if [[ ! -d "${RESULT_ROOT}" ]]; then
  echo "RESULT_ROOT not found: ${RESULT_ROOT}" >&2
  echo "Run from the repo root on TACC, or set RESULT_ROOT=/path/to/diffusion_jax_refined/cifar2/result." >&2
  exit 2
fi

tmp_matches="$(mktemp "${TMPDIR:-/tmp}/newref-futuremix-scoring.XXXXXX")"
trap 'rm -f "${tmp_matches}"' EXIT

find "${RESULT_ROOT}" \
  \( \
    -path "*/attribution_score/*" \
    -o -path "*/projected_traj_tracin_artifacts/*/stream_scores/*" \
    -o -path "*/projected_traj_tracin_artifacts/*/stream_term_scores/*" \
  \) \
  \( -ipath "*NewRef*" -o -ipath "*FutureMix*" \) \
  -type d -print0 \
| sort -z >"${tmp_matches}"

TARGETS=()
last=""
while IFS= read -r -d '' path; do
  if [[ -n "${last}" && "${path}" == "${last}/"* ]]; then
    continue
  fi
  TARGETS+=("${path}")
  last="${path}"
done <"${tmp_matches}"

if (( ${#TARGETS[@]} == 0 )); then
  echo "No NewRef/FutureMix scoring paths found under ${RESULT_ROOT}."
  exit 0
fi

echo "Matched ${#TARGETS[@]} scoring/artifact directories under:"
echo "  ${RESULT_ROOT}"
echo
du -sh "${TARGETS[@]}" 2>/dev/null || true
echo

if [[ "${DELETE}" != "1" ]]; then
  echo "Dry run only. Re-run with DELETE=1 to remove these directories."
  exit 0
fi

echo "Deleting matched NewRef/FutureMix scoring directories..."
rm -rf -- "${TARGETS[@]}"
echo "Done."
