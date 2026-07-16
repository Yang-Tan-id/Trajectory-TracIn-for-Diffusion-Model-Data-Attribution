#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (($# > 0)); then
  MODES_TEXT="$*"
else
  MODES_TEXT="${TRAIN_MODES:-${TRAIN_MODE:-prompted_solo}}"
fi
MODES_TEXT="${MODES_TEXT//,/ }"

for MODE in ${MODES_TEXT}; do
  case "${MODE}" in
    prompted_solo|pm_solo)
      bash "${ROOT}/scripts/00_train_prompted_solo.sh"
      ;;
    prompted_multi|pm_multi)
      bash "${ROOT}/scripts/00_train_prompted_multi.sh"
      ;;
    unprompted_solo|up_solo)
      bash "${ROOT}/scripts/00_train_unprompted_solo.sh"
      ;;
    unprompted_multi|up_multi)
      bash "${ROOT}/scripts/00_train_unprompted_multi.sh"
      ;;
    *)
      echo "Unknown TRAIN_MODE=${MODE}" >&2
      echo "Expected: prompted_solo, prompted_multi, unprompted_solo, unprompted_multi" >&2
      exit 2
      ;;
  esac
done
