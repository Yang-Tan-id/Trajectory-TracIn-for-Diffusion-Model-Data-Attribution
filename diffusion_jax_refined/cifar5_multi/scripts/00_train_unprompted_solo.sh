#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/../common/train_shell_lib.sh"
train_run_unprompted_solo "${ROOT}"
