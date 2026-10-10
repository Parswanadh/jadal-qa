#!/usr/bin/env bash
# LB4 — one command that runs the FROZEN pre-registered OOD evaluation.
#
#   bash tools/pod/run_ood_frozen.sh
#
# It refuses to score (non-zero exit, clear message, results JSON still written)
# unless BOTH hold:
#   1. labels/ood_labels2.json exists with exactly 100 labelled windows;
#   2. the frozen artifact's raw sha256 equals the recorded pre-registered hash
#      4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31.
# Exit codes: 0 COMPLETE, 2 ABORT (guard failed), 3 PENDING (labels not delivered),
# 1 ERROR. See docs/reports/ood_preregistration.md.
set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "${ROOT}/eval/exp_ood/run_ood_frozen.py" "$@"
