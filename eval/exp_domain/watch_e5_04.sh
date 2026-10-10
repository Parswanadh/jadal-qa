#!/usr/bin/env bash
# E5.04 watcher: re-run the frozen-OOD eval automatically WHEN the human labels
# ood_labels2.json appears. Until then it reports "labels missing" (inconclusive).
#
# Usage (background on the pod):
#   nohup bash eval/exp_domain/watch_e5_04.sh >/tmp/opencode/e5_04_watch.log 2>&1 &
# One-shot check:
#   bash eval/exp_domain/watch_e5_04.sh --once
set -u
BUNDLE="${JADAL_BUNDLE:?JADAL_BUNDLE is not set}"
LABELS2="$BUNDLE/labels/ood_labels2.json"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JADAL_EXP_DIR="$REPO/experiments" OMP_NUM_THREADS=1

run() {
  echo "[$(date -u +%FT%TZ)] labels2 present -> re-scoring + confirmatory eval"
  python3 "$REPO/eval/exp_domain/run_e5.py" e5_04
}

if [ "${1:-}" = "--once" ]; then
  if [ -f "$LABELS2" ]; then run; else
    echo "[$(date -u +%FT%TZ)] ood_labels2.json absent -> inconclusive (descriptive only)"
    python3 "$REPO/eval/exp_domain/run_e5.py" e5_04
  fi
  exit 0
fi

echo "watching $LABELS2 (poll 300 s)"
while true; do
  if [ -f "$LABELS2" ]; then run; break; fi
  sleep 300
done
