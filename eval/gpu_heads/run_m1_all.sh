#!/bin/bash
# Sequential M1 GPU runs, one flock acquisition per run (each holds lock only while training).
# Logs to eval/results/gpu_heads/m1_all.log
cd /home/parshu/projects/jadal-qa-wt/gpu-heads || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/parshu/envs/jadal/bin/python
LOCK=/home/parshu/projects/jadal-qa-wt/tools/gpu.lock
LOG=eval/results/gpu_heads/m1_all.log
mkdir -p eval/results/gpu_heads
for spec in "spec hands" "spec work" "eg2 hands" "eg2 work" "both hands" "both work"; do
  # shellcheck disable=SC2086
  set -- $spec
  echo "=== START feat=$1 target=$2 $(date -u +%FT%TZ) ===" >> "$LOG"
  flock -w 3600 "$LOCK" $PY eval/gpu_heads/run_m1.py --feature "$1" --target "$2" --epochs 20 --device cuda >> "$LOG" 2>&1
  echo "=== EXIT($?) feat=$1 target=$2 $(date -u +%FT%TZ) ===" >> "$LOG"
done
echo "=== ALL DONE $(date -u +%FT%TZ) ===" >> "$LOG"
