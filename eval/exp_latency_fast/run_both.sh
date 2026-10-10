#!/usr/bin/env bash
# L3.01 orchestration: full-corpus optimized run (stride 1) + adopted every-2nd
# policy run (stride 2). Both at full resources (7 workers, 6 GPU slots).
set -u
cd "$(dirname "$0")/../.."   # repo root (worktree)
source "${JADAL_ENV:-env.sh}"
export OMP_NUM_THREADS=1
export JADAL_GPU_SLOTS="${JADAL_GPU_SLOTS:-6}"
OUT=eval/results/exp_latency_fast
mkdir -p "$OUT"

COMMON="--pixel-fps 3.33 --eg2-fps 1 --early-exit-s 4 --skip-mosaic --no-compare --workers 7"

echo "=== RUN 1: full corpus, optimized per-window, stride 1 ==="
bash tools/pod/gpu_run.sh python3 eval/exp_latency_fast/run_fast.py \
     $COMMON --window-stride 1 --out "$OUT/windows.jsonl"
rc1=$?
cp -f "$OUT/run_meta.json" "$OUT/run_meta_stride1.json" 2>/dev/null
echo "run1 rc=$rc1"

echo "=== RUN 2: adopted policy, every-2nd window, stride 2 ==="
bash tools/pod/gpu_run.sh python3 eval/exp_latency_fast/run_fast.py \
     $COMMON --window-stride 2 --out "$OUT/windows_stride2.jsonl"
rc2=$?
cp -f "$OUT/run_meta.json" "$OUT/run_meta_stride2.json" 2>/dev/null
cp -f "$OUT/run_meta_stride1.json" "$OUT/run_meta.json" 2>/dev/null
echo "run2 rc=$rc2"
echo "DONE run_both rc1=$rc1 rc2=$rc2"
