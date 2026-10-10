#!/usr/bin/env bash
# Full regeneration of the E1 fps / frame-selection results on the pod.
# One process = ~1 core; 6 processes at a time (pod rule: <=75% of cores).
# usage: bash eval/exp_fps/run_all.sh          (from the repo root)
set -euo pipefail
export JADAL_BUNDLE="${JADAL_BUNDLE:?JADAL_BUNDLE is not set}"
export OMP_NUM_THREADS=1
cd "$(dirname "$0")/../.."
LOG=${LOG:-/tmp/opencode/exp_fps}
mkdir -p "$LOG" eval/results/exp_fps

echo "== E1.01 sweep shards (IndEgo 480p in-domain, 150 human windows)"
for s in 0 1; do python3 eval/exp_fps/run_fps.py --mode sweep --shard $s/2 >"$LOG/sweep_$s.log" 2>&1 & done
wait
python3 eval/exp_fps/merge_e1_01.py

echo "== E1.04 select shards (equal 20-frame budget)"
for s in 0 1; do python3 eval/exp_fps/run_fps.py --mode select --shard $s/2 >"$LOG/select_$s.log" 2>&1 & done
wait
python3 eval/exp_fps/merge_e1_04.py

echo "== E1.03 OOD shards (original 1080p30 sources)"
for s in 0 1; do python3 eval/exp_fps/run_ood.py --queue 1 --shard $s/2 >"$LOG/ood1_$s.log" 2>&1 & done
for s in 0 1 2 3; do python3 eval/exp_fps/run_ood.py --queue 2 --shard $s/4 >"$LOG/ood2_$s.log" 2>&1 & done
wait
python3 eval/exp_fps/merge_ood.py

echo "== E1.02 EG2 (cached embeddings, no decode)"
python3 eval/exp_fps/run_eg2.py

echo "== cost / Pareto"
python3 eval/exp_fps/pareto.py --reps 3 --benches 3
echo "done -> eval/results/exp_fps/{E1.03,E1.04,E1.01,e1_02,pareto}.json"
