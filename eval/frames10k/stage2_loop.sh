#!/bin/bash
# Stage 2 loop: segments FIRST (transfer priority), then remaining 10k shards.
# All GPU work under flock; thread caps on every python (resource rules).
cd /home/parshu/projects/jadal-qa-wt/frames10k || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
LOG=eval/results/frames10k/embed10k.log
PY=python3
LOCK=/home/parshu/projects/jadal-qa-wt/tools/gpu.lock
if [ ! -f eval/results/frames10k/human_segments/segments.npz ]; then
  echo "seg-stage start $(date -u +%FT%TZ)" >> $LOG
  flock -w 3600 $LOCK $PY eval/frames10k/embed_segments.py --batch-size 32 --gpu-decode >> $LOG 2>&1
  echo "seg-stage exit $? $(date -u +%FT%TZ)" >> $LOG
fi
for i in $(seq 1 6); do
  if [ -f eval/results/frames10k/egocentric_10k/embeddings.fp16.npy ]; then
    echo "10k concat done, stopping" >> $LOG
    break
  fi
  echo "gpu-loop iter $i $(date -u +%FT%TZ)" >> $LOG
  flock -w 3600 $LOCK $PY eval/frames10k/embed_frames.py \
    --parquet /workspace/jadal_pod_bundle/data/ego10k_eval/egocentric_10k.parquet \
    --out-dir eval/results/frames10k/egocentric_10k \
    --batch-size 32 --max-new-shards 3 --gpu-decode >> $LOG 2>&1
  [ -f eval/results/frames10k/egocentric_10k/embeddings.fp16.npy ] && break
done
echo "stage2-loop finished $(date -u +%FT%TZ)" >> $LOG
