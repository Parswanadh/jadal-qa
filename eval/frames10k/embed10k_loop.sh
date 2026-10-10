#!/bin/bash
# Chunked EG2 embedding of egocentric_10k under flock; each hold encodes at
# most 2x1000 frames (short holds, <=10 min). Reruns resume from shards.
cd /home/parshu/projects/jadal-qa-wt/frames10k || exit 1
for i in $(seq 1 12); do
  if [ -f eval/results/frames10k/egocentric_10k/embeddings.fp16.npy ]; then
    echo "chunk-loop: concat already done, stopping" >> eval/results/frames10k/embed10k.log
    break
  fi
  echo "chunk-loop iter $i $(date -u +%FT%TZ)" >> eval/results/frames10k/embed10k.log
  flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock \
    python3 eval/frames10k/embed_frames.py \
      --parquet /workspace/jadal_pod_bundle/data/ego10k_eval/egocentric_10k.parquet \
      --out-dir eval/results/frames10k/egocentric_10k \
      --batch-size 32 --max-new-shards 2 >> eval/results/frames10k/embed10k.log 2>&1
  sleep 5
done
echo "chunk-loop finished $(date -u +%FT%TZ)" >> eval/results/frames10k/embed10k.log
# Stage 2: human-segment frames (whole-segment coverage, ~3000 frames, one hold)
if [ -f eval/results/frames10k/egocentric_10k/embeddings.fp16.npy ] && [ ! -f eval/results/frames10k/human_segments/segments.npz ]; then
  echo "seg-stage start $(date -u +%FT%TZ)" >> eval/results/frames10k/embed10k.log
  flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock \
    python3 eval/frames10k/embed_segments.py \
      --batch-size 32 >> eval/results/frames10k/embed10k.log 2>&1
  echo "seg-stage done $(date -u +%FT%TZ)" >> eval/results/frames10k/embed10k.log
fi
