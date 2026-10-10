#!/usr/bin/env bash
# Fetch the frames10k transfer artifacts from the (unmerged) producing branch.
#
# The frame-head transfer numbers quoted in FACTS.md (17:40/17:50 IST) are NOT on
# origin/main: eval/results/frames10k/ exists only on origin/feat/frames10k.
# This script extracts the exact blobs so eval/pod_verify/recompute_frame_heads.py
# can be re-run from scratch. Read-only w.r.t. every worktree (uses `git show`).
#
# Usage:  bash eval/pod_verify/fetch_frames10k_artifacts.sh [out_dir]
# Then:   python3 eval/pod_verify/recompute_frame_heads.py <out_dir>
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
OUT="${1:-$REPO/eval/pod_verify/_artifacts}"
BRANCH="${FRAMES10K_REF:-origin/feat/frames10k}"
mkdir -p "$OUT"
for f in \
  eval/results/frames10k/human_segments/segments.npz \
  eval/results/frames10k/human_segments/transfer.json \
  eval/results/frames10k/human_segments/frame_heads.used.json \
  eval/results/frames10k/human_segments/frame_heads.used_10k.json \
  eval/results/frames10k/human_segments/meta.json \
  eval/results/frames10k/transfer_10k.log \
  eval/results/frames10k/transfer_rerun.log ; do
  git -C "$REPO" show "$BRANCH:$f" > "$OUT/$(basename "$f")"
  printf 'extracted %s (%s bytes)\n' "$(basename "$f")" "$(stat -c%s "$OUT/$(basename "$f")")"
done
# 8k-run result JSON (the commit that produced "0.808 / 0.816"):
git -C "$REPO" show 62e09e3:eval/results/frames10k/human_segments/transfer.json > "$OUT/transfer.8k.json"
echo "8k transfer -> $OUT/transfer.8k.json"
echo "done: $OUT"
