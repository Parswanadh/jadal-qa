"""Parallel extraction of per-frame spectral features for the 10k-frame study.

Reads image bytes from parquet, decodes with cv2, runs
eval.frames_spectral.frame_features.extract_frame_bgr, caches float32.

Usage:
  python -m eval.frames_spectral.extract_all egocentric_10k [--procs 8]
  python -m eval.frames_spectral.extract_all all   # all three domains

Output: eval/results/frames_spectral/<name>.npz with X float32 (n,d),
ids (str), hand_count int8, active_labor int8 (1=yes), plus FEATURES list
and timing (ms_per_frame_single_equiv measured on a 20-frame sample).
Deterministic order (pool.imap preserves input order). <= 8 procs.
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pyarrow.parquet as pq

from eval.frames_spectral.frame_features import FEATURES, extract_frame_bgr

DATA = "/home/parshu/projects/awing/data/ego10k_eval"
OUT = "eval/results/frames_spectral"
NAMES = ["egocentric_10k", "ego4d", "epic_kitchens"]


def _feat_of(args) -> np.ndarray:
    i, raw = args
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return extract_frame_bgr(img).astype(np.float32)


def extract_one(name: str, procs: int = 8) -> str:
    t = pq.read_table(f"{DATA}/{name}.parquet")
    ids = t.column("frame_id").to_pylist()
    hand = np.asarray(t.column("hand_count").to_pylist(), dtype=np.int8)
    work = np.asarray(
        [1 if s == "yes" else 0 for s in t.column("active_labor").to_pylist()],
        dtype=np.int8)
    blobs = [c["bytes"] for c in t.column("image").to_pylist()]
    n = len(ids)
    # single-core timing on a 20-frame sample
    t0 = time.time()
    for b in blobs[:20]:
        _feat_of((0, b))
    sample_ms = (time.time() - t0) / 20 * 1000.0
    t0 = time.time()
    with Pool(min(procs, 8)) as pool:
        X = np.stack(list(pool.imap(_feat_of, zip(range(n), blobs), chunksize=16)))
    wall = time.time() - t0
    path = f"{OUT}/{name}.npz"
    np.savez_compressed(path, X=X.astype(np.float32),
                        ids=np.asarray(ids),
                        hand=hand, work=work,
                        features=np.asarray(FEATURES),
                        sample_ms_single=np.float64(sample_ms),
                        wall_s=np.float64(wall), procs=np.int64(min(procs, 8)))
    print(f"{name}: n={n} d={X.shape[1]} sample_ms={sample_ms:.1f} "
          f"wall_s={wall:.0f} ({n / wall:.0f} frames/s) -> {path}", flush=True)
    return path


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "egocentric_10k"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    targets = NAMES if which == "all" else [which]
    for name in targets:
        extract_one(name, procs)
