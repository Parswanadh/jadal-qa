#!/usr/bin/env python3
"""GRID.06 feature extraction: the SAME mosaic-native features on OOD-160.

Out-of-domain = 160 human-labelled Egocentric-10K windows (138 good / 22 bad,
47 factories, ONE labeller), 480p window clips under
``…/ood_labels/video{,2}/<segment_id>*.mp4``.  The frozen shipped-head scores for
the same windows come from the harden gate run and are loaded by
``common.load_ood160()``.

Writes ``eval/results/exp_grid/features_ood_m3x3_t224.json`` (feature vectors only,
no pixels).  The head is FROZEN afterwards: it is trained on the IndEgo windows
only and applied here unchanged.

Run: python3 eval/exp_grid/extract_ood.py [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))

import common as C  # noqa: E402
from eval.exp_grid.extract import CONFIGS, decode_window, features_for, native_fps  # noqa: E402

OOD_DIRS = ["/home/parshu/projects/jadal-qa-wt/ood_labels/video",
            "/home/parshu/projects/jadal-qa-wt/ood_labels/video2"]
OUT = os.path.join(ROOT, "eval", "results", "exp_grid")


def find_clip(seg_id: str):
    for d in OOD_DIRS:
        hits = sorted(glob.glob(os.path.join(d, seg_id + "*.mp4")))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default="m3x3_t224")
    args = ap.parse_args()

    cfg = next(c for c in CONFIGS if c["tag"] == args.config)
    ood = C.load_ood160()
    ids = ood["ids"] if not args.limit else ood["ids"][:args.limit]

    store, missing, fps_cache = {}, [], {}
    t0 = time.time()
    per_ms = []
    for j, wid, seg in zip(range(len(ids)), ids, ood["segment_id"]):
        path = find_clip(seg)
        if path is None:
            missing.append(wid)
            continue
        fps = fps_cache.setdefault(path, native_fps(path))
        tw = time.time()
        frames = decode_window(path, 0.0)
        vec, _side, _canvas = features_for(frames, cfg, wid, "ego10k", fps)
        store[wid] = {k: (None if not np.isfinite(v) else round(v, 8)) for k, v in vec.items()}
        per_ms.append(1000.0 * (time.time() - tw))
        if (j + 1) % 40 == 0:
            print(f"{j+1}/{len(ids)} windows, {np.mean(per_ms[-40:]):.0f} ms/window", flush=True)

    from jadal.mosaic import build as B
    from jadal.mosaic import features as F
    payload = {
        "tag": f"ood_{cfg['tag']}", "config": cfg, "n_windows": len(store),
        "missing": missing, "feature_names": F.feature_names(),
        "builder_version": B.BUILDER_VERSION, "feature_version": F.FEATURE_VERSION,
        "git_sha": C.git_sha(),
        "ms_per_window": round(float(np.mean(per_ms)), 1) if per_ms else None,
        "wall_s": round(time.time() - t0, 1),
        "source_clips": {w: find_clip(s) for w, s in zip(ids, ood["segment_id"])},
        "windows": store,
        "command": f"python3 eval/exp_grid/extract_ood.py --config {args.config}",
        "device": C.DEVICE_CPU,
        "note": "480p transcoded Egocentric-10K window clips (the same files the "
                "frozen shipped-head scores were produced on).",
    }
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"features_{payload['tag']}.json")
    with open(p, "w") as f:
        json.dump(payload, f, sort_keys=True)
    print(f"wrote {p} ({len(store)} windows, missing {len(missing)})")


if __name__ == "__main__":
    main()
