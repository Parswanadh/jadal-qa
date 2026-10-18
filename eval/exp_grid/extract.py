#!/usr/bin/env python3
"""GRID feature extraction harness (stream S1).

Decodes each gold window ONCE and extracts the mosaic-native feature vector for
every requested config, so the ablation arms share the decode and are exactly
comparable.  Cache: ``eval/results/exp_grid/features_<tag>.json`` (small, in-repo,
no pixels).  Mosaic PNGs are NEVER written here; set ``--artifacts DIR`` to keep a
handful for eyeballing.

Configs (tag -> n tiles, tile px, timestamps, shuffled order):
  m3x3_t224          primary: N=9, 3x3, 224 px tiles, timestamps on, time order
  m3x3_t224_nots     same canvas, timestamps NOT drawn (leak control)
  m3x3_t224_shuf     timestamps on, tiles placed in shuffled order (structure control)

Run:
  JADAL_BUNDLE=<bundle> python3 eval/exp_grid/extract.py --limit 3      # smoke
  JADAL_BUNDLE=<bundle> python3 eval/exp_grid/extract.py                # all 150
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))

import common as C  # noqa: E402
from jadal.mosaic import build as B  # noqa: E402
from jadal.mosaic import features as F  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results", "exp_grid")

CONFIGS = [
    # primary: the timestamps ARE drawn and the top-right box IS masked out
    {"tag": "m3x3_t224", "n": 9, "grid": (3, 3), "tile_px": 224,
     "timestamps": True, "shuffle_seed": None, "mask_timestamp": True},
    # content reference: nothing drawn, nothing masked -> the clean-tile features
    {"tag": "m3x3_t224_nots", "n": 9, "grid": (3, 3), "tile_px": 224,
     "timestamps": False, "shuffle_seed": None, "mask_timestamp": False},
    # LEAK control: timestamps drawn and NOT masked -> the text is inside the features
    {"tag": "m3x3_t224_leak", "n": 9, "grid": (3, 3), "tile_px": 224,
     "timestamps": True, "shuffle_seed": None, "mask_timestamp": False},
    # arrangement control: pooling is order-invariant, so this is expected to be a
    # no-op for these features; it is reported as the "does tile layout matter" answer
    {"tag": "m3x3_t224_shuf", "n": 9, "grid": (3, 3), "tile_px": 224,
     "timestamps": True, "shuffle_seed": 12345, "mask_timestamp": True},
    # --- GRID.04 compact post-kill-gate control ladder (brief: "beyond a small control") ---
    {"tag": "m2x2_t224", "n": 4, "grid": (2, 2), "tile_px": 224,
     "timestamps": True, "shuffle_seed": None, "mask_timestamp": True},
    {"tag": "m4x3_t224", "n": 12, "grid": (3, 4), "tile_px": 224,
     "timestamps": True, "shuffle_seed": None, "mask_timestamp": True},   # headline config
    {"tag": "m4x4_t224", "n": 16, "grid": (4, 4), "tile_px": 224,
     "timestamps": True, "shuffle_seed": None, "mask_timestamp": True},
    {"tag": "m4x3_t160", "n": 12, "grid": (3, 4), "tile_px": 160,
     "timestamps": True, "shuffle_seed": None, "mask_timestamp": True},
    {"tag": "m4x3_t112", "n": 12, "grid": (3, 4), "tile_px": 112,
     "timestamps": True, "shuffle_seed": None, "mask_timestamp": True},
]


def decode_window(path: str, start_s: float, max_frames: int = 200):
    """Decode from start_s up to max_frames RGB frames (the pod's mosaic decode)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr[..., ::-1].copy())          # BGR -> RGB
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"no frames {path}@{start_s}")
    return frames


def native_fps(path: str) -> float:
    import cv2
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS) or 10.0
    c.release()
    return float(fps)


def features_for(frames, cfg, window_id, source, fps):
    """Build the mosaic and return (feature dict, sidecar, canvas-or-None)."""
    grid = tuple(cfg["grid"])
    times = B.tile_times(cfg["n"])
    idx = B.frame_indices(times, fps, len(frames))
    order = None
    if cfg.get("shuffle_seed") is not None:
        order = list(np.random.default_rng(cfg["shuffle_seed"]).permutation(cfg["n"]))
    use = [frames[k] for k in idx]
    img, side = B.build_mosaic(use, times, idx, grid, cfg["tile_px"],
                               window_id=window_id, source=source, fps_used=cfg["n"] / B.WINDOW_S,
                               timestamps=cfg["timestamps"], order=order)
    vec = F.feature_vector(np.asarray(img), side, diff_frames=use,
                           mask_timestamp=bool(cfg.get("mask_timestamp", True)))
    return vec, side, np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", default="", help="comma-separated tags")
    ap.add_argument("--artifacts", default="", help="optional dir for PNGs (a FEW only)")
    ap.add_argument("--artifacts-every", type=int, default=50)
    args = ap.parse_args()

    cfgs = [c for c in CONFIGS if not args.configs or c["tag"] in args.configs.split(",")]
    if not cfgs:
        raise SystemExit("no configs selected")

    g = C.load_gold()
    idxs = list(range(len(g["ids"])))
    if args.limit:
        idxs = idxs[:args.limit]

    fps_cache = {}
    store = {c["tag"]: {} for c in cfgs}
    meta = {c["tag"]: {"config": c, "builder_version": B.BUILDER_VERSION,
                       "feature_version": F.FEATURE_VERSION,
                       "feature_names": F.feature_names()} for c in cfgs}
    t0 = time.time()
    per_window_ms = []
    for j, i in enumerate(idxs):
        path, start = g["path"][i], float(g["start_s"][i])
        fps = fps_cache.setdefault(path, native_fps(path))
        tw = time.time()
        frames = decode_window(path, start)
        for c in cfgs:
            vec, side, canvas = features_for(frames, c, g["ids"][i], "indego", fps)
            store[c["tag"]][g["ids"][i]] = {k: (None if not np.isfinite(v) else round(v, 8))
                                            for k, v in vec.items()}
            if args.artifacts and j % args.artifacts_every == 0:
                os.makedirs(args.artifacts, exist_ok=True)
                p = os.path.join(args.artifacts, f"{c['tag']}_{g['ids'][i]}.png")
                from PIL import Image
                Image.fromarray(canvas).save(p)
        per_window_ms.append(1000.0 * (time.time() - tw) / len(cfgs))
        if (j + 1) % 25 == 0:
            print(f"{j+1}/{len(idxs)} windows, {np.mean(per_window_ms[-25:]):.0f} ms/window/config",
                  flush=True)

    os.makedirs(OUT, exist_ok=True)
    for c in cfgs:
        tag = c["tag"]
        names = F.feature_names()
        digests = {}
        for wid, vec in store[tag].items():
            digests[wid] = F.feature_digest({k: (np.nan if v is None else v)
                                             for k, v in vec.items()})
        payload = {
            "tag": tag, "config": c, "n_windows": len(store[tag]),
            "feature_names": names, "builder_version": B.BUILDER_VERSION,
            "feature_version": F.FEATURE_VERSION,
            "git_sha": C.git_sha(),
            "feature_digest_sha256": hashlib.sha256(
                json.dumps(digests, sort_keys=True).encode()).hexdigest(),
            "per_window_digest": digests,
            "windows": store[tag],
            "ms_per_window_per_config": round(float(np.mean(per_window_ms)), 1),
            "wall_s": round(time.time() - t0, 1),
            "command": "JADAL_BUNDLE=<bundle> python3 eval/exp_grid/extract.py",
            "device": C.DEVICE_CPU,
        }
        p = os.path.join(OUT, f"features_{tag}.json")
        with open(p, "w") as f:
            json.dump(payload, f, sort_keys=True)
        print(f"wrote {p}  ({len(store[tag])} windows, {np.mean(per_window_ms):.0f} ms/window/config)")


if __name__ == "__main__":
    main()
