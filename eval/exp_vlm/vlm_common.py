"""Shared helpers for the GRID.V track (VLM contact sheets).

Deterministic window selection, decode, and sheet building on top of
`tools/dsh/vlm_sheet.py`.  The window subsets are FIXED here (seeded) so that the
pre-registered experiment sizes cannot drift between runs, and so two experiments
can share windows without re-deriving them differently.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))
sys.path.insert(0, ROOT)

import common as C  # noqa: E402
import vlm_sheet as S  # noqa: E402

SHEET_DIR = os.environ.get("JADAL_SHEET_DIR", "/tmp/opencode/dsh/sheets")
OOD_DIRS = ["/home/parshu/projects/jadal-qa-wt/ood_labels/video",
            "/home/parshu/projects/jadal-qa-wt/ood_labels/video2"]


def fixed_subset(items, n: int, seed: int = 0):
    """Deterministic size-n subset (sorted-key permutation, stable across machines)."""
    if n >= len(items):
        return list(items)
    order = np.random.default_rng(seed).permutation(len(items))
    return [items[i] for i in sorted(order[:n])]


def stratified_subset(rows, n: int, seed: int = 0, key="y"):
    """n rows balanced on ``key`` (half bad, half good where available).

    AUROC is prevalence-independent, so balancing buys discrimination power per
    API call; say in any report that the subset is balanced and therefore NOT
    representative of the base rate.
    """
    bad = [r for r in rows if r[key] == 0]
    good = [r for r in rows if r[key] == 1]
    nb = min(len(bad), n // 2)
    ng = min(len(good), n - nb)
    out = fixed_subset(bad, nb, seed) + fixed_subset(good, ng, seed + 1)
    return sorted(out, key=lambda r: r.get("window") or r.get("segment_id"))


def ood_windows(n: int | None = None, seed: int = 0):
    """The 160 labelled Egocentric-10K windows, optionally a fixed subset."""
    o = C.load_ood160()
    rows = []
    for wid, seg, y, fac, rsn, p26 in zip(o["ids"], o["segment_id"], o["y"],
                                          o["factory"], o["reason"], o["p26"]):
        path = None
        for d in OOD_DIRS:
            hits = sorted(glob.glob(os.path.join(d, seg + "*.mp4")))
            if hits:
                path = hits[0]
                break
        rows.append({"window": wid, "segment_id": seg, "path": path, "y": int(y),
                     "factory": fac, "reason": rsn, "p26": float(p26)})
    return fixed_subset(rows, n, seed) if n else rows


def indego_windows(n: int | None = None, seed: int = 0):
    g = C.load_gold()
    rows = [{"window": g["ids"][i], "path": g["path"][i], "start_s": float(g["start_s"][i]),
             "clip": g["clip"][i], "y": int(g["y"][i]), "reason": g["reason"][i]}
            for i in range(len(g["ids"]))]
    return fixed_subset(rows, n, seed) if n else rows


def decode_window(path: str, start_s: float = 0.0, max_frames: int = 1200):
    """Decode from ``start_s`` up to ``max_frames`` RGB frames.  Returns (frames, fps)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr[..., ::-1].copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames {path}@{start_s}")
    return frames, fps


def sheet_for(path: str, start_s: float, span_s: float, *, n_tiles: int = 12,
              cols: int = 4, labels: bool = True, shuffle_seed=None,
              t0_s: float | None = None, cache_key: str | None = None):
    """Decode + build one sheet.  Returns (png_bytes, sidecar_doc)."""
    frames, fps = decode_window(path, start_s, max_frames=int(span_s * 30) + 30)
    times = S.cell_times(n_tiles, span_s, start_s if t0_s is None else t0_s)
    idx = S.frame_indices(times, fps, len(frames), start_s)
    use = [frames[k] for k in idx]
    order = None
    if shuffle_seed is not None:
        order = list(np.random.default_rng(shuffle_seed).permutation(n_tiles))
    img, meta = S.build_sheet(use, times, cols=cols, labels=labels, order=order)
    png = S.png_bytes(img)
    import hashlib
    doc = {"sheet_version": S.SHEET_VERSION,
           "png_sha256": hashlib.sha256(png).hexdigest(),
           "size": list(img.size), "tiles": meta, "n_tiles": n_tiles, "cols": cols,
           "labels": labels, "shuffle_seed": shuffle_seed,
           "start_s": start_s, "span_s": span_s,
           "source_frame_indices": idx, "native_fps": fps,
           "font": {"file": S.font_file(), "size": S.FONT_SIZE},
           "cache_key": cache_key}
    return png, doc


def save_ledger_row(path: str, row: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
