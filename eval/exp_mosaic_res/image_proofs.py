"""Commit-quality image proofs for claim X3 (docs/experiments/exp_mosaic_res/images).

Builds one deterministic sample mosaic per grid x tile-resolution cell (all with
top-right per-tile timestamps) plus the equal-canvas duel pair, saves each as a
<200 KB PNG with its JSON sidecar (exact per-tile `t` values).

Regenerate: python3 eval/exp_mosaic_res/image_proofs.py   (CPU only, no GPU)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from jadal.mosaic import build as B  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(ROOT, "docs", "experiments", "exp_mosaic_res", "images")
WINDOW_S = 10.0
MAX_BYTES = 200 * 1024

CELLS = [
    # (name, rows, cols, tile_px)
    ("grid3x3_tile224px", 3, 3, 224),
    ("grid3x3_tile160px", 3, 3, 160),
    ("grid3x3_tile112px", 3, 3, 112),
    ("grid3x3_tile96px", 3, 3, 96),
    ("grid3x3_tile64px", 3, 3, 64),
    ("grid2x2_tile112px", 2, 2, 112),
    ("grid4x4_tile112px", 4, 4, 112),
    ("grid5x5_tile112px", 5, 5, 112),
    ("grid6x6_tile112px", 6, 6, 112),
    ("duel_1080p_3x3_tile224px", 3, 3, 224),
    ("duel_480p_4x4_tile168px", 4, 4, 168),
]


def pick_window():
    man = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    for it in man["items"]:
        if int(it["label"]) == 1 and it["id"] in ("g001", "g002", "g003"):
            return it
    return man["items"][1]


def decode(path, start_s, max_frames):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr[..., ::-1].copy())
    cap.release()
    return frames


def save_under_budget(img, png_path, budget=MAX_BYTES):
    """Try PNG compress_level=9, then adaptive-palette quantise until < budget."""
    for level in (9, 6):
        img.save(png_path, format="PNG", optimize=True, compress_level=level)
        if os.path.getsize(png_path) < budget:
            return "png" + str(level), os.path.getsize(png_path)
    for colors in (128, 96, 64):
        q = img.convert("P", palette=Image.ADAPTIVE, colors=colors, dither=Image.FLOYDSTEINBERG)
        q.save(png_path, format="PNG", optimize=True, compress_level=9)
        if os.path.getsize(png_path) < budget:
            return f"png_quant{colors}", os.path.getsize(png_path)
    return "png_quant64", os.path.getsize(png_path)


def main():
    os.makedirs(OUT, exist_ok=True)
    it = pick_window()
    native = 10.0
    frames = decode(it["path"], float(it["start_s"]), int(WINDOW_S * native))
    print(f"sample window {it['id']} clip={it['clip']} y={it['label']} "
          f"frames={len(frames)} src={it['path'].split('/')[-1]}", flush=True)
    index = []
    for (name, r, c, tp) in CELLS:
        img, sc = B.build_window(it["id"], "indego", frames, native,
                                 n_tiles=r * c, tile_px=tp, timestamps=True)
        png = os.path.join(OUT, name + ".png")
        mode, size = save_under_budget(img, png)
        js = os.path.join(OUT, name + ".json")
        # sidecar already carries per-tile t_s; add view provenance
        sc = dict(sc, view={"rows": r, "cols": c, "tile_px": tp,
                            "window_id": it["id"], "clip": it["clip"],
                            "label": it["label"], "sample_rule":
                            "t_i=(i+0.5)*10/N; src_frame_index=round(t_i*native_fps)",
                            "timestamps": "top-right, FONT_SIZE=12px"})
        json.dump(sc, open(js, "w"), indent=1, sort_keys=True)
        canvas = canvas_px = img.size[0] * img.size[1]
        index.append({"name": name, "rows": r, "cols": c, "tile_px": tp,
                      "frames": r * c, "canvas_px": canvas_px,
                      "png_bytes": size, "encode_mode": mode,
                      "png": f"images/{name}.png", "json": f"images/{name}.json",
                      "t_s": [t["t_s"] for t in sc["tiles"]]})
        print(f"  {name:26s} {img.size[0]}x{img.size[1]} {size/1024:6.1f} KB "
              f"({mode})", flush=True)
    json.dump({"sample_window": it["id"], "clip": it["clip"], "label": it["label"],
               "source": it["path"], "cells": index},
              open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
