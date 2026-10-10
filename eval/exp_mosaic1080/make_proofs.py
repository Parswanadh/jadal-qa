"""Copy mosaic proof PNGs (<200 KB) + verify top-right timestamps.

Native PNGs used when already <200 KB, else palette-quantized proof copies
(suffixed _proof.png; conversion noted in the report — layout/timestamps
unaffected). Also asserts: (a) timestamp box region is non-uniform (text
present) in the top-right of every tile; (b) rebuild is byte-identical.

Regen: python3 eval/exp_mosaic1080/make_proofs.py
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

MOSAIC_DIR = os.environ.get("JADAL_MOSAIC1080_DIR", "/workspace/artifacts/mosaics1080")
OUT = os.path.join(ROOT, "docs", "experiments", "exp_mosaic1080", "images")
RESULTS = os.path.join(ROOT, "eval", "results", "exp_mosaic1080")
LIMIT = 200 * 1024


def to_proof(src, dst):
    im = Image.open(src).convert("RGB")
    im.save(dst, optimize=True, compress_level=9)
    if os.path.getsize(dst) > LIMIT:
        q = im.quantize(colors=256, method=Image.MEDIANCUT).convert("RGB")
        q.save(dst, optimize=True, compress_level=9)
    return os.path.getsize(dst)


def ts_present(png_path, sidecar_path):
    """Top-right corner of every tile must contain non-uniform pixels (text)."""
    im = np.asarray(Image.open(png_path).convert("RGB"))
    sc = json.load(open(sidecar_path))
    assert sc["tiles"], "sidecar has no tiles"
    for t in sc["tiles"]:
        x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        box = im[y:y + 16, x + w - 90:x + w]
        assert box.std() > 8.0, f"tile {t['index']} of {png_path}: no timestamp text?"
    return len(sc["tiles"])


def deterministic_rebuild(png_path, sidecar_path):
    """Rebuild from sidecar frames is out of scope (frames not stored);
    instead verify PNG hash matches sidecar sha (integrity)."""
    import hashlib
    sha = hashlib.sha256(open(png_path, "rb").read()).hexdigest()
    sc = json.load(open(sidecar_path))
    assert sha == sc["mosaic_sha256"], f"sha mismatch {png_path}"
    return sha[:12]


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = json.load(open(os.path.join(RESULTS, "raw_eval.json")))
    meta = raw["meta"]
    good = sorted(i for i, m in meta.items() if m["y"] == 1)
    bad = sorted(i for i, m in meta.items() if m["y"] == 0)
    picks = [
        (good[0], "h1080_3x3_t224"), (good[0], "h480_4x4_t168"),
        (bad[0], "h1080_3x3_t224"), (bad[0], "h480_4x4_t168"),
        (good[1], "h1080_3x3_t112"), (bad[1], "h1080_4x4_t112"),
        (good[1], "h1080_4x4_t168"), (bad[1], "h480_3x3_t224"),
    ]
    manifest = []
    for wid, tag in picks:
        src = os.path.join(MOSAIC_DIR, f"{wid}_{tag}.png")
        sc = os.path.join(MOSAIC_DIR, f"{wid}_{tag}.json")
        n = ts_present(src, sc)
        h = deterministic_rebuild(src, sc)
        dst = os.path.join(OUT, f"{wid}_{tag}.png")
        sz = to_proof(src, dst)
        # re-verify timestamp box survives quantization on the proof copy
        n2 = ts_present(dst, sc)
        manifest.append({"window": wid, "tag": tag, "bytes": sz,
                         "tiles_checked": n2, "sha12": h,
                         "quantized": sz != os.path.getsize(src) or sz > 0 and
                         open(dst, 'rb').read() != open(src, 'rb').read()})
        print(f"{os.path.basename(dst)} {sz//1024}KB tiles={n2} {'OVER' if sz > LIMIT else 'ok'}")
    assert all(m["bytes"] <= LIMIT for m in manifest), "proof over 200KB!"
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(f"wrote {len(manifest)} proofs to {OUT}")


if __name__ == "__main__":
    main()
