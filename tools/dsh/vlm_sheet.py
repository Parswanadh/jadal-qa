"""Mosaic contact sheets FOR A VISION-LANGUAGE MODEL (V-track).

This is NOT `jadal/mosaic/build.py` and does not replace it.  That module builds the
small evidence mosaic scored by our own features (GRID.01-06); its burned 12 px
`t=3.5s` label is too small for a model to rely on.  This module builds the sheet
that a VLM can actually read, following the design the builder verified by hand
(`dsh_pack/seed_vlm_mosaic/mk.py`):

  * default 12 tiles in a 4x3 grid, 426x240 px per tile, row-major time order;
  * a LARGE bold yellow-on-black label in the top-left of every tile holding the
    tile number AND the time, e.g. ``#7  00:05.4s`` -- so the model can say what
    happened when;
  * times are cell-centred over the covered span, and ``t0_s`` shifts every label
    into CLIP-ABSOLUTE time (needed for the 30 s task slices of GRID.V05);
  * a JSON sidecar with the tile table and the PNG sha256.

Ablation switches required by GRID.V03 are first-class options, never forks:
``labels=False`` (no timestamps), ``order="shuffled"`` (tiles permuted but labels
travel with their tile), ``n_tiles``/``cols`` (6 or 24 tiles), and ``style`` for the
label format.

Deterministic: fixed font, fixed encoder parameters, no sampling.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SHEET_VERSION = "vlm-sheet-1.0.0"

TILE_W, TILE_H = 426, 240
COLS, ROWS = 4, 3
GUTTER = 4
BG = (60, 60, 60)
LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 255, 0)
FONT_SIZE = 26
FONT_PATH = os.environ.get(
    "JADAL_SHEET_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
LABEL_BOX = (235, 34)          # px, matches the verified seed sheet


def label_text(idx: int, t_s: float) -> str:
    """``#N  MM:SS.s`` -- 1-based tile number plus the time of that tile."""
    return f"#{idx + 1}  {int(t_s // 60):02d}:{t_s % 60:04.1f}s"


def font_file() -> str:
    return FONT_PATH if os.path.exists(FONT_PATH) else "<PIL default>"


def _load_font():
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    return ImageFont.load_default(size=FONT_SIZE)


def cell_times(n_tiles: int, span_s: float, t0_s: float = 0.0):
    """Cell-centred sample times over the span, offset into clip-absolute time."""
    return [t0_s + (i + 0.5) * span_s / n_tiles for i in range(n_tiles)]


def frame_indices(times_s, native_fps: float, n_frames: int, t0_s: float = 0.0):
    """Nearest decoded frame index for each absolute time, clipped to the buffer."""
    out = []
    for t in times_s:
        idx = int(round((t - t0_s) * native_fps))
        out.append(max(0, min(idx, n_frames - 1)))
    return out


def build_sheet(frames, times_s, *, cols: int = COLS, tile_w: int = TILE_W,
                tile_h: int = TILE_H, labels: bool = True, order=None,
                font=None):
    """Stitch tiles into one RGB sheet.

    ``frames``  : list of HxWx3 uint8 RGB arrays, time order, length == n_tiles
    ``times_s`` : the absolute time shown on each tile's label
    ``order``   : optional placement permutation (``order[slot] = tile index``);
                  the label travels with its tile.
    Returns ``(PIL.Image, tiles_meta)``.
    """
    n = len(frames)
    if n != len(times_s):
        raise ValueError("frames and times_s must have the same length")
    rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w + (cols + 1) * GUTTER,
                               rows * tile_h + (rows + 1) * GUTTER), BG)
    font = font or _load_font()
    place = list(range(n)) if order is None else list(order)
    if sorted(place) != list(range(n)):
        raise ValueError("order must be a permutation of 0..n-1")
    meta = []
    for slot, ti in enumerate(place):
        r, c = divmod(slot, cols)
        x = GUTTER + c * (tile_w + GUTTER)
        y = GUTTER + r * (tile_h + GUTTER)
        img = Image.fromarray(np.asarray(frames[ti], np.uint8), "RGB").resize(
            (tile_w, tile_h), Image.Resampling.LANCZOS)
        if labels:
            d = ImageDraw.Draw(img)
            text = label_text(ti, float(times_s[ti]))
            bw, bh = LABEL_BOX
            d.rectangle([0, 0, bw, bh], fill=LABEL_BG)
            d.text((6, 2), text, font=font, fill=LABEL_FG)
        else:
            text = None
        canvas.paste(img, (x, y))
        meta.append({"index": int(ti), "slot": int(slot), "row": int(r), "col": int(c),
                     "t_s": round(float(times_s[ti]), 3), "label": text,
                     "x": x, "y": y, "w": tile_w, "h": tile_h})
    return canvas, meta


def png_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def save_sheet(img, meta, png_path: str, sidecar_path: str | None = None,
               *, extra: dict | None = None) -> dict:
    """Write the PNG and its sidecar (with the sha256 of the PNG bytes)."""
    data = png_bytes(img)
    sha = hashlib.sha256(data).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    with open(png_path, "wb") as f:
        f.write(data)
    doc = {"sheet_version": SHEET_VERSION, "png_sha256": sha,
           "size": list(img.size), "tiles": meta, "font": {"file": font_file(),
                                                           "size": FONT_SIZE}}
    if extra:
        doc.update(extra)
    if sidecar_path:
        with open(sidecar_path, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
    return doc


def sheet_from_window(frames, *, start_s: float = 0.0, span_s: float = 10.0,
                      native_fps: float = 10.0, n_tiles: int = 12, cols: int = COLS,
                      labels: bool = True, order=None, tile_w: int = TILE_W,
                      tile_h: int = TILE_H):
    """Convenience: pick ``n_tiles`` cell-centred frames from a decoded window."""
    times = cell_times(n_tiles, span_s, start_s)
    idx = frame_indices(times, native_fps, len(frames), start_s)
    use = [frames[k] for k in idx]
    return build_sheet(use, times, cols=cols, tile_w=tile_w, tile_h=tile_h,
                       labels=labels, order=order)
