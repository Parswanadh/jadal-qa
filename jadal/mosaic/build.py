"""E2 mosaic builder — deterministic "N timestamped frames in ONE image" input.

Spec (skill `jadal-mosaic-spec`):
  * N in {4,6,9,12,16}; grids 2x2, 2x3, 3x3, 3x4, 4x4; row-major time order
    (top-left earliest).
  * Tile px in {112,160,224}; INTER_AREA area interpolation; aspect preserved,
    letterboxed with mid-grey.
  * 2 px mid-grey gutter between/around tiles.
  * Timestamp ``t=3.5s`` white on a dark semi-transparent box in the TOP-RIGHT
    of EVERY tile; fixed font + size.
  * Uniform times over the window (cell-centred index rule, same rule for all
    datasets): ``t_i = (i + 0.5) * window_s / N``.
  * JSON sidecar with the exact keys below; byte-identical PNG across repeat
    builds.

Everything here is deterministic (fixed encoder params, fixed font, fixed
layout); no sampling at build time.  Path resolution uses ``JADAL_BUNDLE`` so
the repo can be open-sourced without pod-specific paths.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BUILDER_VERSION = "mosaic-1.0.0"

WINDOW_S = 10.0
GUTTER = 2                      # px, mid-grey, between and around tiles
BG = (128, 128, 128)            # mid-grey letterbox / gutter background
TS_BOX_ALPHA = 140              # dark box opacity (0-255)
TS_TEXT = (255, 255, 255)
FONT_SIZE = 12                  # fixed size, readable at 112 px tiles
FONT_PATH = os.environ.get(
    "JADAL_MOSAIC_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

#: canonical spec grids; N -> (rows, cols)
SPEC_GRIDS = {4: (2, 2), 6: (2, 3), 9: (3, 3), 12: (3, 4), 16: (4, 4)}


def grid_for_n(n: int):
    """Spec grid for the canonical N, else the exact factorisation closest to square."""
    if n in SPEC_GRIDS:
        return SPEC_GRIDS[n]
    if n < 1:
        raise ValueError("n must be >= 1")
    best = (1, n)
    for r in range(1, int(np.sqrt(n)) + 1):
        if n % r == 0:
            best = (r, n // r)
    return best


def tile_times(n: int, window_s: float = WINDOW_S):
    """Cell-centred uniform times over the window (documented index rule)."""
    return [(i + 0.5) * window_s / n for i in range(n)]


def frame_indices(times, native_fps: float, n_frames: int):
    """Nearest native frame index for each time, clipped to the decoded range."""
    out = []
    for t in times:
        idx = int(round(t * native_fps))
        out.append(max(0, min(idx, n_frames - 1)))
    return out


def _resample_area():
    """PIL INTER_AREA (=3); this Pillow build drops the named constant."""
    return getattr(Image.Resampling, "INTER_AREA", Image.Resampling(3))


def font_file() -> str:
    """Path of the font actually used (or a sentinel for the PIL default).

    Recorded in every sidecar so a PNG-hash mismatch across environments is
    explainable: identical pixels require the identical ``(file, size)``.
    """
    if os.path.exists(FONT_PATH):
        return FONT_PATH
    return "PIL:load_default"


def _load_font():
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    return ImageFont.load_default(size=FONT_SIZE)


def ts_text(t_s: float) -> str:
    """Canonical timestamp string burned into each tile: ``t=X.Xs``."""
    return f"t={float(t_s):.1f}s"


def letterbox_tile(rgb: np.ndarray, tile_px: int) -> Image.Image:
    """Resize an RGB frame to fit ``tile_px`` x ``tile_px``, preserving aspect.

    INTER_AREA, padded to square with mid-grey (letterbox)."""
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    w, h = img.size
    scale = min(tile_px / w, tile_px / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), _resample_area())
    canvas = Image.new("RGB", (tile_px, tile_px), BG)
    canvas.paste(img, ((tile_px - nw) // 2, (tile_px - nh) // 2))
    return canvas


def _draw_timestamp(tile: Image.Image, t_s: float, font) -> None:
    """White ``t=X.Xs`` on a dark semi-transparent box, top-right of the tile."""
    text = ts_text(t_s)
    overlay = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    pad, m = 2, 2
    box_w, box_h = tw + 2 * pad, th + 2 * pad
    bx = tile.size[0] - m - box_w
    by = m
    d.rectangle([bx, by, bx + box_w, by + box_h], fill=(0, 0, 0, TS_BOX_ALPHA))
    d.text((bx + pad - x0, by + pad - y0), text, font=font, fill=TS_TEXT)
    composed = Image.alpha_composite(tile.convert("RGBA"), overlay).convert("RGB")
    tile.paste(composed, (0, 0))


def build_mosaic(frames, times_s, src_frame_indices, grid, tile_px, *,
                 window_id: str, source: str, fps_used: float,
                 timestamps: bool = True, order=None):
    """Build the mosaic image and its sidecar dict.

    ``frames``  : list of HxWx3 uint8 RGB arrays in time order (len == N)
    ``order``   : optional placement permutation (``order[slot] = tile index``);
                  timestamps travel with their tile.
    Returns ``(PIL.Image, sidecar)``; PNG bytes are hashed into
    ``sidecar['mosaic_sha256']`` by :func:`save_mosaic`.
    """
    rows, cols = grid
    n = rows * cols
    if len(frames) != n:
        raise ValueError(f"grid {grid} needs {n} frames, got {len(frames)}")
    if len(times_s) != n or len(src_frame_indices) != n:
        raise ValueError("times_s / src_frame_indices must match grid")

    canvas_w = cols * tile_px + (cols + 1) * GUTTER
    canvas_h = rows * tile_px + (rows + 1) * GUTTER
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    font = _load_font()

    place = list(range(n)) if order is None else list(order)
    if sorted(place) != list(range(n)):
        raise ValueError("order must be a permutation of 0..N-1")

    tiles = []
    for slot, ti in enumerate(place):
        r, c = divmod(slot, cols)
        x = GUTTER + c * (tile_px + GUTTER)
        y = GUTTER + r * (tile_px + GUTTER)
        tile = letterbox_tile(frames[ti], tile_px)
        if timestamps:
            _draw_timestamp(tile, float(times_s[ti]), font)
        canvas.paste(tile, (x, y))
        tiles.append({"index": int(ti), "t_s": round(float(times_s[ti]), 3),
                      "src_frame_index": int(src_frame_indices[ti]),
                      "x": x, "y": y, "w": tile_px, "h": tile_px})

    sidecar = {
        "window_id": window_id,
        "source": source,
        "grid": [int(rows), int(cols)],
        "tile_px": int(tile_px),
        "fps_used": float(fps_used),
        "tiles": tiles,
        "mosaic_sha256": "",
        "builder_version": BUILDER_VERSION,
        "font": {"file": font_file(), "size": FONT_SIZE},
    }
    return canvas, sidecar


def png_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def save_mosaic(img: Image.Image, sidecar: dict, png_path: str, json_path: str):
    """Write the PNG (byte-identical across builds) and the hashed sidecar."""
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    data = png_bytes(img)
    sha = hashlib.sha256(data).hexdigest()
    with open(png_path, "wb") as f:
        f.write(data)
    sidecar = dict(sidecar)
    sidecar["mosaic_sha256"] = sha
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=1, sort_keys=True)
    return sidecar


def build_window(window_id: str, source: str, frames, native_fps: float, *,
                 n_tiles: int, tile_px: int, window_s: float = WINDOW_S,
                 timestamps: bool = True, shuffle_seed=None, motion=False):
    """Convenience wrapper: pick indices from the cell-centred rule and build.

    ``frames`` is the decoded native-rate list (RGB uint8).  ``motion=True``
    uses ``|I_t - I_{t-1}|`` tiles.  Returns ``(PIL.Image, sidecar)``.
    """
    n = n_tiles
    grid = grid_for_n(n)
    times = tile_times(n, window_s)
    idx = frame_indices(times, native_fps, len(frames))
    order = None
    if shuffle_seed is not None:
        order = list(np.random.default_rng(shuffle_seed).permutation(n))
    if motion:
        use = []
        for k in idx:
            prev = max(0, k - 1)
            diff = np.abs(frames[k].astype(np.int16) - frames[prev].astype(np.int16))
            use.append(diff.astype(np.uint8))
    else:
        use = [frames[k] for k in idx]
    fps_used = n / window_s
    return build_mosaic(use, times, idx, grid, tile_px, window_id=window_id,
                        source=source, fps_used=fps_used,
                        timestamps=timestamps, order=order)
