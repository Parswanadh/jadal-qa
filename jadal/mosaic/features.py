"""Mosaic-native features: score ONE composite image with OUR OWN extractors.

Why this module exists
----------------------
The shipped 26-number head is mostly temporal: 10 of its 26 features are measured
*along time at each pixel* (temp_b0..b3, frozen_frac, flicker_idx, motion_energy,
global_shift, dup_hash_mean, zncc_self, rppg_pulse).  A still mosaic has no time
axis, so those cannot be computed the same way and the shipped head cannot be
applied to a mosaic at all.  This module defines the replacement feature set that
*can* be computed from the canvas alone:

  * ``spatial``   — per-tile spatial statistics, pooled over tiles (mean/min/max/std):
                    luma percentiles, dark/bright fraction, radial spectrum slope,
                    high-frequency ratio, DCT energy, PCA residual, skin fraction,
                    grey-world illuminant ratios, brightness-calibrated Laplacian
                    variance (the blur statistic) and histogram entropy.
  * ``intertile`` — temporal *proxies* between consecutive tiles: mean absolute
                    difference, near-identical-pair share (freeze/duplicate proxy),
                    pHash and ZNCC similarity, phase-correlation shift (camera-motion
                    proxy), brightness variance and flicker across tiles.
  * ``diff``      — the same pooled statistics on the DIFFERENCE mosaic
                    ``|I_t - I_{t-1}|`` (the earlier EmbeddingGemma-2 test found the
                    motion mosaic the best in-domain input and the worst out of domain,
                    so it is kept as a separate, ablatable group).

Timestamp and letterbox masks
-----------------------------
The builder burns ``t=X.Xs`` into the top-right of every tile.  Our features are not
OCR, but a text box is high-frequency content that *would* leak into sharpness, DCT and
spectrum features, so a conservative top-right rectangle is masked out of every tile by
default (``mask_timestamp=True``).  GRID.04's timestamps-on/off arm measures whatever
residual leak the mask leaves.  Mid-grey letterbox padding (added by
``jadal.mosaic.build.letterbox_tile`` for non-square sources) is detected and excluded
so that a 16:9 factory frame and a square IndEgo frame are measured on their content.

Determinism
-----------
No randomness, no sampling, fixed float64 accumulation; ``feature_vector`` is
deterministic for a given canvas and sidecar, and ``feature_digest`` is a sha256 over
the quantised feature values (receipt-compatible: canonical JSON, sorted keys).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from jadal.mosaic import build as B

FEATURE_VERSION = "mosaic-features-1.0.0"

#: groups in the order they are concatenated
GROUPS = ("spatial", "intertile", "diff")

#: per-tile spatial statistics (pooled later)
SPATIAL_STATS = (
    "luma_p05", "luma_p50", "luma_p95", "luma_std", "dark_frac", "bright_frac",
    "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "skin_frac",
    "illum_r_ratio", "illum_b_ratio", "lapvar_cal", "entropy",
)

#: how per-tile statistics are pooled over tiles
POOLS = ("mean", "min", "max", "std")

#: direct inter-tile (temporal-proxy) features
INTERTILE_STATS = (
    "it_mad_mean", "it_mad_min", "it_mad_max", "it_frozen_frac", "it_dup_phash_frac",
    "it_zncc_mean", "it_zncc_min", "it_shift_mean", "it_shift_max", "it_bright_var",
    "it_flicker", "it_below005_frac",
)

#: statistics of the difference mosaic (pooled mean over tiles unless stated)
DIFF_STATS = (
    "dm_energy_mean", "dm_energy_max", "dm_luma_mean_mean", "dm_luma_mean_std",
    "dm_hf_ratio_mean", "dm_radial_slope_mean", "dm_mad_mean",
)

LUMA_W = np.array([0.299, 0.587, 0.114])
BG_GREY = 128.0


# ------------------------------------------------------------------ helpers --
def to_luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma in [0, 1], float64."""
    return (np.asarray(rgb, dtype=np.float64) @ LUMA_W) / 255.0


def _timestamp_box(tile_px: int):
    """Conservative (x0, y0, x1, y1) mask rectangle for the burned timestamp.

    build.py draws a ~51x21 px box (font size 12, pad 2, margin 2) at the tile's
    top-right corner regardless of tile size, so the mask must not shrink below
    that even for 112 px tiles.
    """
    w = int(min(max(64, tile_px * 0.32), tile_px))
    h = int(min(max(26, tile_px * 0.14), tile_px))
    return tile_px - w, 0, tile_px, h


def tile_mask(tile_px: int, *, mask_timestamp: bool = True) -> np.ndarray:
    """Boolean content mask (True = use) for one tile, before letterbox detection."""
    m = np.ones((tile_px, tile_px), bool)
    if mask_timestamp:
        x0, y0, x1, y1 = _timestamp_box(tile_px)
        m[y0:y1, x0:x1] = False
    return m


def letterbox_mask(tile: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """Drop rows/columns that are >=98 % mid-grey padding (build.letterbox_tile)."""
    near_bg = np.abs(np.asarray(tile, np.float64) - BG_GREY).max(axis=2) <= 2.0
    m = base_mask.copy()
    rows = near_bg.mean(axis=1) >= 0.98
    cols = near_bg.mean(axis=0) >= 0.98
    m[rows, :] = False
    m[:, cols] = False
    if m.sum() < 0.05 * m.size:          # never mask everything
        return base_mask
    return m


def tile_grid(canvas: np.ndarray, sidecar: dict):
    """Yield (tile_rgb, mask, t_s) per tile in TIME order using the sidecar offsets."""
    tiles = sorted(sidecar["tiles"], key=lambda t: t["index"])
    base = tile_mask(int(sidecar["tile_px"]))
    out = []
    for t in tiles:
        x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        rgb = np.asarray(canvas)[y:y + h, x:x + w]
        m = letterbox_mask(rgb, base)
        out.append((rgb, m, float(t["t_s"])))
    return out


# ------------------------------------------------------------ tile features --
def _fft_profile(luma: np.ndarray):
    s = min(luma.shape)
    a = luma[:s, :s] - float(luma[:s, :s].mean())
    win = np.hanning(s)[:, None] * np.hanning(s)[None, :]
    F = np.fft.fftshift(np.fft.fft2(a * win))
    P = np.abs(F) ** 2
    c = s // 2
    yy, xx = np.indices((s, s))
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2).astype(int)
    nb = s // 2
    prof = np.bincount(r.ravel(), P.ravel(), minlength=nb + 1)[:nb + 1]
    cnt = np.bincount(r.ravel(), minlength=nb + 1)[:nb + 1]
    prof = prof / np.maximum(cnt, 1)
    lo, hi = 2, max(3, nb // 2)
    rr = np.arange(lo, hi)
    if len(rr) >= 2 and prof[lo:hi].min() > 0:
        slope = float(np.polyfit(np.log(rr), np.log(prof[lo:hi]), 1)[0])
    else:
        slope = float("nan")
    hf = float(prof[hi:].sum() / max(prof[1:].sum(), 1e-12))
    return slope, hf


def _dct_energy(luma: np.ndarray) -> float:
    try:
        from scipy.fft import dctn
    except Exception:                                     # pragma: no cover
        return float("nan")
    h, w = (luma.shape[0] // 8) * 8, (luma.shape[1] // 8) * 8
    if h < 8 or w < 8:
        return float("nan")
    blocks = luma[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    D = dctn(blocks, axes=(1, 2), norm="ortho")
    return float(np.abs(D[:, 1:, 1:]).mean())


def _lapvar_cal(gray: np.ndarray) -> float:
    """Laplacian variance divided by mean brightness (the brightness-calibrated
    sharpness statistic, GRID.09): raw Laplacian variance scales with contrast, so
    a dark-but-sharp frame and a bright blurred frame are otherwise confusable."""
    g = gray.astype(np.float64)
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1])
    return float(lap.var() / (float(g.mean()) + 1e-6))


def _skin_frac(rgb: np.ndarray, mask: np.ndarray) -> float:
    r, g, b = (rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16),
               rgb[..., 2].astype(np.int16))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    skin = ((r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15) &
            (np.abs(r - g) > 15) & (r > g) & (r > b))
    return float(skin[mask].mean()) if mask.sum() else float("nan")


def spatial_features(rgb: np.ndarray, mask: np.ndarray, gray: np.ndarray) -> dict:
    """The 15 per-tile statistics (unpooled)."""
    v = gray[mask]
    if v.size < 16:
        return {k: float("nan") for k in SPATIAL_STATS}
    r = rgb[mask].astype(np.float64)
    gm = max(float(r[:, 1].mean()), 1e-6)
    px = r - r.mean(axis=0)
    cov = px.T @ px / max(len(px), 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    slope, hf = _fft_profile(np.where(mask, gray, float(gray[mask].mean())))
    hist, _ = np.histogram(v, bins=32, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    return {
        "luma_p05": float(np.percentile(v, 5)),
        "luma_p50": float(np.percentile(v, 50)),
        "luma_p95": float(np.percentile(v, 95)),
        "luma_std": float(v.std()),
        "dark_frac": float((v < 0.05).mean()),
        "bright_frac": float((v > 0.95).mean()),
        "radial_slope": slope,
        "hf_ratio": hf,
        "dct_energy": _dct_energy(np.where(mask, gray, float(gray[mask].mean()))),
        "pca_resid": float(1.0 - ev[0] / max(ev.sum(), 1e-12)),
        "skin_frac": _skin_frac(rgb, mask),
        "illum_r_ratio": float(r[:, 0].mean() / gm),
        "illum_b_ratio": float(r[:, 2].mean() / gm),
        "lapvar_cal": _lapvar_cal(np.where(mask, gray, float(gray[mask].mean()))),
        "entropy": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
    }


# --------------------------------------------------------- inter-tile block --
def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else 1.0


def _phase_shift(a: np.ndarray, b: np.ndarray) -> float:
    A = np.fft.rfft2(a - a.mean())
    B = np.fft.rfft2(b - b.mean())
    R = A * np.conj(B)
    R /= (np.abs(R) + 1e-12)
    c = np.fft.irfft2(R, s=a.shape)
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    if iy > a.shape[0] // 2:
        iy -= a.shape[0]
    if ix > a.shape[1] // 2:
        ix -= a.shape[1]
    return float(np.hypot(iy, ix))


def _phash(gray: np.ndarray, mask: np.ndarray, size: int = 32) -> np.ndarray:
    from PIL import Image
    g = np.where(mask, gray, float(gray[mask].mean())) if mask.sum() else gray
    img = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8), "L").resize(
        (size, size), Image.Resampling.BILINEAR)
    a = np.asarray(img, np.float64)
    from scipy.fft import dctn
    d = dctn(a, norm="ortho")[:8, :8]
    return (d > np.median(d[1:, 1:])).ravel()


def intertile_features(tiles) -> dict:
    """Temporal proxies between consecutive tiles (time order)."""
    if len(tiles) < 2:
        return {k: float("nan") for k in INTERTILE_STATS}
    lums, mads, znccs, shifts, means, hashes = [], [], [], [], [], []
    for rgb, m, _t in tiles:
        g = to_luma(rgb)
        lums.append((g, m))
        means.append(float(g[m].mean()))
        hashes.append(_phash(g, m))
    for i in range(len(lums) - 1):
        (g0, m0), (g1, m1) = lums[i], lums[i + 1]
        mm = m0 & m1
        if mm.sum() < 16:
            mm = m0 | m1
        d = np.abs(g0[mm] - g1[mm])
        mads.append(float(d.mean()))
        znccs.append(_zncc(g0[mm], g1[mm]))
        shifts.append(_phase_shift(np.where(mm, g0, 0.0), np.where(mm, g1, 0.0)))
    mads = np.asarray(mads)
    znccs = np.asarray(znccs)
    shifts = np.asarray(shifts)
    dup = np.mean([int(np.count_nonzero(hashes[i] != hashes[i + 1]) <= 2)
                   for i in range(len(hashes) - 1)])
    means = np.asarray(means)
    return {
        "it_mad_mean": float(mads.mean()),
        "it_mad_min": float(mads.min()),
        "it_mad_max": float(mads.max()),
        "it_frozen_frac": float((mads < 0.002).mean()),
        "it_dup_phash_frac": float(dup),
        "it_zncc_mean": float(znccs.mean()),
        "it_zncc_min": float(znccs.min()),
        "it_shift_mean": float(shifts.mean()),
        "it_shift_max": float(shifts.max()),
        "it_bright_var": float(means.var()),
        "it_flicker": float(np.abs(np.diff(means)).mean()) if len(means) > 1 else float("nan"),
        "it_below005_frac": float((mads < 0.005).mean()),
    }


def _nanmean(xs) -> float:
    """Mean that returns NaN instead of warning when every value is NaN
    (a perfectly static difference mosaic has no radial spectrum)."""
    a = np.asarray(xs, float)
    return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else float("nan")


def diff_mosaic(frames, tile_px: int):
    """|I_t - I_{t-1}| tiles, letterboxed exactly as the builder does.

    Returns a list of tile_px x tile_px uint8 RGB arrays in time order; the first
    tile is all-zero (no predecessor).  ``frames`` are the raw decoded frames.
    """
    out = []
    prev = None
    for f in frames:
        tile = np.asarray(B.letterbox_tile(f, tile_px), np.int16)
        out.append(np.zeros_like(tile, np.uint8) if prev is None
                   else np.abs(tile - prev).astype(np.uint8))
        prev = tile
    return out


def diff_features(frames, tile_px: int, *, mask_timestamp: bool = True) -> dict:
    """Pooled statistics of the difference mosaic (mean over tiles).

    ``frames`` are the raw decoded frames in time order; each is letterboxed to
    ``tile_px`` first, so the tile mask (timestamp + padding) applies.
    """
    base = tile_mask(tile_px, mask_timestamp=mask_timestamp)
    tiles = diff_mosaic(frames, tile_px)
    en, lm, hf, rs = [], [], [], []
    for t in tiles:
        m = letterbox_mask(t, base)
        if m.sum() < 16:
            m = base
        g = to_luma(t)
        en.append(float(np.abs(np.asarray(t, np.float64)).mean() / 255.0))
        lm.append(float(g[m].mean()))
        s, h = _fft_profile(np.where(m, g, float(g[m].mean())))
        rs.append(s)
        hf.append(h)
    lma = np.asarray(lm)
    return {
        "dm_energy_mean": float(np.mean(en)),
        "dm_energy_max": float(np.max(en)),
        "dm_luma_mean_mean": float(lma.mean()),
        "dm_luma_mean_std": float(lma.std()),
        "dm_hf_ratio_mean": _nanmean(hf),
        "dm_radial_slope_mean": _nanmean(rs),
        "dm_mad_mean": float(np.abs(np.diff(lma)).mean()) if len(lma) > 1 else float("nan"),
    }


# ----------------------------------------------------------------- assembly --
def feature_names(groups=GROUPS, pools=POOLS) -> list:
    """Ordered feature names for the requested groups/pools (deterministic)."""
    names = []
    for g in GROUPS:
        if g not in groups:
            continue
        if g == "spatial":
            names += [f"sp_{s}_{p}" for s in SPATIAL_STATS for p in pools]
        elif g == "intertile":
            names += list(INTERTILE_STATS)
        elif g == "diff":
            names += list(DIFF_STATS)
    return names


def feature_vector(canvas, sidecar, *, groups=GROUPS, pools=POOLS,
                   mask_timestamp: bool = True, diff_frames=None) -> dict:
    """Full feature dict for one mosaic.

    ``canvas``  : HxWx3 uint8 RGB mosaic (as returned by jadal.mosaic.build)
    ``sidecar`` : its sidecar dict (tile offsets and times)
    ``diff_frames``: optional list of N RGB frames in time order; when given, the
                     ``diff`` group is computed from |I_t - I_{t-1}| of those frames.
    """
    tiles = []
    for rgb, m, t in tile_grid(canvas, sidecar):
        if not mask_timestamp:
            m = letterbox_mask(rgb, np.ones_like(m))
        tiles.append((rgb, m, t))

    out = {}
    if "spatial" in groups:
        per = [spatial_features(rgb, m, to_luma(rgb)) for rgb, m, _ in tiles]
        for s in SPATIAL_STATS:
            vals = np.asarray([p[s] for p in per], float)
            for p in pools:
                key = f"sp_{s}_{p}"
                if p == "mean":
                    out[key] = float(np.nanmean(vals))
                elif p == "min":
                    out[key] = float(np.nanmin(vals))
                elif p == "max":
                    out[key] = float(np.nanmax(vals))
                else:
                    out[key] = float(np.nanstd(vals))
    if "intertile" in groups:
        out.update(intertile_features(tiles))
    if "diff" in groups:
        if diff_frames is None:
            raise ValueError("diff group requested but diff_frames not given")
        out.update(diff_features(diff_frames, int(sidecar["tile_px"]),
                                 mask_timestamp=mask_timestamp))
    return out


def feature_digest(feats: dict, ndigits: int = 6) -> str:
    """Receipt-compatible digest: sha256 of the quantised, canonically ordered values."""
    q = {k: (None if feats[k] is None or (isinstance(feats[k], float) and np.isnan(feats[k]))
             else round(float(feats[k]), ndigits))
         for k in sorted(feats)}
    doc = {"feature_version": FEATURE_VERSION, "values": q}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()
