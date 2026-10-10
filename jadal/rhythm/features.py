"""R: spatially-resolved, ego-motion-compensated residual-motion + skin representation.

This module turns a 10-second video window into a fixed-size tensor

    X  in  R^{T x G x G x C}   (grid channels)
    g  in  R^{T x 8}           (global per-frame channels)

The point of the representation is that every existing learned model in this
project consumed only GLOBAL per-frame scalars, so none of them could localise
*where* motion happens or *how* it repeats. The single strongest hand-built
feature known on the 150 human windows is `flow_act` (residual optical flow in
the hand region after ego-motion removal), AUROC 0.806 on its own -- as good as
any learned model. This module keeps that signal but resolves it on a grid.

Per frame pair (t-1, t):
  1. gray at ANALYSIS_W x ANALYSIS_H (default 160x90), area interpolation.
  2. ego-motion: global translation by phase correlation on a Hann-windowed
     frame; previous frame is warped to the current one.
  3. residual = |I_t - warp(I_{t-1})|; residual flow = Farneback flow minus the
     global translation, magnitude and direction coherence (resultant length).
  4. skin mask: YCrCb chromaticity rule (same rule as jadal/perpixel/series.py).
  5. everything pooled to a G x G grid of mean values.

Channels (C = GRID_CHANNELS = 6), all per frame:
  0 res_mean    mean residual |I_t - warp(I_{t-1})| / 255
  1 res_p90     90th percentile of that residual / 255
  2 skin_frac   fraction of skin pixels
  3 tvar        temporal variance of the (unwarped) gray frame over a 5-frame box
  4 flow_mag    mean |flow - global| / ANALYSIS_W
  5 flow_coh    resultant length (direction coherence) of (flow - global)

Global channels (8, identical definitions to the frozen perpixel series):
  luma_mean, lap_var, diff_e, skin_frac, shift_px, under_frac, over_frac, dct_hf

Determinism: all maths is OpenCV/numpy float32 with fixed constants; no random
numbers anywhere in this module. Same input file -> same tensor bytes.

WHAT IS NOT DONE:
  * Ego-motion is a TRANSLATION only (phase correlation). No affine/ECC, no
    rotation or scale. `ego_compensate=False` disables it entirely so the
    contribution can be ablated (the repo has contradictory evidence about it).
  * "Hand region" is never detected here. The grid is a fixed partition of the
    frame; the model is expected to learn which cells matter. No MediaPipe.
  * Skin is a chromaticity rule, not a hand detector: wood, cardboard and skin
    share chromaticity. It is a weak channel, included because it is cheap.
  * Temporal variance uses a box filter over a FIXED 5-frame window, so it is
    not scale-free across frame rates. Frames are resampled to TARGET_FPS first.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---- fixed representation constants (never per-clip, never per-window) ---- #
ANALYSIS_W = 160
ANALYSIS_H = 90
TARGET_FPS = 10.0
GRID_CHANNELS = 6
CHANNEL_NAMES = ("res_mean", "res_p90", "skin_frac", "tvar", "flow_mag", "flow_coh")
GLOBAL_NAMES = (
    "luma_mean",
    "lap_var",
    "diff_e",
    "skin_frac",
    "shift_px",
    "under_frac",
    "over_frac",
    "dct_hf",
)
DEFAULT_T = 100  # 10 s at TARGET_FPS

# YCrCb skin chromaticity rule (matches jadal/perpixel/series.py)
SKIN_CR = (133, 173)
SKIN_CB = (77, 127)

_DCT_BLOCK = 8


# --------------------------------------------------------------------------- #
# frame sampling
# --------------------------------------------------------------------------- #
def sample_indices(fps: float, start_s: float, end_s: float, target_fps: float = TARGET_FPS):
    """Frame indices to read for [start_s, end_s) resampled to target_fps.

    Returns (indices, source_fps). Indices are absolute frame numbers and are
    strictly increasing. The first frame of the window is always included.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    duration = max(end_s - start_s, 1e-6)
    n_out = max(int(round(duration * target_fps)), 2)
    # uniform time grid inside the window
    times = start_s + (np.arange(n_out) + 0.5) * (duration / n_out)
    idx = np.floor(times * fps).astype(np.int64)
    idx = np.clip(idx, int(np.floor(start_s * fps)), max(int(np.ceil(end_s * fps)) - 1, 0))
    # deduplicate while preserving order (short windows / low fps)
    keep = np.ones(len(idx), bool)
    keep[1:] = idx[1:] != idx[:-1]
    idx = idx[keep]
    if len(idx) < 2:  # degenerate: fall back to consecutive frames
        f0 = int(np.floor(start_s * fps))
        idx = np.array([f0, f0 + 1], dtype=np.int64)
    return idx, float(fps)


def _read_frames(cap: cv2.VideoCapture, indices: np.ndarray) -> np.ndarray:
    """Read the given absolute frame indices, returning float32 gray 160x90."""
    frames = np.empty((len(indices), ANALYSIS_H, ANALYSIS_W), np.float32)
    pos = 0
    for i, fi in enumerate(indices):
        if pos != int(fi):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if not ok:  # seek past EOF: rewind to the last good frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi) - 1)
            ok, bgr = cap.read()
            if not ok:
                raise IOError(f"cannot decode frame {fi}")
        pos = int(fi) + 1
        small = cv2.resize(bgr, (ANALYSIS_W, ANALYSIS_H), interpolation=cv2.INTER_AREA)
        frames[i] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return frames


def _read_frames_color(cap: cv2.VideoCapture, indices: np.ndarray) -> np.ndarray:
    """Same as _read_frames but returns BGR uint8 for the skin channel."""
    out = np.empty((len(indices), ANALYSIS_H, ANALYSIS_W, 3), np.uint8)
    pos = 0
    for i, fi in enumerate(indices):
        if pos != int(fi):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi) - 1)
            ok, bgr = cap.read()
            if not ok:
                raise IOError(f"cannot decode frame {fi}")
        pos = int(fi) + 1
        out[i] = cv2.resize(bgr, (ANALYSIS_W, ANALYSIS_H), interpolation=cv2.INTER_AREA)
    return out


# --------------------------------------------------------------------------- #
# per-frame global channels (match the frozen perpixel series)
# --------------------------------------------------------------------------- #
def _dct_hf(gray: np.ndarray) -> float:
    """High-frequency DCT energy fraction, mean over 8x8 blocks."""
    h, w = gray.shape
    h8, w8 = (h // _DCT_BLOCK) * _DCT_BLOCK, (w // _DCT_BLOCK) * _DCT_BLOCK
    if h8 == 0 or w8 == 0:
        return 0.0
    blocks = gray[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3)
    flat = np.ascontiguousarray(blocks.reshape(-1, 8, 8), dtype=np.float32)
    # cv2.dct is single-plane only; batch with a vectorised matrix DCT instead.
    # 2-D DCT-II of an 8x8 block = M @ block @ M.T with M the orthonormal DCT matrix.
    n = np.arange(8, dtype=np.float32)
    k = n[:, None]
    M = np.cos(np.pi * (2 * n[None, :] + 1) * k / 16.0) * np.sqrt(2.0 / 8.0)
    M[0] *= np.sqrt(0.5)
    d = np.einsum("ij,bjk,lk->bil", M, flat, M).reshape(-1, 64)
    total = np.abs(d).sum(axis=1) + 1e-9
    hf = np.abs(d[:, 16:]).sum(axis=1)  # everything outside the 4x4 low corner
    return float(np.mean(hf / total))


def _global_channels(gray: np.ndarray, skin: np.ndarray, shift_px: np.ndarray) -> np.ndarray:
    """(T, 8) global per-frame channels."""
    T = len(gray)
    out = np.zeros((T, len(GLOBAL_NAMES)), np.float32)
    for t in range(T):
        g = gray[t]
        lap = cv2.Laplacian(g, cv2.CV_32F)
        d = np.abs(g - gray[t - 1]).mean() / 255.0 if t > 0 else 0.0
        out[t] = (
            g.mean() / 255.0,
            float(lap.var()),
            float(d),
            float(skin[t].mean()),
            float(shift_px[t]) / ANALYSIS_W,
            float((g < 10).mean()),
            float((g > 245).mean()),
            _dct_hf(g),
        )
    return out


# --------------------------------------------------------------------------- #
# ego-motion + residual
# --------------------------------------------------------------------------- #
def _phase_shift(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """Global translation (dx, dy) mapping prev onto cur, by phase correlation.

    IMPORTANT: `cv2.phaseCorrelate` overwrites BOTH input arrays in place on this
    OpenCV build (5.0.0). Passing the caller's frames directly silently destroys
    the gray stack and makes the whole representation non-deterministic. The
    copies below are load-bearing, not defensive style -- do not remove them.
    Verified by tests/test_rhythm_features.py::test_grid_channels_does_not_mutate_its_inputs.
    """
    h, w = cur.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(prev.copy(), cur.copy(), win)
    return float(dx), float(dy)


def _warp_translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _grid_mean(a: np.ndarray, G: int) -> np.ndarray:
    """(H, W) -> (G, G) block means. Uses resize with INTER_AREA (exact for divisors)."""
    return cv2.resize(a, (G, G), interpolation=cv2.INTER_AREA)


def grid_channels(gray: np.ndarray, color: np.ndarray, G: int,
                  ego_compensate: bool = True, use_flow: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute (T, G, G, C) grid channels and (T,) shift magnitudes."""
    T = len(gray)
    X = np.zeros((T, G, G, GRID_CHANNELS), np.float32)
    shifts = np.zeros(T, np.float32)

    skin = np.zeros((T, ANALYSIS_H, ANALYSIS_W), np.float32)
    for t in range(T):
        ycrcb = cv2.cvtColor(color[t], cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        skin[t] = ((cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1]) &
                   (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1])).astype(np.float32)

    # temporal variance over a fixed 5-frame box (causal-free, symmetric)
    tvar = np.zeros_like(gray)
    k = 5
    pad = k // 2
    gp = np.pad(gray, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    acc = np.zeros_like(gray)
    for j in range(k):
        acc += gp[j:j + T]
    m1 = acc / k
    acc2 = np.zeros_like(gray)
    for j in range(k):
        d = gp[j:j + T] - m1
        acc2 += d * d
    tvar = acc2 / k

    prev_flow = None
    for t in range(T):
        if t == 0:
            res = np.zeros_like(gray[0])
            flow_res_mag = np.zeros_like(gray[0])
            flow_coh = np.zeros_like(gray[0])
            dx = dy = 0.0
        else:
            dx, dy = _phase_shift(gray[t - 1], gray[t])
            shifts[t] = float(np.hypot(dx, dy))
            ref = _warp_translate(gray[t - 1], dx, dy) if ego_compensate else gray[t - 1]
            res = np.abs(gray[t] - ref)

            if use_flow:
                flow = cv2.calcOpticalFlowFarneback(
                    gray[t - 1], gray[t], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                fx, fy = flow[:, :, 0], flow[:, :, 1]
                if ego_compensate:
                    fx = fx - dx
                    fy = fy - dy
                flow_res_mag = np.hypot(fx, fy)
                # direction coherence over a local 9x9 window
                ksz = 9
                sx = cv2.boxFilter(fx, -1, (ksz, ksz))
                sy = cv2.boxFilter(fy, -1, (ksz, ksz))
                flow_coh = np.hypot(sx, sy) / (cv2.boxFilter(flow_res_mag, -1, (ksz, ksz)) + 1e-6)
            else:
                flow_res_mag = np.zeros_like(gray[0])
                flow_coh = np.zeros_like(gray[0])

        p90 = float(np.percentile(res, 90)) / 255.0
        chans = (
            _grid_mean(res, G) / 255.0,
            np.full((G, G), p90, np.float32),
            _grid_mean(skin[t], G),
            _grid_mean(tvar[t], G) / 255.0,
            _grid_mean(flow_res_mag, G) / ANALYSIS_W,
            _grid_mean(flow_coh, G),
        )
        X[t] = np.stack(chans, axis=-1)
        prev_flow = flow_res_mag

    return X, shifts


# --------------------------------------------------------------------------- #
# window-level entry points
# --------------------------------------------------------------------------- #
def extract_window(video_path: str, start_s: float, end_s: float, G: int = 6,
                   ego_compensate: bool = True, use_flow: bool = True,
                   target_fps: float = TARGET_FPS) -> dict:
    """Extract R for one window. Returns dict with X (T,G,G,C), g (T,8), fps, T."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 1e-3:
            fps = float(TARGET_FPS)
        idx, fps = sample_indices(fps, start_s, end_s, target_fps)
        gray = _read_frames(cap, idx)
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        color = _read_frames_color(cap, idx)
    finally:
        cap.release()

    X, shifts = grid_channels(gray, color, G, ego_compensate, use_flow)
    skin = np.zeros((len(gray), ANALYSIS_H, ANALYSIS_W), np.float32)
    for t in range(len(gray)):
        ycrcb = cv2.cvtColor(color[t], cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        skin[t] = ((cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1]) &
                   (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1])).astype(np.float32)
    g = _global_channels(gray, skin, shifts)
    return {"X": X, "g": g, "fps": fps, "T": int(len(gray)), "frames": idx}


def resample_T(a: np.ndarray, T_out: int) -> np.ndarray:
    """Linear-resample the time axis of (T, ...) to T_out frames."""
    T = a.shape[0]
    if T == T_out:
        return a
    src = np.linspace(0.0, 1.0, T, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, T_out, dtype=np.float32)
    flat = a.reshape(T, -1)
    out = np.empty((T_out, flat.shape[1]), np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst, src, flat[:, j])
    return out.reshape((T_out,) + a.shape[1:])


def tensor_digest(X: np.ndarray, g: np.ndarray) -> str:
    """SHA-256 of the quantised tensor bytes (1e-4 grid), for receipt digests."""
    xq = np.round(np.asarray(X, np.float32), 4)
    gq = np.round(np.asarray(g, np.float32), 4)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(xq, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(gq, dtype=np.float32).tobytes())
    return h.hexdigest()


# ---- cache helpers -------------------------------------------------------- #
def cache_path(cache_dir: str | Path, window_id: str, G: int) -> Path:
    return Path(cache_dir) / f"{window_id}_G{G}.npz"


def save_cache(cache_dir: str | Path, window_id: str, out: dict, G: int) -> Path:
    p = cache_path(cache_dir, window_id, G)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        X=out["X"].astype(np.float16),
        g=out["g"].astype(np.float16),
        fps=np.float32(out["fps"]),
        T=np.int32(out["T"]),
    )
    return p


def load_cache(cache_dir: str | Path, window_id: str, G: int) -> Optional[dict]:
    p = cache_path(cache_dir, window_id, G)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {"X": z["X"].astype(np.float32), "g": z["g"].astype(np.float32),
            "fps": float(z["fps"]), "T": int(z["T"])}
