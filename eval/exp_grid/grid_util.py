#!/usr/bin/env python3
"""Shared decode/build helpers for the GRID.05 (S0 parity) and GRID.07 (cost) runs.

Kept here rather than in ``jadal/`` because it is experiment scaffolding: it
decodes source pixels with cv2 and times stages, neither of which the shipped
pipeline does (the shipped path decodes 160x90 with ffmpeg).

No pixels are written anywhere by this module.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2  # noqa: E402
import poc_run  # noqa: E402
from jadal.mosaic import build as B  # noqa: E402
from jadal.mosaic import features as F  # noqa: E402

WORK = (160, 90)


def decode_rgb(path: str, start_s: float, seconds: float, *, keep=None,
               resize_to=None):
    """Decode ``seconds`` of a clip as RGB uint8, keeping only the ``keep`` indices.

    Returns ``(frames, native_fps, n_read)`` where ``frames`` is a dict
    ``{index: RGB uint8 array}`` when ``keep`` is given, else a list of every
    frame.  ``resize_to`` (w, h) downscales each kept frame with INTER_AREA.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)
    want = max(1, int(round(seconds * fps)))
    keeps = None if keep is None else set(int(k) for k in keep)
    out = {} if keeps is not None else []
    n = 0
    while n < want:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if keeps is None or n in keeps:
            rgb = np.ascontiguousarray(fr[..., ::-1])
            if resize_to is not None:
                rgb = cv2.resize(rgb, resize_to, interpolation=cv2.INTER_AREA)
            if keeps is None:
                out.append(rgb)
            else:
                out[n] = rgb
        n += 1
    cap.release()
    if n == 0:
        raise RuntimeError(f"no frames {path}@{start_s}")
    return out, fps, n


def luma_of_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    """The production luma definition, on uint8 RGB -> float64 in [0, 1]."""
    f = np.asarray(rgb, dtype=np.float64) / 255.0
    return poc_run.luma_of(f)


def production_probe_from_clip(path: str, start_s: float = 0.0,
                               seconds: float = 2.0) -> dict:
    """The shipped 2-second S0 probe on a clip: decode -> 160x90 -> luma -> probe."""
    frames, fps, n = decode_rgb(path, start_s, seconds, resize_to=WORK)
    probe = production_probe_from_frames(frames, fps)
    probe["_decode"] = {"native_fps": fps, "n_frames_read": n,
                        "n_source_frames": len(frames)}
    return probe


def production_probe_from_frames(frames, fps: float) -> dict:
    """``poc_run.s0_from_luma`` on already-decoded RGB frames.

    Frames are downscaled to the 160x90 working resolution first if they are not
    already there (cv2 INTER_AREA; production uses ffmpeg ``scale=160:90``,
    bicubic -- recorded as a confound in the results, not hidden).

    The frame-repeat to the 30 fps production normalisation is the same rule the
    portal's ``_normalise_fps`` applies (IndEgo is 10 fps, so 2 s of source
    becomes 20 native frames repeated 3x = the 60 probe frames).
    """
    frames = list(frames)
    if frames[0].shape[0] != WORK[1] or frames[0].shape[1] != WORK[0]:
        frames = [cv2.resize(f, WORK, interpolation=cv2.INTER_AREA) for f in frames]
    rep = max(1, int(round(30.0 / fps))) if fps < 29.5 else 1
    idx = np.repeat(np.arange(len(frames)), rep)
    if len(idx) < 60:
        idx = np.concatenate([idx, np.full(60 - len(idx), len(frames) - 1)])
    luma = np.stack([luma_of_rgb_u8(frames[i]) for i in idx[:60]])
    probe = poc_run.s0_from_luma(luma)
    probe["_probe_source_frames"] = len(frames)
    probe["_repeat"] = rep
    return probe


def tile_frame_indices(n_frames: int, fps: float, n_tiles: int = 9,
                       window_s: float = 10.0):
    """The builder's cell-centred index rule for the decoded frame count."""
    times = B.tile_times(n_tiles, window_s)
    return B.frame_indices(times, fps, n_frames), times


def plan_clip_indices(path: str, start_s: float, seconds: float,
                      n_tiles: int = 9):
    """Plan the tile frame indices for a 10 s window without decoding.

    Returns ``(idx, times, fps, n_available)``.  ``n_available`` comes from the
    container's frame count when the container reports one; when it does not, the
    window length alone is assumed and a short read would raise later rather than
    silently shorten the window.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total > 0:
        start_frame = int(round(start_s * fps))
        avail = max(1, min(int(round(seconds * fps)), total - start_frame))
    else:
        avail = int(round(seconds * fps))
    idx, times = tile_frame_indices(avail, fps, n_tiles, seconds)
    return idx, times, fps, avail


def build_mosaic_from_frames(kept: dict, idx, times, *, n_tiles=9, tile_px=224,
                            window_id="w", source="clip", fps_used=None,
                            timestamps=True):
    """Build the canvas from already-decoded frames ({index: RGB uint8})."""
    use = [np.asarray(kept[int(k)], np.uint8) for k in idx]
    grid = B.grid_for_n(n_tiles)
    return B.build_mosaic(use, times, idx, grid, tile_px, window_id=window_id,
                          source=source,
                          fps_used=float(fps_used if fps_used is not None
                                         else n_tiles / B.WINDOW_S),
                          timestamps=timestamps)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


#: mosaic feature groups the S0 parity probe reads (spatial + intertile only:
#: the ``diff`` group needs a difference mosaic and no S0 rule uses it)
S0_GROUPS = ("spatial", "intertile")
