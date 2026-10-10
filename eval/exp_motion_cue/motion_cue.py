"""E6 motion cue: deterministic residual-motion extraction from video.

Why this file exists
--------------------
E6 asks whether the strongest *motion* cue is really just ego-motion magnitude
("is the wearer walking?").  The documented cue (docs/ALGO_SPEC.md) is
"residual optical flow after ego-motion removal".  None of the cached scalars
(`series_gold.npz` channels `shift_px`, `diff_e`) is that quantity: `shift_px`
is the phase-correlation shift vs the first frame (ego-motion magnitude) and
`diff_e` is the uncompensated frame difference.  This module recomputes the
cue from the video with one code path for both regimes:

  gray 160x90 (area), every `stride`-th frame (IndEgo 10 fps -> stride 1,
  Egocentric-10K 1080p/30 fps transcode -> stride 3 so the effective rate is
  ~10 fps, as in docs/ALGO_SPEC.md).

Per consecutive frame pair (t-1, t) at the analysis resolution:
  * `ego`   = |phaseCorrelate shift| in px  -> global (translation-only) motion
  * `resid` = mean |I_t - warp(I_{t-1})|    -> residual motion NOT explained by
               the translation-only model  (the cue under test)
  * `diff`  = mean |I_t - I_{t-1}|          -> uncompensated frame difference

Deterministic: fixed size, fixed stride, no sampling, no randomness; cv2
phaseCorrelate / warpAffine are deterministic for identical input bytes.

Only the residual-motion magnitude is the cue; higher residual = more motion
left after compensation.  Orientation is decided by the caller (see run_e6.py).
"""
from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np

W, H = 160, 90          # analysis resolution (matches caches/perpixel series_gold)
WIN_FRAMES = 100        # IndEgo: ~10 s at 10 fps


def decode_gray(path: str, start_s: float = 0.0, n_frames: int = WIN_FRAMES,
                stride: int = 1) -> np.ndarray:
    """Decode `n_frames` analysis frames (after `stride` subsampling) -> float32 (T,H,W) in [0,1]."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)
    out: list[np.ndarray] = []
    i = 0
    while len(out) < n_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
            out.append(g)
        i += 1
    cap.release()
    if len(out) < 2:
        raise RuntimeError(f"too few frames: {path} @ {start_s}")
    return np.stack(out).astype(np.float32) / 255.0


def cue_series(gray: np.ndarray) -> dict:
    """Per-window motion scalars from an analysis-resolution gray stack (T,H,W) in [0,1]."""
    T = gray.shape[0]
    ego, resid, diff = [], [], []
    for t in range(1, T):
        prev, cur = gray[t - 1], gray[t]
        (dx, dy), _resp = cv2.phaseCorrelate(prev, cur)
        ego.append(float(np.hypot(dx, dy)))
        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        warped = cv2.warpAffine(prev, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        resid.append(float(np.mean(np.abs(cur - warped))))
        diff.append(float(np.mean(np.abs(cur - prev))))
    ego = np.asarray(ego, np.float64)
    resid = np.asarray(resid, np.float64)
    diff = np.asarray(diff, np.float64)
    return {
        # the cue under test: residual motion after translation-only compensation
        "resid_mean": float(resid.mean()),
        "resid_med": float(np.median(resid)),
        "resid_p90": float(np.percentile(resid, 90)),
        "resid_std": float(resid.std()),
        # ego-motion magnitude (translation-only model) from the same pixels
        "ego_mean": float(ego.mean()),
        "ego_med": float(np.median(ego)),
        "ego_p90": float(np.percentile(ego, 90)),
        # uncompensated frame difference (matches cached diff_e definition)
        "diff_mean": float(diff.mean()),
        "n_pairs": int(T - 1),
    }


def _extract_one(job: dict) -> dict:
    """Module-level worker (picklable for multiprocessing)."""
    gray = decode_gray(job["path"], job.get("start_s", 0.0),
                       n_frames=job.get("n_frames", WIN_FRAMES),
                       stride=job.get("stride", 1))
    return {"id": job["id"], **cue_series(gray)}


def extract_windows(jobs: Iterable[dict], workers: int = 3) -> list[dict]:
    """jobs: {id, path, start_s, stride}. Returns list of {id, **cue_series}.

    Multiprocessing over windows (<= workers, CPU only); deterministic order.
    """
    jobs = list(jobs)
    _one = _extract_one

    if workers <= 1:
        return [_one(j) for j in jobs]
    from multiprocessing import Pool
    with Pool(workers) as pool:
        return pool.map(_one, jobs)


def bundle() -> str:
    return os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
