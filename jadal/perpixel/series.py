"""Per-frame feature time series: EVERY decoded frame at 160x90 + descriptors.

Decode: one ffmpeg subprocess per segment, scale=160:90, rgb24 pipe,
frames EXACTLY [start_s, end_s). All math float64, stored float32.
Deterministic: fixed decode order, no randomness.

Per-frame channels (units):
  luma_mean  : mean luma in [0,1]
  lap_var    : variance of Laplacian of luma (sharpness, raw units)
  diff_e     : mean |luma - prev luma|, first frame 0 (motion energy/frame)
  skin_frac  : YCrCb-rule skin mask area fraction in [0,1]
  shift_px   : phase-correlation shift (px at 160x90) of frame vs FIRST frame
  under_frac : fraction luma < 0.05
  over_frac  : fraction luma > 0.95
  dct_hf     : mean |DCT| with top-left 8x8 zeroed (log1p in descriptors uses raw)

Descriptors (per series s of length T, fps=T/duration):
  band energies lo(0.15-1Hz)/mid(1-3)/hi(3-8) as fraction of non-DC energy
  spectral entropy (normalised, nats/ln(nbins))
  autocorrelation peak (lag>=1, normalised) + lag of peak (in frames)
  burstiness = std/mean of (s + eps)
  still_run = longest run of diff_e below floor / T  (only for diff_e)
  motion_floor_frac = fraction of frames with diff_e above data-driven floor
    (floor = median(diff_e) + MAD(diff_e); reported for diff_e only)
  mean/std/min/max of each series (raw units)
"""

from __future__ import annotations

import subprocess

import cv2
import numpy as np

W, H = 160, 90

SERIES_CHANNELS = ["luma_mean", "lap_var", "diff_e", "skin_frac",
                   "shift_px", "under_frac", "over_frac", "dct_hf"]

DESC_SERIES = ["luma_mean", "diff_e", "skin_frac", "lap_var"]


def decode_160x90(path: str, start_s: float, end_s: float) -> tuple[np.ndarray, float]:
    """Decode EXACTLY [start_s, end_s) at 160x90. Returns (frames float64 [0,1], fps)."""
    dur = end_s - start_s
    if dur <= 0:
        raise ValueError(f"empty range [{start_s},{end_s})")
    pre = max(0.0, start_s - 3.0)
    off = start_s - pre
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error",
         "-ss", str(pre), "-i", path,
         "-ss", str(off), "-t", str(dur),
         "-map", "0:v:0", "-vf", f"scale={W}:{H}",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, timeout=300)
    n = len(p.stdout) // (W * H * 3)
    if n == 0:
        raise ValueError(f"no frames decoded in [{start_s},{end_s}) of {path}")
    frames = np.frombuffer(p.stdout, dtype=np.uint8, count=n * W * H * 3)
    frames = frames.reshape(n, H, W, 3).astype(np.float64) / 255.0
    return frames, n / dur


def _phase_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Phase-correlation shift magnitude (px) between luma frames a, b."""
    a0 = a - a.mean()
    b0 = b - b.mean()
    FA = np.fft.rfft2(a0)
    FB = np.fft.rfft2(b0)
    R = FA * np.conj(FB)
    R = R / (np.abs(R) + 1e-12)
    cc = np.fft.irfft2(R, s=a.shape)
    jy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    Hh, Ww = a.shape
    return float(np.hypot(min(jy, Hh - jy), min(ix, Ww - ix)))


def per_frame_series(frames: np.ndarray) -> np.ndarray:
    """frames (T,H,W,3) float64 [0,1] -> (T,8) float64 series in SERIES_CHANNELS order."""
    T = frames.shape[0]
    out = np.zeros((T, len(SERIES_CHANNELS)), dtype=np.float64)
    prev = None
    ref = (0.299 * frames[0, :, :, 0] + 0.587 * frames[0, :, :, 1]
           + 0.114 * frames[0, :, :, 2])
    for i in range(T):
        rgb = frames[i]
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        diff = 0.0 if prev is None else float(np.mean(np.abs(gray - prev)))
        prev = gray
        cr = 0.5 * (r - gray) / ((1 - 0.299) + 1e-12) + 0.5
        cb = 0.5 * (b - gray) / ((1 - 0.114) + 1e-12) + 0.5
        skin = ((cr * 255 >= 133) & (cr * 255 <= 173)
                & (cb * 255 >= 77) & (cb * 255 <= 127)).mean()
        shift = 0.0 if i == 0 else _phase_shift(ref, gray)
        D = cv2.dct(gray.astype(np.float32)).astype(np.float64)
        hf = D.copy()
        hf[:8, :8] = 0.0
        out[i] = [float(gray.mean()), lap, diff, float(skin), shift,
                  float(np.mean(gray < 0.05)), float(np.mean(gray > 0.95)),
                  float(np.mean(np.abs(hf)))]
    return out


def _band_frac(x: np.ndarray, fps: float) -> tuple[float, float, float]:
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 0.0
    F = np.fft.rfft(x - x.mean())
    P = F.real ** 2 + F.imag ** 2
    fr = np.fft.rfftfreq(n, d=1.0 / fps)
    tot = P[1:].sum()
    if not np.isfinite(tot) or tot <= 0:
        return 0.0, 0.0, 0.0
    def b(lo, hi):
        m = (fr[1:] >= lo) & (fr[1:] < hi)
        return float(P[1:][m].sum() / tot)
    return b(0.15, 1.0), b(1.0, 3.0), b(3.0, 8.0)


def _spec_entropy(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    F = np.fft.rfft(x - x.mean())
    P = (F.real ** 2 + F.imag ** 2)[1:]
    tot = P.sum()
    if not np.isfinite(tot) or tot <= 0:
        return 0.0
    q = P / tot
    q = q[q > 0]
    return float(-(q * np.log(q)).sum() / np.log(len(q) + 1e-12))


def _ac_peak_lag(x: np.ndarray) -> tuple[float, int]:
    n = len(x)
    x0 = x - x.mean()
    if float((x0 ** 2).sum()) <= 1e-18 or n < 4:
        return 0.0, 0
    ac = np.correlate(x0, x0, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-18)
    seg = ac[1:]
    j = int(np.argmax(seg))
    return float(seg[j]), j + 1


def describe_series(S: np.ndarray, fps: float) -> dict:
    """S (T,8) float64 -> dict of scalar descriptors (float)."""
    idx = {c: j for j, c in enumerate(SERIES_CHANNELS)}
    d: dict[str, float] = {}
    for ch in DESC_SERIES:
        x = np.asarray(S[:, idx[ch]], dtype=np.float64)
        lo, mid, hi = _band_frac(x, fps)
        pk, lag = _ac_peak_lag(x)
        mu = float(x.mean())
        sd = float(x.std())
        d[f"{ch}_band_lo"] = lo
        d[f"{ch}_band_mid"] = mid
        d[f"{ch}_band_hi"] = hi
        d[f"{ch}_spec_ent"] = _spec_entropy(x)
        d[f"{ch}_ac_peak"] = pk
        d[f"{ch}_ac_lag"] = float(lag)
        d[f"{ch}_mean"] = mu
        d[f"{ch}_std"] = sd
        d[f"{ch}_min"] = float(x.min())
        d[f"{ch}_max"] = float(x.max())
        d[f"{ch}_burst"] = float(sd / (abs(mu) + 1e-9))
    # motion-floor descriptors from diff_e (data-driven floor: median + MAD)
    diff = np.asarray(S[:, idx["diff_e"]], dtype=np.float64)
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med)))
    floor = med + mad
    above = diff > floor
    d["motion_floor"] = float(floor)
    d["motion_frac_above"] = float(above.mean())
    # longest still run (below floor), as fraction of T
    best = cur = 0
    for v in diff:
        if v <= floor:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    d["still_run_frac"] = float(best / max(len(diff), 1))
    # global shift / saturation summaries
    sh = np.asarray(S[:, idx["shift_px"]], dtype=np.float64)
    d["shift_mean"] = float(sh.mean())
    d["shift_max"] = float(sh.max())
    d["under_mean"] = float(S[:, idx["under_frac"]].mean())
    d["over_mean"] = float(S[:, idx["over_frac"]].mean())
    d["dct_hf_mean"] = float(np.log1p(S[:, idx["dct_hf"]].mean()))
    d["dct_hf_std"] = float(S[:, idx["dct_hf"]].std())
    d["nframes"] = float(S.shape[0])
    d["fps"] = float(fps)
    return {k: float(v) for k, v in d.items()}


DESCRIPTOR_NAMES = sorted(describe_series(
    np.arange(32, dtype=np.float64).reshape(16, 2)
    @ np.ones((2, 8), dtype=np.float64), 10.0).keys())
