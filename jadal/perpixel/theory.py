"""Signal-theory measurements for one window (deterministic, seed-free).

All math float64; every output quantised to Q=4 decimals BEFORE any
comparison (call quantise()). Thresholds are compared as integers
(value_q * 10**4) so floating-point noise cannot flip a decision.

RAW vs CLEANED split (Amendment A):
  - quality estimators (noise sigma, blockiness, exposure) use RAW pixels:
    cleaning would erase exactly what they measure.
  - activity/temporal stats use CLEANED luma: per-frame global-mean removal
    (kills exposure flicker), global-motion compensation, temporal
    median-of-3, spatial Gaussian. Each step is ablated in
    eval/perpixel/theory_clean.py and stays only if it helps nested CV.

Units: luma in [0,1] code values; sigma in code values; frequencies in Hz;
T-statistics dimensionless; blur_norm dimensionless (laplacian var / sigma^2).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage, stats

Q = 4
QI = 10 ** Q  # fixed-point scale

THEORY_FEATURES = [
    "sigma_immer", "sigma_wavelet", "blockiness",
    "p05", "p50", "p95", "snr",
    "lap_var", "blur_norm", "mtf50",
    "frozen_idx", "frozen_logp",
    "still_run", "mot_frac",
    "g_peak_freq", "g_logp",
    "skin_mean", "skin_dyn",
]


def quantise(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        f = float(v)
        if not np.isfinite(f):
            f = 1e9 if f > 0 else -1e9
        out[k] = int(round(f * QI))
    return out


def immerkaer_sigma(gray: np.ndarray) -> float:
    """Immerkaer fast noise sigma on one luma frame, code values."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    c = ndimage.convolve(gray.astype(np.float64), k, mode="reflect")
    h, w = gray.shape
    return float(np.sqrt(np.pi / 2) * np.abs(c[1:-1, 1:-1]).sum()
                 / (6 * (w - 2) * (h - 2)))


def wavelet_mad_sigma(gray: np.ndarray) -> float:
    """MAD of the Haar HH1 band / 0.6745, code values."""
    import pywt
    _, (_, _, hh) = pywt.dwt2(gray.astype(np.float64), "haar")
    return float(np.median(np.abs(hh)) / 0.6745)


def blockiness(gray: np.ndarray) -> float:
    """8x8 grid discontinuity ratio (boundary vs interior diffs). ~1 = clean."""
    g = gray.astype(np.float64)
    bh = np.abs(np.diff(g, axis=1))
    bv = np.abs(np.diff(g, axis=0))
    w = g.shape[1]
    bcols = [c for c in range(7, w - 1, 8)]
    bmask_h = np.zeros(bh.shape[1], bool)
    bmask_h[bcols] = True
    h = g.shape[0]
    brows = [r for r in range(7, h - 1, 8)]
    bmask_v = np.zeros(bv.shape[0], bool)
    bmask_v[brows] = True
    bnd = np.concatenate([bh[:, bmask_h].ravel(), bv[bmask_v].ravel()])
    intr = np.concatenate([bh[:, ~bmask_h].ravel(), bv[~bmask_v].ravel()])
    return float(bnd.mean() / (intr.mean() + 1e-12))


def phase_shift_vec(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Integer (dy, dx) aligning b to a via phase correlation."""
    a0 = a - a.mean()
    b0 = b - b.mean()
    R = np.fft.rfft2(a0) * np.conj(np.fft.rfft2(b0))
    R = R / (np.abs(R) + 1e-12)
    cc = np.fft.irfft2(R, s=a.shape)
    jy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    Hh, Ww = a.shape
    dy = int(jy if jy <= Hh // 2 else jy - Hh)
    dx = int(ix if ix <= Ww // 2 else ix - Ww)
    return dy, dx


def clean_luma(luma: np.ndarray) -> np.ndarray:
    """Flicker removal + global-motion compensation + temp-median-3 + spatial.

    luma (T,H,W) float64 -> cleaned (T,H,W) float64. Deterministic.
    """
    T = luma.shape[0]
    fl = luma - luma.mean(axis=(1, 2), keepdims=True)
    ref = fl[0]
    gmc = np.empty_like(fl)
    gmc[0] = fl[0]
    for i in range(1, T):
        dy, dx = phase_shift_vec(ref, fl[i])
        gmc[i] = np.roll(fl[i], shift=(-dy, -dx), axis=(0, 1))
    tm = ndimage.median_filter(gmc, size=(3, 1, 1), mode="nearest")
    out = np.empty_like(tm)
    for i in range(T):
        out[i] = cv2.GaussianBlur(tm[i].astype(np.float32), (0, 0), 0.8)
    return out.astype(np.float64)


def welch_bands(x: np.ndarray, fps: float, nperseg: int = 32):
    """Hann-windowed Welch periodogram; band fracs of non-DC + entropy."""
    n = len(x)
    seg = min(n, nperseg)
    if seg < 8:
        return 0.0, 0.0, 0.0, 0.0
    step = seg // 2
    wins = []
    w = np.hanning(seg)
    for s in range(0, n - seg + 1, step):
        X = np.fft.rfft((x[s:s + seg] - x[s:s + seg].mean()) * w)
        wins.append(X.real ** 2 + X.imag ** 2)
    P = np.mean(wins, axis=0)
    fr = np.fft.rfftfreq(seg, d=1.0 / fps)

    def b(lo, hi):
        m = (fr[1:] >= lo) & (fr[1:] < hi)
        return float(P[1:][m].sum() / (P[1:].sum() + 1e-18))
    tot = P[1:].sum()
    q = P[1:] / (tot + 1e-18)
    q = q[q > 0]
    ent = float(-(q * np.log(q)).sum() / np.log(len(q) + 1e-12)) if len(q) else 0.0
    return b(0.15, 1.0), b(1.0, 3.0), b(3.0, 8.0), ent


def fisher_g(logp_out: list, x: np.ndarray, fps: float,
             lo: float = 0.3, hi: float = 8.0):
    """Fisher g-test for a periodic component in [lo,hi] Hz.

    Returns (peak_freq, log10 p) with analytic white-noise-null FAR
    p ~= n*(1-g)^(n-1). The null is NOT true for motion signals; use as a
    ranking statistic and check empirical FAR (see theory_clean.py).
    """
    n = len(x)
    F = np.fft.rfft(x - x.mean())
    P = F.real ** 2 + F.imag ** 2
    fr = np.fft.rfftfreq(n, d=1.0 / fps)
    m = (fr >= lo) & (fr <= hi)
    if m.sum() < 3 or P[m].sum() <= 0:
        return 0.0, 0.0
    Pm = P[m]
    g = float(Pm.max() / Pm.sum())
    nn = int(m.sum())
    p = float(np.clip(nn * (1 - g) ** (nn - 1), 1e-300, 1.0))
    return float(fr[m][int(np.argmax(Pm))]), float(np.log10(p))


def skin_series(rgb: np.ndarray) -> np.ndarray:
    """YCrCb-rule skin area fraction per frame (hand proxy; stated proxy)."""
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    r, b = rgb[..., 0], rgb[..., 2]
    cr = 0.5 * (r - gray) / (1 - 0.299 + 1e-12) + 0.5
    cb = 0.5 * (b - gray) / (1 - 0.114 + 1e-12) + 0.5
    return ((cr * 255 >= 133) & (cr * 255 <= 173)
            & (cb * 255 >= 77) & (cb * 255 <= 127)).mean(axis=(1, 2))


def radial_mtf50(gray: np.ndarray) -> float:
    """Spatial freq (cycles/px) where cumulative radial power hits 50%."""
    F = np.fft.rfft2(gray - gray.mean())
    P = F.real ** 2 + F.imag ** 2
    h, w = gray.shape
    yy, xx = np.mgrid[0:P.shape[0], 0:P.shape[1]]
    rr = np.sqrt((yy / h) ** 2 + (xx / w) ** 2)
    order = np.argsort(rr.ravel(), kind="stable")
    cum = np.cumsum(P.ravel()[order])
    tot = cum[-1]
    if tot <= 0:
        return 0.0
    return float(rr.ravel()[order][int(np.searchsorted(cum, 0.5 * tot))])


def extract_theory(frames90: np.ndarray, fps: float) -> dict:
    """frames90 (T,H,W,3) float64 [0,1] -> dict of THEORY_FEATURES (float)."""
    T = frames90.shape[0]
    luma = (0.299 * frames90[..., 0] + 0.587 * frames90[..., 1]
            + 0.114 * frames90[..., 2])
    samp = np.linspace(0, T - 1, min(T, 12)).astype(int)
    sig_i = float(np.median([immerkaer_sigma(luma[i]) for i in samp]))
    sig_w = float(np.median([wavelet_mad_sigma(luma[i]) for i in samp]))
    blk = float(np.median([blockiness(luma[i]) for i in samp]))
    flat = luma.ravel()
    p05, p50, p95 = (float(np.percentile(flat, q)) for q in (5, 50, 95))
    snr = float(flat.mean() / (sig_i + 1e-12))
    laps = [float(cv2.Laplacian(luma[i], cv2.CV_64F).var()) for i in samp]
    lap = float(np.median(laps))
    blur = float(lap / (sig_i ** 2 + 1e-12))
    mtf = float(np.median([radial_mtf50(luma[i]) for i in samp]))
    # frozen chi-square (CFAR/Neyman-Pearson): H0 = still frames + sensor noise
    d = np.diff(luma, axis=0)
    Ndof = float(d.size)
    Tchisq = float((d ** 2).sum() / (2 * sig_i ** 2 + 1e-18))
    fidx = float(Tchisq / Ndof)
    try:
        logp = float(stats.chi2.logsf(Tchisq, Ndof) / np.log(10))
    except Exception:
        logp = 0.0
    logp = float(np.clip(logp, -1e9, 0.0))
    # cleaned activity stats
    cl = clean_luma(luma)
    dc = np.diff(cl, axis=0)
    pix_thr = 3.0 * np.sqrt(2) * sig_i  # per-pixel 3-sigma two-sided
    mot = float((np.abs(dc) > pix_thr).mean())
    me = np.abs(dc).mean(axis=(1, 2))
    med = float(np.median(me))
    mad = float(np.median(np.abs(me - med)))
    floor = med + mad
    best = cur = 0
    for v in me:
        if v <= floor:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    still = float(best / max(len(me), 1))
    gf, glp = fisher_g([], me, fps)
    sk = skin_series(frames90)
    out = {
        "sigma_immer": sig_i, "sigma_wavelet": sig_w, "blockiness": blk,
        "p05": p05, "p50": p50, "p95": p95, "snr": snr,
        "lap_var": lap, "blur_norm": blur, "mtf50": mtf,
        "frozen_idx": fidx, "frozen_logp": logp,
        "still_run": still, "mot_frac": mot,
        "g_peak_freq": gf, "g_logp": glp,
        "skin_mean": float(sk.mean()), "skin_dyn": float(sk.std()),
    }
    return {k: float(v) for k, v in out.items()}
