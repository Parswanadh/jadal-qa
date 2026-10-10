"""F50 — 24 additional per-pixel window features (26 -> 50).

Extends the frozen 26-feature set in `jadal/gpu/features.py`. Every feature below
is a scalar per 10-s window, computed from the same decoded frames (luma + rgb,
160x90) that the frozen extractor uses, so it can be appended to the 26 columns.

All are cheap: no optical flow, no per-frame FFT beyond 8x8 DCT + one rFFT-free
histogram. Cost is dominated by the decode; the 24 kernels add < 1 ms/window on
one CPU core (measured in eval/results/exp_feat50/cost.json).

Definitions (name — definition — cost):
  Z3 transferable set (6)
    ct_fsize_std          std(per-frame compressed packet size)/mean, over the
                          window (dimensionless frame-size variability). 1 ffprobe.
    mo_hand_motion_excess skin-mask-weighted mean |luma[t]-luma[t-1]| minus the
                          global mean |diff| (hand region moves more than bg). O(T).
    inv_clip_frac         fraction of pixels with luma<0.02 or luma>0.98, mean
                          over sampled frames. O(T).
    inv_laplacian_var     variance of the 4-neighbour Laplacian of the mid luma
                          frame (sharpness). O(HW).
    inv_hist_entropy      Shannon entropy (bits) of the 32-bin luma histogram of
                          the mid frame. O(HW).
    inv_dct_hf_lf         mid-frame 8x8 DCT energy with radial freq r>=4 divided
                          by energy with r<=1. O(HW).
  Finer DCT/spectral bands (4)
    dct_band_r1..r3       fraction of 8x8 DCT AC energy in radial band r=1,2,3.
    dct_band_r4plus       fraction of 8x8 DCT AC energy in radial band r>=4.
  Colour-channel ratios + colour statistics (3)
    col_rg_ratio          mean(R)/(mean(G)+eps), mean over sampled frames.
    col_bg_ratio          mean(B)/(mean(G)+eps), mean over sampled frames.
    col_sat_std           std over pixels of saturation (max-min)/(max+eps).
  Temporal-lag differences, lags 1-4 (4)
    lag{k}_diff           mean |luma[t]-luma[t-k]| over all t (k=1,2,3,4).
  Per-pixel noise-floor statistics (2)
    noise_floor_med       median over pixels of the temporal std of first
                          differences (per-pixel sensor/compression noise floor).
    noise_floor_p90       90th percentile of the same per-pixel temporal std.
  Exposure clipping at thresholds (2)
    clip_low_5pct         fraction of pixels with luma<0.05 (mean over frames).
    clip_high_99          fraction of pixels with luma>0.99 (mean over frames).
  Edge / Laplacian statistics (2)
    edge_density          fraction of pixels with |Laplacian|>0.10 (mid frame).
    lap_mean_abs          mean |Laplacian| (mid frame).
  Hand-region occupancy (1)
    hand_occupancy        fraction of pixels in the YCrCb skin locus (mean over
                          sampled frames).

Not claimed: these are RGB features (no hyperspectral content), single labeller,
small OOD set. Hardware: CPU reference (also the edge/Pi path).
"""
from __future__ import annotations

import numpy as np

NEW_FEATURE_NAMES = [
    # Z3 transferable set
    "ct_fsize_std", "mo_hand_motion_excess", "inv_clip_frac",
    "inv_laplacian_var", "inv_hist_entropy", "inv_dct_hf_lf",
    # finer DCT/spectral bands
    "dct_band_r1", "dct_band_r2", "dct_band_r3", "dct_band_r4plus",
    # colour-channel ratios + colour statistics
    "col_rg_ratio", "col_bg_ratio", "col_sat_std",
    # temporal-lag differences (lags 1-4)
    "lag1_diff", "lag2_diff", "lag3_diff", "lag4_diff",
    # per-pixel noise-floor statistics
    "noise_floor_med", "noise_floor_p90",
    # exposure clipping at thresholds
    "clip_low_5pct", "clip_high_99",
    # edge / laplacian statistics
    "edge_density", "lap_mean_abs",
    # hand-region occupancy
    "hand_occupancy",
]
assert len(NEW_FEATURE_NAMES) == 24, len(NEW_FEATURE_NAMES)


def _laplacian(img):
    """4-neighbour Laplacian with edge-replicated borders."""
    p = np.pad(img, 1, mode="edge")
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * img)


def _dct8(img):
    """8x8 block DCT coefficients, returns (nb, 8, 8) float64."""
    from jadal.gpu.features import _DCT8
    H, W = img.shape
    H8, W8 = (H // 8) * 8, (W // 8) * 8
    im = img[:H8, :W8].reshape(H8 // 8, 8, W8 // 8, 8).transpose(0, 2, 1, 3)
    return np.einsum("ab,ijbc,cd->ijad", _DCT8, im, _DCT8.T)


def _radial_bands():
    r = np.sqrt((np.arange(8)[:, None] ** 2 + np.arange(8)[None, :] ** 2).astype(float))
    return {"r1": (r >= 1) & (r < 1.5), "r2": (r >= 1.5) & (r < 2.5),
            "r3": (r >= 2.5) & (r < 3.5), "r4plus": r >= 3.5,
            "lo": r <= 1.0, "hi": r >= 3.5}


_BANDS = _radial_bands()


def _skin_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    return (cr > 0.55) & (cr < 0.72) & (cb > 0.38) & (cb < 0.52)


def cpu_new_features(luma, rgb, fps=10.0, fsize_std=None):
    """luma: (T,H,W) float64 in [0,1]; rgb: (T,H,W,3) float64 in [0,1].

    fsize_std: optional pre-computed std/mean of per-frame compressed packet
    sizes (ffprobe). If None, ct_fsize_std is set to 0.0 and the caller is
    responsible for filling it in (see eval/gpu/run_features50.py).
    Returns {name: float} for the 24 NEW_FEATURE_NAMES.
    """
    luma = np.asarray(luma, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    T = luma.shape[0]
    mid = luma[T // 2]
    idx = np.linspace(0, T - 1, min(T, 8)).astype(int)

    # --- Z3 transferable set ---
    ct_fsize_std = float(fsize_std) if fsize_std is not None else 0.0
    diff1 = np.abs(np.diff(luma, axis=0)) if T > 1 else np.zeros((1,) + luma.shape[1:])
    glob_motion = float(diff1.mean())
    hand = 0.0
    if T > 1:
        sm = _skin_mask(rgb[idx])
        wsum = sm.sum()
        if wsum > 0:
            d = np.abs(rgb[idx[1:]] - rgb[idx[:-1]]).mean(axis=-1)
            hand = float((d * sm[1:]).sum() / max(wsum, 1.0))
    mo_hand_motion_excess = hand - glob_motion
    inv_clip_frac = float(np.mean([((luma[i] < 0.02) | (luma[i] > 0.98)).mean()
                                   for i in idx]))
    lap = _laplacian(mid)
    inv_laplacian_var = float(lap.var())
    h, _ = np.histogram(mid, bins=32, range=(0.0, 1.0))
    p = h / max(h.sum(), 1)
    nz = p[p > 0]
    inv_hist_entropy = float(-(nz * np.log2(nz)).sum())
    D = _dct8(mid)
    E = D ** 2
    tot = E.sum()
    hi = float(E[..., _BANDS["hi"]].sum())
    lo = float(E[..., _BANDS["lo"]].sum())
    inv_dct_hf_lf = hi / lo if lo > 0 else 0.0

    # --- finer DCT bands ---
    bands = {}
    for nm in ("r1", "r2", "r3", "r4plus"):
        bands["dct_band_" + nm] = float(E[..., _BANDS[nm]].sum() / tot) if tot > 0 else 0.0

    # --- colour ratios + stats ---
    rs = rgb[idx]
    gmean = float(rs[..., 1].mean())
    col_rg_ratio = float(rs[..., 0].mean() / (gmean + 1e-9))
    col_bg_ratio = float(rs[..., 2].mean() / (gmean + 1e-9))
    mx = rs.max(axis=-1)
    mn = rs.min(axis=-1)
    col_sat_std = float(((mx - mn) / (mx + 1e-9)).std())

    # --- temporal lags ---
    lags = {}
    for k in (1, 2, 3, 4):
        lags[f"lag{k}_diff"] = float(np.abs(luma[k:] - luma[:-k]).mean()) if T > k else 0.0

    # --- per-pixel noise floor ---
    if T > 2:
        nf = diff1.std(axis=0).ravel()
        noise_floor_med = float(np.median(nf))
        noise_floor_p90 = float(np.percentile(nf, 90))
    else:
        noise_floor_med = noise_floor_p90 = 0.0

    # --- exposure clipping thresholds ---
    clip_low_5pct = float(np.mean([(luma[i] < 0.05).mean() for i in idx]))
    clip_high_99 = float(np.mean([(luma[i] > 0.99).mean() for i in idx]))

    # --- edge / laplacian ---
    edge_density = float((np.abs(lap) > 0.10).mean())
    lap_mean_abs = float(np.abs(lap).mean())

    # --- hand occupancy ---
    hand_occupancy = float(_skin_mask(rgb[idx]).mean())

    return {
        "ct_fsize_std": ct_fsize_std,
        "mo_hand_motion_excess": mo_hand_motion_excess,
        "inv_clip_frac": inv_clip_frac,
        "inv_laplacian_var": inv_laplacian_var,
        "inv_hist_entropy": inv_hist_entropy,
        "inv_dct_hf_lf": inv_dct_hf_lf,
        **bands,
        "col_rg_ratio": col_rg_ratio,
        "col_bg_ratio": col_bg_ratio,
        "col_sat_std": col_sat_std,
        **lags,
        "noise_floor_med": noise_floor_med,
        "noise_floor_p90": noise_floor_p90,
        "clip_low_5pct": clip_low_5pct,
        "clip_high_99": clip_high_99,
        "edge_density": edge_density,
        "lap_mean_abs": lap_mean_abs,
        "hand_occupancy": hand_occupancy,
    }
