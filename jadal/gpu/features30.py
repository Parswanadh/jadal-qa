"""Per-pixel feature set v30 = the 26 E5 pixel features + 4 transferable
invariants from the Z3 domain-agnostic family.

The 4 added features are decode-based, per-pixel and cheap. They are exactly
the Z3 invariants that classified **stable** IndEgo -> Egocentric-10K in the
Z3.01/160 transfer table (branch m/new-features-160); the two remaining Z3
"transferable" candidates (``ct_fsize_std``, ``mo_hand_motion_excess``) are
container-telemetry / multi-frame motion and are NOT per-pixel, so they are not
part of this vector.

New features (name, definition, cost):

  * ``inv_clip_frac``     Fraction of luma pixels clipped at the exposure rails
                          (luma <= 2/255 or luma >= 253/255), averaged over the
                          decoded frames. Cost: one O(HW) comparison per frame.
  * ``inv_laplacian_var`` Variance of the 3x3 Laplacian of luma (sharpness /
                          focus / blur cue), averaged over frames.
                          Cost: one O(HW) cv2.Laplacian per frame.
  * ``inv_hist_entropy``  Shannon entropy (nats) of the 32-bin luma histogram
                          (0..1), averaged over frames. Cost: one O(HW) histogram
                          per frame.
  * ``inv_dct_hf_lf``     DCT high-band (radius > 20) / low-band (radius < 8)
                          energy ratio of luma resized to 64x64, averaged over
                          frames (fine spatial-frequency balance).
                          Cost: one 64x64 DCT per frame.

All four match the reference definitions in
``eval/exp_features/extract.py::_invariants_step`` (the Z3 extractor), so the
cached Z3 values can be reused for the existing gold / OOD / hold-out windows.
``window_invariants`` averages over frames ``0 .. T-2`` (the Z3 loop is over
consecutive frame pairs), giving an apples-to-apples 30-vector.
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES, cpu_window_features

# The four decode-based invariants appended to the 26 E5 pixel features.
NEW_FEATURES = ["inv_clip_frac", "inv_laplacian_var", "inv_hist_entropy",
                "inv_dct_hf_lf"]

FEATURE_NAMES30 = list(FEATURE_NAMES) + list(NEW_FEATURES)


def _luma(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def invariants_step(rgb):
    """One-frame invariants. ``rgb``: (H,W,3) float64 in [0,1].

    Returns the 4-feature dict. Definitions are byte-compatible with the Z3
    extractor (``_invariants_step``), so cached values reproduce exactly.
    """
    import cv2
    g = _luma(rgb)
    x = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    xg = _luma(x)
    d = cv2.dct(xg.astype(np.float32))
    yy, xx = np.mgrid[0:64, 0:64]
    r = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
    hf_lf = float((d[r > 20] ** 2).sum() / ((d[r < 8] ** 2).sum() + 1e-9))
    lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    lap_var = float(lap.var())
    clip = float(((g <= 2 / 255.0) | (g >= 253 / 255.0)).mean())
    h, _ = np.histogram(g, bins=32, range=(0, 1))
    p = h / max(h.sum(), 1)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {"inv_clip_frac": clip, "inv_laplacian_var": lap_var,
            "inv_hist_entropy": ent, "inv_dct_hf_lf": hf_lf}


def window_invariants(rgb):
    """Average the 4 invariants over frames ``0 .. T-2`` (Z3 pair loop).

    ``rgb``: (T,H,W,3) float64 in [0,1]. Returns {name: float}.
    """
    T = rgb.shape[0]
    if T < 2:
        return {k: 0.0 for k in NEW_FEATURES}
    acc = {k: 0.0 for k in NEW_FEATURES}
    for t in range(T - 1):
        inv = invariants_step(rgb[t])
        for k in NEW_FEATURES:
            acc[k] += inv[k]
    n = float(T - 1)
    return {k: acc[k] / n for k in NEW_FEATURES}


def cpu_window_features30(luma, rgb, fps=30.0):
    """30-feature per-pixel vector. luma: (T,H,W), rgb: (T,H,W,3) float64 [0,1]."""
    out = cpu_window_features(luma, rgb, fps=fps)
    out.update(window_invariants(rgb))
    return {k: out[k] for k in FEATURE_NAMES30}
