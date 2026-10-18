"""GRID.05: the shipped S0 rules, recomputed on the MOSAIC canvas.

Why a separate probe
--------------------
The mosaic path scores ONE composite canvas instead of the frame stream, so the
shipped 2-second probe (``scripts/poc_run.py::s0_from_luma``) cannot be called on
it.  This module rebuilds the SAME probe dictionary from canvas statistics, so
``jadal.poc.cascade.s0_decide`` can consume it unchanged:

===============  ==========================  ====================================
probe key        shipped probe (80x45 grey)  mosaic canvas
===============  ==========================  ====================================
``mean``         mean over the 60 frames     mean over ALL pooled tile pixels
``p5``/``p95``   percentiles of the stack    percentiles of the pooled pixels
``spatial_var``  var of the temporal mean    var of the pooled pixels
``temp_e``       mean abs diff frame->frame  inter-tile MAD (``it_mad_mean``)
``lapvar_cal``   per-frame lapvar / mean     per-tile lapvar / tile mean (mean over tiles)
===============  ==========================  ====================================

Thresholds
----------
* BLACK / TOO_DARK / LENS_BLOCKED / OVEREXPOSED use the SHIPPED thresholds
  unchanged: the statistic is the same statistic (a luma percentile / a luma
  variance of the input), just sampled at N tile times instead of 60 frames, so
  inheriting the operating point is the whole point of "parity".
* FROZEN and BLUR are NOT inherited: the mosaic substitutes 224 px tiles for
  80x45 frames, which changes the units of ``temp_e`` and ``lapvar_cal``.  Both
  operating points are calibrated on CLEAN data only (the 2nd percentile of the
  clean CALIB split, i.e. the largest threshold whose clean-only false-reject
  rate on the calibration split is <= 2%) and frozen here as constants.  They are
  NOT comparable with the GRID.09 full-pipeline values and are never interchanged.

Timestamps are burned into every tile by the builder, so the standard
``features.tile_mask`` rectangle is masked out.  Letterbox padding is detected and
excluded, but NOT by ``features.letterbox_mask`` alone: that row test needs >=98 %
of a row to be mid-grey, and the burned timestamp box covers the top-right of the
padding on non-square sources, so those padding rows survive and their ~50 % luma
leaks into the pooled percentiles.  ``_strict_padding_mask`` runs the row test on
the columns outside the timestamp box instead; both poolings are reported in
GRID.05 (``padding_detect="strict"`` is the default, ``"features"`` is the
ablation).  Gutters are never inside a tile.

Determinism: pure float64 numpy, no sampling, no randomness.
"""
from __future__ import annotations

import numpy as np

from jadal.mosaic import features as F
from jadal.poc import s0_blur

#: the four rules whose statistic is unit-identical on both paths (inherited)
INHERITED_RULES = (
    ("BLACK", "p95", "<", 0.03),
    ("TOO_DARK", "mean", "<", 0.05),
    ("LENS_BLOCKED", "spatial_var", "<", 1e-5),
    ("OVEREXPOSED", "p5", ">", 0.70),
)

#: FROZEN on the mosaic: an inter-tile MAD below this is a static window.
#: Frozen 2026-10-18 from eval/results/exp_grid/grid05_calibration.json
#: (n_calib = 106 clean human-good windows, pooled over both datasets, strict
#: pooling).  0.0 = disabled, so an unfrozen checkout cannot reject anything.
MOSAIC_TEMP_E_MAX = 0.05470779972318289

#: BLUR on the mosaic: pooled per-tile ``lapvar_cal`` below this is blurred.
#: Frozen from the same calibration artifact with the same clean-only
#: 2nd-percentile rule.
MOSAIC_BLUR_LAPVAR_CAL_MIN = 0.013163918969293153

#: calibration provenance, asserted against grid05_calibration.json
FROZEN_FROM = "eval/results/exp_grid/grid05_calibration.json"


def mosaic_s0_rules():
    """The rule table in production order: the four inherited, then the two
    calibrated, blur last (so a black clip reports BLACK, never BLUR)."""
    return list(INHERITED_RULES) + [
        ("FROZEN", "temp_e", "<", MOSAIC_TEMP_E_MAX),
        ("BLUR", "lapvar_cal", "<", MOSAIC_BLUR_LAPVAR_CAL_MIN),
    ]


def decide_with(probe: dict, rules):
    """Apply an explicit rule table; same contract as ``cascade.s0_decide``."""
    for rule, key, op, thr in rules:
        if key not in probe:
            continue
        val = float(probe[key])
        if np.isnan(val):
            continue
        if (val < thr) if op == "<" else (val > thr):
            return rule, {"feature": key, "value": round(val, 6), "op": op,
                          "threshold": thr}
    return None, None


def mosaic_s0_decide(probe: dict):
    """Same contract as ``jadal.poc.cascade.s0_decide``: (rule_code, evidence)."""
    return decide_with(probe, mosaic_s0_rules())


def inherited_s0_decide(probe: dict):
    """The four inherited rules ONLY — the 'easy parity' half of the claim."""
    return decide_with(probe, INHERITED_RULES)


def _strict_padding_mask(rgb, base, tile_px):
    """Letterbox detection that still works underneath the burned timestamp.

    ``features.letterbox_mask`` drops a row only when >=98 % of that row is
    mid-grey.  The burned timestamp box sits in the TOP-RIGHT of every tile, so
    for a non-square source the padding rows it covers (padding is drawn first,
    then the box is composited on top) are only ~75 % grey and survive the test.
    Those rows are then ~50 %-luma padding that leaks straight into the pooled
    luma percentiles.

    Measured consequence (GRID.05, one of the twelve unlabelled demo-kit clips):
    a 640x480 all-black clip reads p95 = 0.502 and mean = 0.027 on the canvas,
    i.e. TOO_DARK, while the shipped probe reads p95 = 0.0000, i.e. BLACK.

    The fix is local: run the row test on the columns OUTSIDE the timestamp box,
    where padding rows really are ~100 % grey.  Column detection is unchanged
    (padding columns span the full tile height, timestamp box included).

    ``base`` must be ``features.tile_mask(tile_px)``: the timestamp rectangle
    stays masked out.  Passing an all-True mask here would put the burned white
    text back into the sharpness statistic (measured: mosaic ``lapvar_cal`` on a
    blurred clip went from 0.0010 to 0.0277), which is exactly the leak the mask
    exists to prevent.
    """
    bg = np.abs(np.asarray(rgb, np.float64) - F.BG_GREY).max(axis=2) <= 2.0
    x0, _y0, x1, _y1 = F._timestamp_box(int(tile_px))
    probe_cols = np.ones(bg.shape[1], bool)
    probe_cols[x0:x1] = False
    if probe_cols.sum() < 8:
        return F.letterbox_mask(rgb, base)
    rows = bg[:, probe_cols].mean(axis=1) >= 0.98
    cols = bg.mean(axis=0) >= 0.98
    m = base.copy()
    m[rows, :] = False
    m[:, cols] = False
    if m.sum() < 0.05 * m.size:
        return base
    return m


def pooled_stats(canvas, sidecar, *, mask_timestamp: bool = True,
                 padding_detect: str = "strict"):
    """Pooled pixel statistics over every tile of the canvas + per-tile values.

    ``padding_detect``:
      ``"strict"``   — :func:`_strict_padding_mask` (default; survives the
                       timestamp box, so 16:9 sources keep their padding out);
      ``"features"`` — S1's ``features.letterbox_mask`` unchanged, the ablation.

    Returns ``(pooled, per_tile, diag)``; ``per_tile`` is a list of dicts with the
    tile mean luma and the tile ``lapvar_cal`` (the same construction
    ``features.spatial_features`` uses, so ``lapvar_cal`` here equals
    ``sp_lapvar_cal_mean`` of the full feature vector for the same mask).
    """
    if padding_detect not in ("strict", "features"):
        raise ValueError("padding_detect must be 'strict' or 'features'")
    vals, per_tile = [], []
    n_unmasked_rows = 0
    base = F.tile_mask(int(sidecar["tile_px"])) if mask_timestamp else None
    for rgb, m, _t in F.tile_grid(np.asarray(canvas), sidecar):
        if not mask_timestamp:
            m = F.letterbox_mask(rgb, np.ones_like(m))
        elif padding_detect == "strict":
            # extend the standard mask; never replace it (the timestamp must stay out)
            m = _strict_padding_mask(rgb, base, sidecar["tile_px"])
        g = F.to_luma(rgb)
        v = g[m]
        if v.size < 16:
            continue
        vals.append(v)
        per_tile.append({
            "mean": float(v.mean()),
            "lapvar_cal": s0_blur.lapvar_cal(np.where(m, g, float(v.mean()))),
        })
        n_unmasked_rows += int(m.sum())
    if not vals:
        raise ValueError("no usable tile pixels on this canvas")
    pooled = np.concatenate(vals)
    diag = {
        "n_pixels": int(pooled.size),
        "n_pixels_used": int(n_unmasked_rows),
        # Share of the pooled sample whose LUMA is within +-2/255 of the mid-grey
        # letterbox value.  A DIAGNOSTIC, not a purity guarantee: content that is
        # genuinely mid-grey (or noisy content whose luma clusters near 0.5) also
        # counts.  It is ~0.09 for 16:9 padding leaks, ~0.012 for the same windows
        # with strict pooling, and ~0.08 for random noise on a square canvas.
        "bg_leak_frac": float((np.abs(pooled - F.BG_GREY / 255.0)
                               <= 2.0 / 255.0).mean()),
    }
    return pooled, per_tile, diag


def mosaic_probe(canvas, sidecar, *, mask_timestamp: bool = True,
                 padding_detect: str = "strict") -> dict:
    """The S0 probe dictionary rebuilt from the mosaic canvas.

    Keys match the shipped probe exactly (``mean``, ``p5``, ``p95``,
    ``spatial_var``, ``temp_e``, ``lapvar_cal``) so the same rule consumer works
    on both.  ``temp_e`` carries the inter-tile MAD, which is a *different
    statistic* from the shipped frame-to-frame temporal energy -- hence the
    separate calibrated threshold.
    """
    pooled, per_tile, diag = pooled_stats(canvas, sidecar,
                                          mask_timestamp=mask_timestamp,
                                          padding_detect=padding_detect)
    tiles = list(F.tile_grid(np.asarray(canvas), sidecar))
    it = F.intertile_features(tiles)
    return {
        "mean": float(pooled.mean()),
        "p5": float(np.percentile(pooled, 5)),
        "p95": float(np.percentile(pooled, 95)),
        "spatial_var": float(pooled.var()),
        "temp_e": float(it["it_mad_mean"]),
        "lapvar_cal": float(np.mean([t["lapvar_cal"] for t in per_tile])),
        # provenance, not part of the shipped probe's five keys
        "padding_detect": padding_detect,
        "n_pooled_px": diag["n_pixels"],
        "bg_leak_frac": diag["bg_leak_frac"],
        "n_tiles_used": len(per_tile),
        "tile_lapvar_cal_min": float(np.min([t["lapvar_cal"] for t in per_tile])),
        "it_frozen_frac": float(it["it_frozen_frac"]),
        "it_mad_min": float(it["it_mad_min"]),
        "it_below005_frac": float(it["it_below005_frac"]),
    }
