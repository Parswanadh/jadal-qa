"""Deterministic profile router (E4.04).

choose_profile(descriptors) -> dict(profile, margin, reasons, unseen_flag).
Rule-based, thresholds calibrated on E4.01 data (n=670 windows: 150 IndEgo
gold + 60 OOD paired transcode/source + 100 queue-2 + 300 Ego10K sample;
see eval/results/exp_regimes/diagnosis.json).
Falls back to 'default' + UNSEEN_REGIME when descriptors are out of training range.
The decision (profile + margin + descriptors sha) is part of the receipt.
"""
from __future__ import annotations

# E4.01 calibration notes (combined n=670 unless noted):
#  sharp_norm: IndEgo p10=4345, OOD-src p33=2394 -> lo=2500 catches the soft tail
#  luma_mean: datasets separate at ~0.40 (IndEgo p90=0.387, OOD p10=0.407);
#             within-480p darkness = IndEgo p10 -> lo=0.24
#  clip_frac: overlapping (dark-driven on IndEgo, bright-driven on OOD);
#             hi=0.30 catches only the overexposed tail
#  shift_mean: medians 15-19 everywhere at 1.25 s sampling; walking tail at
#             IndEgo p66=27.5 -> hi=28; temp_energy>=0.26 (OOD p90) as co-trigger
THRESHOLDS = {
    # rule boundaries (E4.01-calibrated v0.2.0)
    "hi_res_width": 1280,      # >= -> factory_1080p30
    "hi_fps": 25.0,            # >= -> factory_1080p30
    "motion_shift_hi": 28.0,   # shift_mean >= -> walking_logistics
    "motion_temp_hi": 0.26,    # temp_energy >= -> walking_logistics
    "sharp_norm_lo": 2500.0,   # sharp_norm <= -> blurry_lowlight
    "luma_lo": 0.24,           # luma_mean <= -> blurry_lowlight
    "clip_hi": 0.30,           # clip_frac >= -> blurry_lowlight
}

# Observed [min, max] over the E4.01 window pool (rounded outward).
# A window outside ANY of these is UNSEEN_REGIME -> 'default'.
TRAINING_RANGES = {
    "width": [480, 1920],
    "height": [480, 1080],
    "fps": [10.0, 30.0],
    "sharp_norm": [150.0, 55000.0],
    "luma_mean": [0.05, 0.65],
    "shift_mean": [0.0, 230.0],
    "noise_sigma": [0.0, 0.015],
}

KEYS = ("width", "fps", "shift_mean", "static_frac",
        "sharp_norm", "luma_mean", "clip_frac")


def _get(d, k, default=0.0):
    try:
        return float(d.get(k, default))
    except (TypeError, ValueError):
        return float(default)


def _margin(dist, scale):
    return round(float(dist) / scale, 4) if scale else 0.0


def choose_profile(d):
    """Pick a profile name from a descriptor dict. Pure function, deterministic."""
    reasons = []
    unseen = []
    for k, (lo, hi) in TRAINING_RANGES.items():
        v = _get(d, k, None if k in d else lo)
        if k in d and not (lo <= _get(d, k) <= hi):
            unseen.append(f"{k}={_get(d, k):.4g} outside [{lo},{hi}]")
    if unseen:
        return {"profile": "default", "margin": 0.0,
                "reasons": ["UNSEEN_REGIME: " + "; ".join(unseen)],
                "unseen_regime": True}

    width = _get(d, "width")
    fps = _get(d, "fps")
    shift = _get(d, "shift_mean")
    temp = _get(d, "temp_energy")
    sharp = _get(d, "sharp_norm")
    luma = _get(d, "luma_mean")
    clip = _get(d, "clip_frac")

    if width >= THRESHOLDS["hi_res_width"] or fps >= THRESHOLDS["hi_fps"]:
        m = min(_margin(width - THRESHOLDS["hi_res_width"] + 1, 640),
                _margin(fps - THRESHOLDS["hi_fps"], 5))
        reasons.append(f"hires: width={width:.0f} fps={fps:.1f}")
        return {"profile": "factory_1080p30", "margin": max(m, 0.0),
                "reasons": reasons, "unseen_regime": False}

    margins = []
    if shift >= THRESHOLDS["motion_shift_hi"] or temp >= THRESHOLDS["motion_temp_hi"]:
        margins.append(_margin(shift - THRESHOLDS["motion_shift_hi"], 28.0))
        margins.append(_margin(temp - THRESHOLDS["motion_temp_hi"], 0.26))
        reasons.append(f"motion: shift_mean={shift:.2f} temp_energy={temp:.3f}")
        return {"profile": "walking_logistics",
                "margin": max([m for m in margins] + [0.0]),
                "reasons": reasons, "unseen_regime": False}

    if (sharp <= THRESHOLDS["sharp_norm_lo"] or luma <= THRESHOLDS["luma_lo"]
            or clip >= THRESHOLDS["clip_hi"]):
        margins.append(_margin(THRESHOLDS["sharp_norm_lo"] - sharp, 2500.0))
        margins.append(_margin(THRESHOLDS["luma_lo"] - luma, 0.24))
        margins.append(_margin(clip - THRESHOLDS["clip_hi"], 0.30))
        reasons.append(
            f"quality: sharp_norm={sharp:.1f} luma={luma:.3f} clip={clip:.3f}")
        return {"profile": "blurry_lowlight",
                "margin": max([m for m in margins] + [0.0]),
                "reasons": reasons, "unseen_regime": False}

    m = min(_margin(sharp - THRESHOLDS["sharp_norm_lo"], 2500.0),
            _margin(luma - THRESHOLDS["luma_lo"], 0.24),
            _margin(THRESHOLDS["motion_shift_hi"] - shift, 28.0))
    reasons.append(f"clear: sharp_norm={sharp:.1f} luma={luma:.3f} shift={shift:.2f}")
    return {"profile": "clear_stationary", "margin": max(m, 0.0),
            "reasons": reasons, "unseen_regime": False}


def receipt_fragment(d, decision):
    """Evidence fragment for the QA receipt."""
    return {"profile": decision["profile"],
            "router_margin": decision["margin"],
            "router_reasons": decision["reasons"],
            "unseen_regime": decision["unseen_regime"],
            "descriptors_sha": d.get("_sha256")}
