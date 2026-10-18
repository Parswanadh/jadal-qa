"""S0 exact rules + S1 logistic head for the PoC.

Operating point (frozen from eval/results/gpu/cascade.json):
  threshold = 0.6 (Youden on LOCO out-of-fold p), margin = 0.25.
  |p-0.5| < margin -> 'review' (stage S2: handed on, heavy tier NOT run).

Head: StandardScaler + LogisticRegression(max_iter=2000) trained on ALL 150
IndEgo gold windows from eval/results/gpu/features.json (LOCO-free full fit;
LOCO was only for reporting AUROC). No frozen-coefficients file exists in the
repo, so the head is retrained deterministically in-process and the sha256 of
(canonical JSON of {coef, intercept, scaler_mean, scaler_scale, feature_names,
threshold, margin}) is recorded in every window's `versions.head_sha256`.

DOMAIN SHIFT (documented, binding): the head was calibrated on 480p/10fps
IndEgo. Egocentric-10K input is 1080p/30fps HEVC. Input is normalised before
scoring: scaled to the same 160x90 working resolution and temporally
subsampled to an effective 10fps (every 3rd frame) for the temporal/spectral
features, while ALL decoded frames feed the S0 probe/noise stats where cheap.
Verdicts here are descriptive, not accuracy.
"""
import hashlib
import json
import os

import numpy as np

from jadal.gpu.features import FEATURE_NAMES, cpu_window_features
from jadal.poc import s0_blur

FEATURES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "eval", "results", "gpu", "features.json")
CASCADE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "eval", "results", "gpu", "cascade.json")

# (rule_code, s0_key, op, threshold) — same values as eval/gpu/cascade.py S0_RULES
# for the five shipped rules; GRID.09 appends the brightness-calibrated blur rule
# LAST, so the five shipped rules keep priority (a black clip reports BLACK, not
# BLUR).
#
# SHIPPED VERDICT PATH: unchanged.  The BLUR entry is present but DISABLED, because
# s0_blur.BLUR_LAPVAR_CAL_MIN ships 0.0 and `lapvar_cal < 0.0` can never fire.
# Flipping that one constant to BLUR_LAPVAR_CAL_MIN_CALIBRATED enables the rule at
# the operating point measured in eval/results/exp_grid/GRID.09.json.  The probe
# gained `lapvar_cal`/`lapvar_cal_45` (observability only); the five shipped keys
# are arithmetically identical to before.
S0_RULES = [
    ("BLACK", "p95", "<", 0.03),
    ("TOO_DARK", "mean", "<", 0.05),
    ("LENS_BLOCKED", "spatial_var", "<", 1e-5),
    ("OVEREXPOSED", "p5", ">", 0.70),
    ("FROZEN", "temp_e", "<", 5e-4),
    s0_blur.BLUR_RULE,
]

# S0 probe keys -> window-feature names for the evidence block
S0_FEATURE_NAME = {"p95": "luma_p95", "mean": "luma_mean",
                   "spatial_var": "spatial_var", "p5": "luma_p05",
                   "temp_e": "temp_e", "lapvar_cal": s0_blur.BLUR_EVIDENCE_NAME}

THRESHOLD = 0.6
MARGIN = 0.25

# Normalisation targets (match IndEgo training conditions)
WORK_W, WORK_H = 160, 90
EFFECTIVE_FPS = 10.0
TEMPORAL_STRIDE = 3  # 30fps -> every 3rd frame ~= 10fps effective


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def s0_decide(s0):
    """Exact rules on the 2-s tiny probe. Returns (rule_code, evidence) or (None, None)."""
    for rule, key, op, thr in S0_RULES:
        if key not in s0:
            continue
        val = float(s0[key])
        if (val < thr) if op == "<" else (val > thr):
            return rule, {"feature": S0_FEATURE_NAME[key], "value": round(val, 4),
                          "op": op, "threshold": thr}
    return None, None


def train_head():
    """Deterministic full-fit head from features.json. Returns (clf, scaler, head_sha)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    feats = json.load(open(FEATURES_JSON))["windows"]
    casc = json.load(open(CASCADE_JSON))
    thr = float(casc["loco"]["threshold"])
    assert thr == THRESHOLD, f"cascade threshold moved: {thr}"
    try:
        gold = json.load(open("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"))["items"]
    except FileNotFoundError:
        gold = json.load(open(os.path.expanduser("~") + "/projects/jadal-qa-wt/spectral_data/manifest_gold.json"))["items"]
    ids = [it["id"] for it in gold if it["id"] in feats and "full" in feats[it["id"]]]
    X = np.array([[feats[i]["full"][n] for n in FEATURE_NAMES] for i in ids])
    y = np.array([next(it["label"] for it in gold if it["id"] == i) for i in ids])
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X), y)
    head_doc = {"coef": [round(float(v), 6) for v in clf.coef_[0]],
                "intercept": round(float(clf.intercept_[0]), 6),
                "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
                "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
                "feature_names": FEATURE_NAMES,
                "threshold": THRESHOLD, "margin": MARGIN, "n_train": len(ids)}
    head_sha = hashlib.sha256(canonical(head_doc)).hexdigest()
    return clf, scaler, head_sha, head_doc


def s1_score(feats, clf, scaler):
    v = np.array([[feats[n] for n in FEATURE_NAMES]])
    return float(clf.predict_proba(scaler.transform(v))[0, 1])


def decide_window(s0, feats, clf, scaler):
    """Returns (verdict, stage, reasons, score_or_None)."""
    rule, ev = s0_decide(s0)
    if rule is not None:
        return ("reject", "S0",
                [{"code": rule, "evidence": ev}], None)
    p = s1_score(feats, clf, scaler)
    if abs(p - 0.5) < MARGIN:
        return ("review", "S2",
                [{"code": "UNCERTAIN",
                  "evidence": {"feature": "s1_score", "value": round(p, 4),
                               "op": "in", "threshold": [round(0.5 - MARGIN, 2),
                                                         round(0.5 + MARGIN, 2)]}}],
                round(p, 4))
    if p >= THRESHOLD:
        return ("accept", "S1",
                [{"code": "S1_USABLE",
                  "evidence": {"feature": "s1_score", "value": round(p, 4),
                               "op": ">=", "threshold": THRESHOLD}}],
                round(p, 4))
    return ("reject", "S1",
            [{"code": "S1_UNUSABLE",
              "evidence": {"feature": "s1_score", "value": round(p, 4),
                           "op": "<", "threshold": THRESHOLD}}],
            round(p, 4))


def window_features_full_rate(luma, rgb):
    """Normalise 30fps input to the head's 10fps training condition.

    luma: (T,H,W), rgb: (T,H,W,3) at WORK_W x WORK_H. Temporal/spectral
    features use every TEMPORAL_STRIDE-th frame (effective 10fps); all
    frames were still decoded and feed S0/noise stats upstream.
    """
    return cpu_window_features(luma[::TEMPORAL_STRIDE], rgb[::TEMPORAL_STRIDE],
                               fps=EFFECTIVE_FPS)


def cpu_window_features_alias(luma_full, sub_f64, _rgb_full=None):
    """Window features on boxcar-subsampled frames (anti-aliased 10fps).

    sub_f64: (100,H,W,3) float64 in [0,1]. luma_full is unused (kept for
    call-site clarity); luma is re-derived from sub for consistency.
    """
    luma_sub = (0.299 * sub_f64[..., 0] + 0.587 * sub_f64[..., 1]
                + 0.114 * sub_f64[..., 2])
    return cpu_window_features(luma_sub, sub_f64, fps=EFFECTIVE_FPS)
