"""Integration scorer: the interface the PoC swaps in.

    score(video_path, start_s, end_s, config_path=None) -> dict

Returns at least
    {"usable_prob": float (rounded 4), "verdict": "accept|reject|review",
     "reasons": [{"code", "evidence": {"feature", "value", "op", "threshold"}}],
     "features_digest": "<sha256 hex>", "versions": {...}, "margin": float}

The score is the mean logit of the selected ensemble members, passed through a
sigmoid. The verdict comes from an abstain band: outside it we accept or reject,
inside it we route to REVIEW. Quality tripwires (dark / over-exposed / frozen /
undecodable) are exact rules evaluated on the SAME representation and take
precedence, because those failures are physical and cheap to establish.

Determinism: the CPU path is numpy-only and the tensor is quantised to 1e-4
before hashing, so repeated runs on the same machine give identical digests and
verdicts. The GPU path is not claimed bit-identical to the CPU path; any window
where they disagree is listed by eval/rhythm/determinism.py.

WHAT IS NOT DONE:
  * The weights are loaded from a frozen .npz produced by eval/rhythm/freeze.py.
    Without that file the scorer raises rather than guessing.
  * No streaming path (2-second chunks) is implemented here; the scorer consumes
    a whole window.
  * No hand detector. "Hands visible" is inferred from residual motion and skin
    chromaticity, not from a landmark model.
  * Thresholds are calibrated on the 150 human windows and are NOT certified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from jadal.rhythm import features as F

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rhythm_v1.json"

# exact physical-quality tripwires on the representation (not learned)
TRIPWIRES = (
    # (feature key, op, threshold, reason code)
    ("luma_mean", "<", 0.10, "TOO_DARK"),
    ("under_frac", ">", 0.70, "TOO_DARK"),
    ("over_frac", ">", 0.35, "OVEREXPOSED"),
    ("frozen_frac", ">", 0.95, "NO_ACTIVE_WORK"),
)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def load_config(config_path: Optional[str | Path] = None) -> dict:
    p = Path(config_path) if config_path else DEFAULT_CONFIG
    return json.loads(Path(p).read_text())


def _tripwire_features(X: np.ndarray, g: np.ndarray) -> dict:
    """Scalar features used by the exact rules, from the representation."""
    luma = float(g[:, 0].mean())
    under = float(g[:, 5].mean())
    over = float(g[:, 6].mean())
    # "frozen": fraction of frame pairs whose residual is below a tiny floor
    res = X[:, :, :, 0].mean(axis=(1, 2))          # (T,)
    if len(res) > 1:
        frozen_frac = float((res[1:] < 1e-4).mean())
    else:
        frozen_frac = 0.0
    return {"luma_mean": round(luma, 4), "under_frac": round(under, 4),
            "over_frac": round(over, 4), "frozen_frac": round(frozen_frac, 4)}


def _model_features(X: np.ndarray, g: np.ndarray, G: int) -> np.ndarray:
    """The exact feature vector the frozen head consumes.

    Kept deliberately simple and documented: 4 pooled statistics per grid channel
    (mean, std, max over cells of the temporal mean, and the concentration ratio)
    plus 4 statistics per global channel. The temporal conv encoder of the
    trained model is NOT reproduced here -- this CPU path is the reference
    fallback and is expected to score slightly below the torch path; the report
    states both.
    """
    T, GG, _, C = X.shape
    feats = []
    for c in range(C):
        ch = X[:, :, :, c]
        cell = ch.mean(axis=0).reshape(-1)
        feats += [float(cell.mean()), float(cell.std()), float(cell.max()),
                  float(cell.max() / (cell.sum() + 1e-6))]
    for c in range(g.shape[1]):
        col = g[:, c]
        feats += [float(col.mean()), float(col.std()),
                  float(np.percentile(col, 10)), float(np.percentile(col, 90))]
    return np.asarray(feats, np.float32)


def score(video_path: str, start_s: float, end_s: float,
          config_path: Optional[str | Path] = None) -> dict:
    """Score one window. See module docstring for the return contract."""
    cfg = load_config(config_path)
    G = int(cfg.get("G", 6))
    out = F.extract_window(video_path, start_s, end_s, G=G,
                           ego_compensate=bool(cfg.get("ego_compensate", True)),
                           use_flow=True)
    X, g = out["X"], out["g"]
    digest = F.tensor_digest(X, g)

    trip = _tripwire_features(X, g)
    reasons = []
    for key, op, thr, code in TRIPWIRES:
        v = trip[key]
        fired = (v < thr) if op == "<" else (v > thr)
        if fired:
            reasons.append({"code": code,
                            "evidence": {"feature": key, "value": v, "op": op, "threshold": thr}})
    if reasons:
        return {"usable_prob": 0.0, "verdict": "reject", "reasons": reasons,
                "features_digest": digest, "margin": 0.0,
                "features": trip,
                "versions": {"pipeline": "rhythm_v1", "path": "tripwire",
                             "config_sha256": cfg.get("sha256")}}

    wpath = Path(cfg["weights"]) if cfg.get("weights") else None
    if wpath is None or not Path(wpath).exists():
        raise FileNotFoundError(
            "rhythm weights not found; run eval/rhythm/freeze.py first "
            f"(config says {wpath})")
    z = np.load(wpath, allow_pickle=False)
    mu, sd, w, b = z["mu"], z["sd"], z["w"], z["b"]
    x = _model_features(X, g, G)
    logit = float(((x - mu) / (sd + 1e-6)) @ w + b)
    p = _sigmoid(logit)
    margin = abs(p - 0.5)
    lo, hi = float(cfg.get("accept_threshold", 0.65)), float(cfg.get("reject_threshold", 0.35))
    verdict = "accept" if p >= lo else ("reject" if p <= hi else "review")
    if verdict == "review":
        reasons.append({"code": "LOW_CONFIDENCE",
                        "evidence": {"feature": "usable_prob", "value": round(p, 4),
                                     "op": "in", "threshold": f"({hi}, {lo})"}})
    return {"usable_prob": round(p, 4), "verdict": verdict, "reasons": reasons,
            "features_digest": digest, "margin": round(margin, 4),
            "features": {**trip, "logit": round(logit, 4)},
            "versions": {"pipeline": "rhythm_v1", "path": "linear-cpu",
                         "weights": Path(wpath).name,
                         "weights_sha256": hashlib.sha256(Path(wpath).read_bytes()).hexdigest()[:16],
                         "config_sha256": cfg.get("sha256")}}
