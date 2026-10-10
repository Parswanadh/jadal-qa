"""Deterministic combined scorer: theory rule list + quality rule + activity head.

score(video_path, start_s, end_s, backend='cpu') ->
  {'usable_prob', 'decision', 'reasons': {...with evidence value/op/threshold
   and margin...}, 'features_digest'}.

Decision order (all comparisons on QUANTISED integers, fixed-point Q=4):
  1. theory rule list (stage 1 capture, then stage 3 activity) -> BAD + evidence
  2. M1 quality rule (refit-all) -> BAD + evidence
  3. else activity LR head (Platt-calibrated); usable_prob = p_cal,
     BAD rules force 0.0. Duplicate threshold carried for audit; corpus-NN
     check stays offline (eval/perpixel/detectors.py) with the same threshold.
Outputs rounded to 4 decimals so hashes are stable. 10x repeat -> identical.
CPU path is the reference; GPU path recomputes the G_* block with torch
kernels (caller holds tools/gpu.lock) and must match within tolerance.
Frozen weights: configs/perpixel_v1.json (+ .sha256). Do NOT tune after
freezing: pre-registered for one later scoring on Egocentric-10K labels.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from jadal.perpixel import features as F
from jadal.perpixel import physics as P
from jadal.perpixel import series as S
from jadal.perpixel import theory as T

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs", "perpixel_v1.json")

_Q = 4
_QI = 10 ** _Q
_MISSING = -99999999  # fixed sentinel for NaN physics (documented)


def _load(path=CONFIG):
    with open(path) as f:
        return json.load(f)


def _sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-z)))


def _qi(x: float) -> int:
    if not np.isfinite(x):
        return _MISSING
    return int(round(float(x) * _QI))


def score(video_path, start_s, end_s, backend="cpu", config_path=CONFIG):
    cfg = _load(config_path)
    row, names, _, meta, frames90, fps90 = F.extract_row(
        video_path, start_s, end_s, return_frames=True)
    rmap = {n: float(v) for n, v in zip(names, row)}
    th = T.extract_theory(frames90, fps90)
    thq = T.quantise(th)
    ph = P.extract_physics(frames90, fps90, video_path, start_s, end_s)
    if backend == "gpu":
        import torch
        from jadal.gpu import features as G
        luma = (0.299 * frames90[..., 0] + 0.587 * frames90[..., 1]
                + 0.114 * frames90[..., 2])
        with torch.no_grad():
            dev = torch.device("cuda")
            lt = torch.from_numpy(luma[None]).to(dev).half()
            rt = torch.from_numpy(frames90[None]).to(dev).half()
            gb = G.gpu_window_features_batch(lt, rt, fps=fps90)[0].float(
                ).cpu().numpy()
        for j, gname in enumerate(G.FEATURE_NAMES):
            if gname == "dup_hash_mean":
                continue
            if abs(float(gb[j]) - rmap[f"G_{gname}"]) > 5e-2:
                raise ValueError(f"GPU/CPU mismatch on G_{gname}")
    reasons = {}
    bad_hit = None
    # 1. theory rule list (integer compare)
    for r in cfg["theory_rules"]:
        v = thq[r["feature"]]
        t = int(round(float(r["threshold"]) * _QI))
        hit = bool(v < t) if r["op"] == "<" else bool(v > t)
        margin_units = (t - v) / _QI if r["op"] == "<" else (v - t) / _QI
        reasons[r["name"]] = {"hit": hit, "value": v / _QI, "op": r["op"],
                              "threshold": t / _QI,
                              "margin_units": round(margin_units, _Q),
                              "feature": r["feature"]}
        if hit and bad_hit is None:
            bad_hit = r["name"]
    # 2. M1 quality rule
    q = cfg["quality_rule"]
    qv = rmap[q["feature"]]
    qhit = bool(qv < q["threshold"]) if q["op"] == "<" else bool(
        qv > q["threshold"])
    reasons["m1_quality"] = {"hit": qhit, "value": round(qv, _Q),
                             "op": q["op"],
                             "threshold": round(q["threshold"], _Q),
                             "rule": q["name"]}
    if qhit and bad_hit is None:
        bad_hit = "m1_quality:" + q["name"]
    # 3. activity LR head
    a = cfg["activity_lr"]
    m1v = np.array([rmap[n] for n in names])
    if a["feature_set"] == "m1":
        Xv = m1v
    else:
        pv = np.array([ph.get(k, np.nan) for k in cfg["physics_features"]])
        pv = np.where(np.isfinite(pv), pv,
                      np.array([cfg["nan_fill"][k]
                                for k in cfg["physics_features"]]))
        tv = np.array([th[k] for k in cfg["theory_features"]])
        Xv = (np.concatenate([m1v, tv, pv]) if a["feature_set"] == "all"
              else np.concatenate([tv, pv]))
    z = float((((Xv - np.array(a["scaler_mean"]))
                / np.array(a["scaler_scale"])) * np.array(a["coef"])).sum()
              + a["intercept"])
    p_raw = _sigmoid(z)
    p_cal = _sigmoid(a["platt_A"] * p_raw + a["platt_B"])
    reasons["activity"] = {"p_raw": round(p_raw, _Q),
                           "p_cal": round(p_cal, _Q), "op": ">=",
                           "threshold": round(a["threshold"], _Q),
                           "head": "l2-logistic(" + a["feature_set"] + ")"}
    d = cfg["dup"]
    reasons["duplicate"] = {"hit": False, "op": ">=",
                            "threshold": round(d["threshold"], _Q),
                            "note": "corpus-NN offline; same threshold"}
    p_usable = 0.0 if bad_hit else p_cal
    blob = (np.round(row.astype(np.float64), _Q).tobytes()
            + np.array([thq[k] for k in cfg["theory_features"]],
                       dtype=np.int64).tobytes()
            + np.array([_qi(ph.get(k, float("nan")))
                        for k in cfg["physics_features"]],
                       dtype=np.int64).tobytes())
    digest = hashlib.sha256(blob).hexdigest()
    return {"usable_prob": round(float(p_usable), _Q),
            "decision": ("BAD:" + bad_hit) if bad_hit else "GOOD",
            "reasons": reasons, "features_digest": digest}
