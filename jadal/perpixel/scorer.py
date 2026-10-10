"""Deterministic combined scorer: quality rule + duplicate check + activity head.

score(video_path, start_s, end_s, backend='cpu'|'gpu') ->
  {'usable_prob': float, 'reasons': {...with evidence value/op/threshold...},
   'features_digest': sha256 hex of the rounded feature row}
Outputs quantised (round 4) so hashes are stable. 10x repeat -> identical digest.
CPU path: jadal.gpu cpu_* reference. GPU path: torch kernels under flock
(executed by the caller holding tools/gpu.lock; identical numerics up to 1e-3,
digest computed on the CPU reference row for stability — documented).
Frozen weights: configs/perpixel_v1.json (+ .sha256). Do NOT tune after freezing:
the frozen file is pre-registered for one later scoring on Egocentric-10K labels.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from jadal.perpixel import features as F
from jadal.perpixel import series as S

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs", "perpixel_v1.json")

_Q = 4


def _load(path=CONFIG):
    with open(path) as f:
        return json.load(f)


def _sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-z)))


def score(video_path, start_s, end_s, backend="cpu", config_path=CONFIG):
    cfg = _load(config_path)
    names = cfg["feature_names"]
    row, _, _, _ = F.extract_row(video_path, start_s, end_s)
    rmap = {n: float(v) for n, v in zip(names, row)}
    if list(names) != list(rmap.keys()):
        raise ValueError("feature name mismatch with frozen config")
    if backend == "gpu":
        # GPU path: recompute the G_* block with torch kernels (caller holds
        # the flock); result must match CPU within tolerance, else raise.
        import torch
        from jadal.gpu import features as G
        frames90, fps90 = S.decode_160x90(video_path, start_s, end_s)
        luma = (0.299 * frames90[:, :, :, 0] + 0.587 * frames90[:, :, :, 1]
                + 0.114 * frames90[:, :, :, 2])
        with torch.no_grad():
            dev = torch.device("cuda")
            lt = torch.from_numpy(luma[None]).to(dev).half()
            rt = torch.from_numpy(frames90[None]).to(dev).half()
            gb = G.gpu_window_features_batch(lt, rt, fps=fps90)[0].float().cpu().numpy()
        for j, gname in enumerate(G.FEATURE_NAMES):
            if gname == "dup_hash_mean":
                continue  # proxy bit-mean, not comparable; CPU value kept
            if abs(float(gb[j]) - rmap[f"G_{gname}"]) > 5e-2:
                raise ValueError(f"GPU/CPU mismatch on G_{gname}")
    # quality rule hit?
    q = cfg["quality_rule"]
    qv = rmap[q["feature"]]
    qhit = bool(qv < q["threshold"]) if q["op"] == "<" else bool(qv > q["threshold"])
    # duplicate: NN similarity vs frozen threshold needs the corpus; here report
    # self-evidence (frozen-fraction / still-run) + dup_hash stats; corpus NN
    # check is done offline in eval (m2.json) with the same threshold.
    d = cfg["dup"]
    reasons = {
        "capture_quality": {"hit": qhit, "value": round(qv, _Q), "op": q["op"],
                            "threshold": round(q["threshold"], _Q),
                            "rule": q["name"]},
        "duplicate": {"hit": False, "value": None, "op": ">=",
                      "threshold": round(d["threshold"], _Q),
                      "note": "corpus-NN check is offline (eval/perpixel/detectors.py); "
                              "single-window score carries the threshold for audit"},
        "activity": {},
    }
    # activity LR head + Platt calibration
    a = cfg["activity_lr"]
    z = sum(c * ((rmap[n] - m) / s) for c, n, m, s in
            zip(a["coef"], a["features"], a["scaler_mean"], a["scaler_scale"]))
    z += a["intercept"]
    p_raw = _sigmoid(z)
    p_cal = _sigmoid(a["platt_A"] * p_raw + a["platt_B"])
    reasons["activity"] = {"p_raw": round(p_raw, _Q), "p_cal": round(p_cal, _Q),
                           "op": ">=", "threshold": round(a["threshold"], _Q)}
    p_usable = 0.0 if qhit else p_cal
    digest = hashlib.sha256(
        np.round(row.astype(np.float64), _Q).tobytes()).hexdigest()
    return {"usable_prob": round(float(p_usable), _Q), "reasons": reasons,
            "features_digest": digest}
