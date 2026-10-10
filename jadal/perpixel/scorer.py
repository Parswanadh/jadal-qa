"""Deterministic scorer v2: quality tripwires + conv activity score.

score(video_path, start_s, end_s) ->
  {'usable_prob', 'decision', 'reasons': {evidence value/op/threshold...},
   'features_digest'}.

Stage 1 (deterministic rules, integer compare at Q=4): OR of frozen quality
tripwires {lowent, blur, ten} -> BAD:<rule> with evidence.
Stage 2: conv1d activity score on the per-frame series (frozen weights,
seed 0, CPU, 1 thread); usable_prob = round4(p). Quality hit forces 0.0.
Duplicate threshold carried for audit; corpus-NN check stays offline.
Digest = sha256 over round-4 M1 row (EXCLUDING B_mv_per_frame: ffmpeg MV
debug counts vary run to run) + theory fixed-point ints + round-4 physics
(NaN -> sentinel) + round-4 conv prob. 10x repeat -> identical.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from jadal.perpixel import features as F
from jadal.perpixel import physics as P
from jadal.perpixel import series as S
from jadal.perpixel import theory as T

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.normpath(os.path.join(HERE, "..", "..", "configs",
                                       "perpixel_v1.json"))
CONV = os.path.normpath(os.path.join(HERE, "..", "..", "configs",
                                     "perpixel_conv.pt"))

_Q = 4
_QI = 10 ** _Q
_MISSING = -99999999


def _load(path=CONFIG):
    with open(path) as f:
        return json.load(f)


def _qi(x: float) -> int:
    if not np.isfinite(x):
        return _MISSING
    return int(round(float(x) * _QI))


def _conv_score(Smat: np.ndarray, fps: float) -> float:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    ck = torch.load(CONV, map_location="cpu", weights_only=False)
    net = torch.nn.Sequential(
        torch.nn.Conv1d(8, 16, 5, padding=2), torch.nn.ReLU(),
        torch.nn.Conv1d(16, 16, 5, padding=2), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
        torch.nn.Linear(16, 1))
    net.load_state_dict(ck["state"])
    net.eval()
    Xa = np.resize(Smat.astype(np.float64), (ck["T"], 8))
    Xt = torch.tensor((Xa - ck["mu"]) / ck["sd"], dtype=torch.float32)
    with torch.no_grad():
        p = torch.sigmoid(net(Xt[None].transpose(1, 2))).numpy().ravel()[0]
    return float(p)


def score(video_path, start_s, end_s, config_path=CONFIG):
    cfg = _load(config_path)
    row, names, Smat, meta, frames90, fps90 = F.extract_row(
        video_path, start_s, end_s, return_frames=True)
    Smat64 = S.per_frame_series(frames90)
    th = T.extract_theory(frames90, fps90)
    thq = T.quantise(th)
    ph = P.extract_physics(frames90, fps90, video_path, start_s, end_s)
    p_act = _conv_score(Smat64, fps90)
    reasons = {}
    bad_hit = None
    for r in cfg["quality_stage"]["rules"]:
        key = r["feature"]
        if r["src"] == "th":
            v = th[key]
        else:
            v = ph.get(key, float("nan"))
        t = float(r["threshold"])
        hit = False if not np.isfinite(v) else (
            bool(v < t) if r["op"] == "<" else bool(v > t))
        margin = float("nan") if not np.isfinite(v) else (
            (t - v) if r["op"] == "<" else (v - t))
        reasons[r["name"]] = {"hit": hit, "value": round(float(v), _Q)
                              if np.isfinite(v) else None, "op": r["op"],
                              "threshold": round(t, _Q),
                              "margin_units": round(margin, _Q)
                              if np.isfinite(margin) else None,
                              "feature": key}
        if hit and bad_hit is None:
            bad_hit = r["name"]
    p_use = 0.0 if bad_hit else p_act
    reasons["activity"] = {"p_conv": round(p_act, _Q), "op": ">=",
                           "threshold": round(
                               float(cfg["activity"]["threshold"]), _Q),
                           "model": "conv1d(seed0)"}
    d = cfg["dup"]
    reasons["duplicate"] = {"hit": False, "op": ">=",
                            "threshold": round(d["threshold"], _Q),
                            "note": "corpus-NN offline; same threshold"}
    keep = [i for i, n in enumerate(names)
            if n not in cfg.get("digest_exclude", [])]
    blob = (np.round(row[keep].astype(np.float64), _Q).tobytes()
            + np.array([thq[k] for k in T.THEORY_FEATURES],
                       dtype=np.int64).tobytes()
            + np.array([_qi(ph.get(k, float("nan")))
                        for k in P.PHYSICS_FEATURES],
                       dtype=np.int64).tobytes()
            + np.array([round(p_act, _Q)], dtype=np.float64).tobytes())
    return {"usable_prob": round(float(p_use), _Q),
            "decision": ("BAD:" + bad_hit) if bad_hit else (
                "GOOD" if p_act >= float(cfg["activity"]["threshold"])
                else "BAD:activity"),
            "reasons": reasons, "features_digest": hashlib.sha256(blob
                                                                  ).hexdigest()}
