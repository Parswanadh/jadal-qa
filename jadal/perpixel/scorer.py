"""Deterministic scorer v2: quality tripwires + conv activity score.

score(video_path, start_s, end_s) ->
  {'usable_prob', 'decision', 'reasons': {evidence value/op/threshold...},
   'features_digest'}.

Stage 1 (deterministic rules, rounded compare at Q=4): OR of frozen quality
tripwires {lowent, blur, ten} -> BAD:<rule> with evidence.
Stage 2: conv1d activity score on the per-frame series (frozen JSON weights,
seed 0, CPU); usable_prob = round4(p). Quality hit forces 0.0.
Duplicate: single-window loop-tripwire (dhash_loop_min <= 4 AND still_run >
0.5 AND mot_frac < train-kept median — all frozen ints) fires BAD:loop with
evidence; cross-window NN stays offline (same constructed threshold).
Digest = sha256 over round-4 M1 row (EXCLUDING B_mv_per_frame: ffmpeg MV
debug counts vary run to run) + theory fixed-point ints + round-4 physics
(NaN -> sentinel) + round-4 conv prob. Repeat runs -> identical.

Fix round 1: weights are JSON (no pickle; C1), sha256-verified BEFORE use;
model cached (no per-call torch.load; M4); no global torch seed/thread
mutation at call time; series length via truncate/zero-pad (I4).
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
                                     "perpixel_conv.json"))

_Q = 4
_QI = 10 ** _Q
_MISSING = -99999999

_NET = None  # cached ((weights_path, sha256), net, mu, sd, T)


def _load_config(path=CONFIG):
    with open(path) as f:
        return json.load(f)


def _load_net(cfg):
    """Load + hash-verify frozen conv weights once (I1: from cfg, not default).

    Cache key is (weights path, sha256) — a changed file at the same path
    is reloaded, never served stale (M4).
    """
    global _NET
    rel = cfg["activity"]["weights"]
    wpath = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", rel))
    want = cfg["activity"]["weights_sha256"]
    blob = open(wpath, "rb").read()
    got = hashlib.sha256(blob).hexdigest()
    if got != want:
        _NET = None
        raise ValueError(f"conv weights hash mismatch: {got[:16]} != "
                         f"{want[:16]} (refusing to score)")
    if _NET is not None and _NET[0] == (wpath, got):
        return _NET[1:]
    ck = json.loads(blob)
    net = torch.nn.Sequential(
        torch.nn.Conv1d(8, 16, 5, padding=2), torch.nn.ReLU(),
        torch.nn.Conv1d(16, 16, 5, padding=2), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
        torch.nn.Linear(16, 1))
    net.load_state_dict({k: torch.tensor(np.asarray(v, dtype=np.float32))
                         for k, v in ck["state"].items()})
    net.eval()
    mu = np.asarray(ck["mu"], dtype=np.float64)
    sd = np.asarray(ck["sd"], dtype=np.float64)
    _NET = ((wpath, got), net, mu, sd, int(ck["T"]))
    return net, mu, sd, int(ck["T"])


def _conv_score(Smat: np.ndarray, cfg) -> float:
    net, mu, sd, Tall = _load_net(cfg)
    Xa = S.fit_length(Smat.astype(np.float64), Tall)
    Xt = torch.tensor((Xa - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        p = torch.sigmoid(net(Xt[None].transpose(1, 2))).numpy().ravel()[0]
    return float(p)


def _qi(x: float) -> int:
    if not np.isfinite(x):
        return _MISSING
    return int(round(float(x) * _QI))


def score(video_path, start_s, end_s, config_path=CONFIG):
    cfg = _load_config(config_path)
    row, names, _, meta, frames90, fps90 = F.extract_row(
        video_path, start_s, end_s, return_frames=True)
    Smat64 = S.per_frame_series(frames90)
    th = T.extract_theory(frames90, fps90)
    ph = P.extract_physics(frames90, fps90, video_path, start_s, end_s)
    p_act = _conv_score(Smat64, cfg)
    reasons = {}
    bad_hit = None
    for r in cfg["quality_stage"]["rules"]:
        key = r["feature"]
        v = th[key] if r["src"] == "th" else ph.get(key, float("nan"))
        t = float(r["threshold"])
        hit = bool(np.isfinite(v)) and (
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
    # I2: single-window loop-tripwire (deterministic, frozen ints).
    # dhash_loop_min <= 4 (near-identical frame recurs after >= 2 s) AND a
    # still-dominated window (still_run > 0.5). Cross-window NN stays offline.
    lz = cfg.get("loop_tripwire", {"dhash_max": 4, "still_min": 0.5})
    loop_hit = (ph.get("dhash_loop_min", 64) <= lz["dhash_max"]
                and th.get("still_run", 0.0) > lz["still_min"])
    reasons["loop_tripwire"] = {
        "hit": bool(loop_hit), "dhash_loop_min": ph.get("dhash_loop_min"),
        "still_run": round(float(th.get("still_run", 0.0)), _Q),
        "rule": (f"dhash_loop_min <= {lz['dhash_max']} AND still_run > "
                 f"{lz['still_min']}")}
    if loop_hit and bad_hit is None:
        bad_hit = "loop"
    p_use = 0.0 if bad_hit else p_act
    reasons["activity"] = {"p_conv": round(p_act, _Q), "op": ">=",
                           "threshold": round(
                               float(cfg["activity"]["threshold"]), _Q),
                           "model": "conv1d(seed0,json-weights)"}
    d = cfg["dup"]
    # I3: the in-scorer duplicate reason IS the loop-tripwire (promoted from
    # evidence-only to a live duplicate hit); cross-window corpus-NN stays
    # offline. Frozen scope = quality + activity + single-window duplicate.
    reasons["duplicate"] = {
        "hit": bool(loop_hit), "via": "loop_tripwire",
        "op": ">=", "threshold": round(d["threshold"], _Q),
        "note": ("in-scorer duplicate evidence is single-window only "
                 "(loop-tripwire); cross-window corpus-NN stays offline "
                 "(M2, same threshold)")}
    keep = [i for i, n in enumerate(names)
            if n not in cfg.get("digest_exclude", [])]
    blob = (np.round(row[keep].astype(np.float64), _Q).tobytes()
            + np.array([T.quantise(th)[k] for k in T.THEORY_FEATURES],
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
