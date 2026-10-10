"""Streaming + independent LOCO verification of the cascade (CPU-only).

Reads eval/results/gpu/features.json (full + prefix + s0 per window) and the
shared manifests. Trains its own LOCO logistic head (independent of the
parallel cascade.json) and reports:
  - LOCO AUC + clip-bootstrap CI on gold (corroboration row)
  - streaming time-to-first-reject per degradation (2-s chunks)
  - per-chunk CPU extraction latency (real-time capable if < 2 s)

Writes eval/results/gpu/stream.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = os.path.join(HERE, "..", "results", "gpu", "features.json")
OUT = os.path.join(HERE, "..", "results", "gpu", "stream.json")
GOLD = "/workspace/jadal_pod_bundle/labels/manifest_gold.json"

FEAT_NAMES = [
    "temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac", "flicker_idx",
    "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "pca_top1_share",
    "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac",
    "motion_energy", "global_shift", "skin_frac", "illum_r", "illum_g", "illum_b",
    "rppg_pulse", "dup_hash_mean", "zncc_self",
]

S0_RULES = [("black", "s0p95", "<", 0.03), ("dark", "s0p50", "<", 0.04),
            ("lens_blocked", "s0var", "<", 1e-5), ("overexposed", "s0bright", ">", 0.5),
            ("frozen", "s0mot", "<", 1e-6)]


def vec(d):
    return np.array([d[k] for k in FEAT_NAMES], dtype=np.float64)


def s0_probe_eval(s0):
    if s0 is None or "error" in s0:
        return None
    m = {"s0p95": s0["p95"], "s0p50": (s0["p5"] + s0["mean"]) / 2,
         "s0var": s0["spatial_var"], "s0bright": 1.0 if s0["p95"] > 0.95 else 0.0,
         "s0mot": s0["temp_e"]}
    for name, key, op, thr in S0_RULES:
        v = m[key]
        if (op == "<" and v < thr) or (op == ">" and v > thr):
            return (name, key, op, thr, v)
    return None


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    feats = json.load(open(FEATS))["windows"]
    gold = {it["id"]: it for it in json.load(open(GOLD))["items"]}
    clips = sorted({it["clip"] for it in gold.values()})

    ids = [i for i in gold if i in feats and "full" in feats[i]]
    X = np.stack([vec(feats[i]["full"]) for i in ids])
    y = np.array([gold[i]["label"] for i in ids])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd

    # LOCO: prob of usable for each window from held-out-clip model
    p = np.zeros(len(ids))
    for c in clips:
        te = np.array([gold[i]["clip"] == c for i in ids])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xs[~te], y[~te])
        p[te] = clf.predict_proba(Xs[te])[:, 1]
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(200):
        cs = rng.choice(clips, size=len(clips), replace=True)
        sel = np.concatenate([np.where(np.array([gold[i]["clip"] for i in ids]) == c)[0]
                              for c in cs])
        if len(set(y[sel])) == 2:
            boots.append(roc_auc_score(y[sel], p[sel]))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    thr, margin = 0.6, 0.25
    # streaming: per-chunk decisions; S0 on first chunk, S1 on each prefix
    # need prefix LOCO scores: train prefix-k heads the same way
    nchunk = 5
    P = np.zeros((len(ids), nchunk))
    for k in range(nchunk):
        Xk = np.stack([vec(feats[i]["prefix"][k]) for i in ids])
        Xk = (Xk - Xk.mean(0)) / (Xk.std(0) + 1e-12)
        for c in clips:
            te = np.array([gold[i]["clip"] == c for i in ids])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xk[~te], y[~te])
            P[te, k] = clf.predict_proba(Xk[te])[:, 1]

    ttf = {}  # time-to-first-reject per gold-bad reason + synth degradation
    for j, i in enumerate(ids):
        ev = s0_probe_eval(feats[i].get("s0"))
        if ev:
            t_first = 1  # S0 fires on first 2-s chunk
            how = f"S0:{ev[0]}"
        else:
            t_first, how = None, None
            for k in range(nchunk):
                if P[j, k] < thr - margin:
                    t_first, how = k + 1, "S1"
                    break
        if y[j] == 0:
            key = f"gold:{gold[i].get('reason', '?')}"
            ttf.setdefault(key, []).append(t_first)
    # synth windows
    for wid, w in feats.items():
        if not wid.startswith("syn_") or "prefix" not in w:
            continue
        deg = w.get("degradation", "?")
        ev = s0_probe_eval(w.get("s0"))
        if ev:
            ttf.setdefault(f"synth:{deg}", []).append(1)
            continue
        Xs1 = vec(w["full"])
        # use full-window LOCO-ish score via the pooled model
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y)
        hit = None
        for k, pr in enumerate(w["prefix"]):
            pk = clf.predict_proba(((vec(pr) - mu) / sd).reshape(1, -1))[0, 1]
            if pk < thr - margin:
                hit = k + 1
                break
        ttf.setdefault(f"synth:{deg}", []).append(hit)

    def summ(v):
        vs = [x for x in v if x is not None]
        return {"n": len(v), "n_rejected": len(vs),
                "median_chunks": float(np.median(vs)) if vs else None,
                "median_s": float(np.median(vs)) * 2 if vs else None}

    # per-chunk CPU latency: re-extract prefixes for 3 windows, timed
    sys.path.insert(0, os.path.join(HERE, "..", ".."))
    from eval.gpu.decode import decode_window
    from jadal.gpu.features import cpu_window_features
    lat = []
    for i in ids[:3]:
        g = gold[i]
        rgb, fps = decode_window(g["path"], g["start_s"],
                                 round(g["end_s"] - g["start_s"], 2), 160, 90)
        luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        chunk = max(int(round(2.0 * fps)), 1)
        for end in range(chunk, rgb.shape[0] + 1, chunk):
            t0 = time.perf_counter()
            cpu_window_features(luma[:end], rgb[:end], fps=min(fps, 30.0))
            lat.append(time.perf_counter() - t0)
    res = {
        "device": "RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU",
        "regenerate": "python3 eval/gpu/stream.py",
        "loco_verify": {"auc": auc, "auc_ci95": ci, "n": len(ids),
                        "note": "independent head; corroborates eval/results/gpu/cascade.json"},
        "threshold": thr, "margin": margin,
        "time_to_first_reject": {k: summ(v) for k, v in sorted(ttf.items())},
        "chunk_latency_s": {"mean": float(np.mean(lat)), "max": float(np.max(lat)),
                            "n": len(lat),
                            "realtime_capable": bool(float(np.max(lat)) < 2.0)},
    }
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1)[:2000])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
