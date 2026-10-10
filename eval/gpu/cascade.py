"""Early-reject cascade S0/S1/S2 + LOCO eval + streaming sim.

Regenerates eval/results/gpu/cascade.json. CPU-only (no GPU lock).
Run: python3 eval/gpu/cascade.py
Requires eval/results/gpu/features.json (via `python eval/gpu/run_features.py`).

S0: exact rules on the 2-s tiny probe stored in features.json, each firing with
    {rule, value, op, threshold}. Thresholds are fixed a priori constants
    (frozen 5e-4 justified by codec-noise floor: frozen re-encodes <= 3e-6,
    live gold >= 1.9e-3); false rejects are REPORTED, not hidden.
S1: StandardScaler + LogisticRegression on the 26 features, leave-one-clip-out;
    |p-0.5| < MARGIN -> uncertain -> S2.
S2: hand-off interface only (expensive tier stub): queued ids + fraction.
    Amortised-cost model states its assumed S2/S1 cost ratio explicitly.
Streaming: proxy — the full-window head applied to cumulative 2-s prefix
    features; time-to-first-reject per degradation (video seconds, not latency).
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", ".."))
from jadal.gpu.features import FEATURE_NAMES  # noqa: E402

SPEC = "/home/parshu/projects/jadal-qa-wt/spectral_data"
FEATS = os.path.join(THIS, "..", "results", "gpu", "features.json")
OUT = os.path.join(THIS, "..", "results", "gpu", "cascade.json")

S0_RULES = [
    ("black", "p95", "<", 0.03),
    ("dark", "mean", "<", 0.05),
    ("lens_blocked", "spatial_var", "<", 1e-5),
    ("overexposed", "p5", ">", 0.70),
    ("frozen", "temp_e", "<", 5e-4),
]
MARGIN = 0.25
H_S2_ASSUMED = 5.0  # NOT measured: assumed cost of S2 heavy tier in units of S1


def s0_decide(s0):
    for rule, key, op, thr in S0_RULES:
        if key not in s0:
            continue
        val = s0[key]
        if (val < thr) if op == "<" else (val > thr):
            return {"exit": "S0", "verdict": "reject",
                    "evidence": {"rule": rule, "value": val, "op": op, "threshold": thr}}
    return {"exit": None, "verdict": None, "evidence": None}


def main():
    t0 = time.perf_counter()
    gold = json.load(open(os.path.join(SPEC, "manifest_gold.json")))["items"]
    W = json.load(open(FEATS))["windows"]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    clips = sorted({it["clip"] for it in gold})
    ids = [it["id"] for it in gold if it["id"] in W and "full" in W[it["id"]]]
    X = np.array([[W[i]["full"][n] for n in FEATURE_NAMES] for i in ids])
    y = np.array([next(it["label"] for it in gold if it["id"] == i) for i in ids])
    cof = [next(it["clip"] for it in gold if it["id"] == i) for i in ids]
    p = np.zeros(len(ids))
    for c in clips:
        tr = [k for k in range(len(ids)) if cof[k] != c]
        te = [k for k in range(len(ids)) if cof[k] == c]
        if not tr or not te:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    ths = np.linspace(0.05, 0.95, 37)
    best, best_t = -1, 0.5
    for t in ths:
        pred = (p >= t).astype(int)
        tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0
        tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0
        if tpr + tnr - 1 > best:
            best, best_t = tpr + tnr - 1, t
    auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else 0.5
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(400):
        cs = rng.choice(clips, size=len(clips), replace=True)
        idx = [k for k in range(len(ids)) if cof[k] in cs]
        if len(set(y[idx])) > 1:
            boot.append(roc_auc_score(y[idx], p[idx]))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else [auc, auc]

    exits = Counter()
    false_rejects = []
    s1_tp = s1_tn = s1_fp = s1_fn = 0
    for it in gold:
        w = W.get(it["id"], {})
        if "full" not in w:
            exits["error"] += 1
            continue
        s0 = s0_decide(w.get("s0", {}))
        if s0["exit"] == "S0":
            exits["S0"] += 1
            if it["label"] == 1:
                false_rejects.append({"id": it["id"], "stage": "S0", "evidence": s0["evidence"]})
            else:
                s1_tn += 1
        else:
            pv = float(p[ids.index(it["id"])])
            if abs(pv - 0.5) < MARGIN:
                exits["S2"] += 1
            elif pv >= best_t:
                exits["S1_accept"] += 1
                if it["label"] == 1:
                    s1_tp += 1
                else:
                    s1_fp += 1
            else:
                exits["S1_reject"] += 1
                if it["label"] == 1:
                    s1_fn += 1
                    false_rejects.append({"id": it["id"], "stage": "S1", "p": pv})
                else:
                    s1_tn += 1
    n = sum(exits.values())
    # cost model: baseline runs S1 + heavy S2 on EVERYTHING
    base = n * (1.0 + H_S2_ASSUMED)
    casc = (exits["S0"] * 0.05 + (exits["S1_accept"] + exits["S1_reject"]) * 1.0
            + exits["S2"] * H_S2_ASSUMED)
    # synth exits per degradation
    synth = json.load(open(os.path.join(SPEC, "manifest_synth.json")))["items"]
    syn_res = {}
    for it in synth:
        d = it.get("degradation", "?")
        syn_res.setdefault(d, Counter())
        w = W.get(it["id"], {})
        if "full" not in w:
            syn_res[d]["error"] += 1
            continue
        syn_res[d]["S0" if s0_decide(w.get("s0", {}))["exit"] else "S1_or_S2"] += 1
    # streaming proxy TTD: first prefix chunk whose S0 fires or S1 p < thr (bad only)
    # need per-prefix scores: refit on all gold (descriptive, labelled proxy)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    def p_of(feat):
        v = np.array([[feat[n] for n in FEATURE_NAMES]])
        return float(clf.predict_proba(sc.transform(v))[0, 1])
    ttd = {}
    for it in gold + synth:
        if it.get("label", 1) == 1:
            continue
        w = W.get(it["id"], {})
        pre = w.get("prefix", [])
        key = it.get("degradation") or ("gold:" + it.get("reason", "?"))
        found = None
        for ci_, feat in enumerate(pre):
            if s0_decide_proxy(feat) or p_of(feat) < best_t:
                found = (ci_ + 1) * 2.0
                break
        ttd.setdefault(key, []).append(found)
    ttd_med = {k: (float(np.median([v for v in vs if v])) if any(vs) else None)
               for k, vs in ttd.items()}
    s2_ids = [it["id"] for it in gold if exits and W.get(it["id"], {}).get("full")
              and not s0_decide(W[it["id"]].get("s0", {}))["exit"]
              and abs(float(p[ids.index(it["id"])]) - 0.5) < MARGIN]
    res = {
        "device": "RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU",
        "regenerate": "python eval/gpu/run_features.py && python eval/gpu/cascade.py",
        "features_source": "eval/results/gpu/features.json",
        "loco": {"auc": auc, "auc_ci95": ci, "threshold": float(best_t),
                 "n": len(ids), "n_good": int(y.sum()), "n_bad": int((1 - y).sum())},
        "margin": MARGIN,
        "s0_rules": S0_RULES,
        "gold_exits": dict(exits),
        "gold_exit_frac": {k: round(v / max(n, 1), 3) for k, v in exits.items()},
        "decided": {"tp": s1_tp, "tn": s1_tn, "fp": s1_fp, "fn": s1_fn,
                    "balanced_acc": round(0.5 * (s1_tp / max(s1_tp + s1_fn, 1)
                                                 + s1_tn / max(s1_tn + s1_fp, 1)), 3),
                    "coverage": round(1 - exits["S2"] / max(n, 1), 3)},
        "false_rejects": false_rejects,
        "n_false_reject": len(false_rejects),
        "synth_exits": {k: dict(v) for k, v in syn_res.items()},
        "cost_model": {"H_S2_assumed_NOT_measured": H_S2_ASSUMED,
                       "compute_saved_frac": round(1 - casc / max(base, 1e-9), 3)},
        "streaming_ttd_s": ttd_med,
        "s2": {"method": "STUB: mediapipe-hands CPU or EG2 GPU (not run here)",
               "queued": s2_ids, "n_queued": len(s2_ids)},
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("s2",)}, indent=1)[:2800])
    print("wrote", OUT)


def s0_decide_proxy(feat):
    """Streaming proxy: reject prefix if frozen-like or near-black via full feats."""
    return feat.get("frozen_frac", 0) > 0.9 or feat.get("luma_p95", 1) < 0.03


if __name__ == "__main__":
    main()
