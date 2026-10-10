"""Merge H3/E1.06 per-cell feature shards -> LOCO eval per (fps,res) cell.

Regen: python3 eval/exp_fps_res/run_fps_res.py && python3 eval/exp_fps_res/merge_fps_res.py
Writes eval/results/exp_fps_res/E1.06.json (Pareto table + per-cell rows).
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
GOLD = os.path.join(BUNDLE, "labels", "manifest_gold.json")
D = os.path.join(ROOT, "eval", "results", "exp_fps_res")

BASELINE_AUROC = 0.8086  # E1.01 10fps/480p native (e1_01.json); task quotes 0.809
BUDGET = 0.02

CELL_ORDER = [(5.0, 480), (5.0, 360), (5.0, 240),
              (10.0 / 3.0, 480), (10.0 / 3.0, 360), (10.0 / 3.0, 240),
              (2.0, 480), (2.0, 360), (2.0, 240),
              (1.0, 480), (1.0, 360), (1.0, 240)]


def cell_name(fps, res):
    if abs(fps - 10.0 / 3.0) < 1e-9:
        tag = "3p33"
    elif float(fps).is_integer():
        tag = f"{int(fps)}"
    else:
        tag = str(fps)
    return f"f{tag}_r{res}"


def git_sha():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def loco_proba(X, y, clips, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for c in sorted(set(clips)):
        tr, te = clips != c, clips == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def clip_bootstrap_ci(y, p, clips, B=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    names = sorted(set(clips))
    out = []
    for _ in range(B):
        cs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(clips == c)[0] for c in cs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    from sklearn.metrics import roc_auc_score
    M = {i["id"]: i for i in json.load(open(GOLD))["items"]}
    rows = []
    for fps, res in CELL_ORDER:
        cell = cell_name(fps, res)
        feats = json.load(open(os.path.join(D, f"_feats_{cell}.json")))
        ids = sorted(k for k, r in feats.items() if "full" in r)
        names = sorted(feats[ids[0]]["full"].keys())
        X = np.nan_to_num(np.array([[feats[k]["full"][n] for n in names] for k in ids], float))
        y = np.array([M[k]["label"] for k in ids])
        clips = np.array([M[k]["clip"] for k in ids])
        p = loco_proba(X, y, clips)
        auc = float(roc_auc_score(y, p))
        lo, hi = clip_bootstrap_ci(y, p, clips)
        nfr = float(np.mean([feats[k]["n_frames"] for k in ids]))
        rows.append({
            "cell": cell, "fps": round(float(fps), 2), "resolution_px": res,
            "crf": 23, "n": len(ids), "n_features": len(names),
            "frames_per_window": round(nfr, 1),
            "auroc": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "delta_vs_baseline": round(auc - BASELINE_AUROC, 4),
            "within_budget": bool(auc - BASELINE_AUROC >= -BUDGET),
            "bytes_per_window": int(np.mean([feats[k]["bytes"] for k in ids])),
            "enc_ms_per_window": round(float(np.mean([feats[k]["enc_ms"] for k in ids])), 1),
            "dec_ms_per_window": round(float(np.mean([feats[k]["dec_ms"] for k in ids])), 1),
            "feat_ms_per_window": round(float(np.mean([feats[k]["feat_ms"] for k in ids])), 1),
        })
        for r in rows[-1:]:
            r["total_ms_per_window"] = round(
                r["enc_ms_per_window"] + r["dec_ms_per_window"] + r["feat_ms_per_window"], 1)
    ok = [r for r in rows if r["within_budget"]]
    cheapest = min(ok, key=lambda r: (r["bytes_per_window"], r["total_ms_per_window"])) if ok else None
    res_out = {
        "experiment": "E1.06", "n": 150, "n_good": 100, "n_bad": 50,
        "label_source": "human (IndEgo gold manifest_gold.json, 100 good/50 bad, one labeller)",
        "protocol": "re-encode each gold window at (fps,res) libx264 veryfast CRF23 (fps filter + square scale) -> decode 160x90 -> 26 cpu per-pixel features (frozen path) -> LOCO logistic C=0.3 class_weight=balanced + StandardScaler; CI=clip-bootstrap B=1000 seed0",
        "split": "leave-one-clip-out over the 7 gold clips (identical to E1.01/X1-RES)",
        "baseline": {"desc": "pixel-26 @10fps/480p native (E1.01 e1_01.json)", "auroc": BASELINE_AUROC},
        "budget": f"within {BUDGET} AUROC of baseline",
        "cheapest_within_budget": cheapest["cell"] if cheapest else None,
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_fps_res/run_fps_res.py && python3 eval/exp_fps_res/merge_fps_res.py",
        "device": "A40 pod host; CPU-only (ffmpeg + numpy float64 + sklearn)",
        "git_sha": git_sha(),
        "rows": rows,
        "fps_comparability": "temp_b0..b3/flicker_idx/motion_energy/rppg_pulse/frozen_frac NOT comparable across fps (E1.01 e1_01.json fps_comparability); cross-fps AUROC deltas mix content loss with feature-definition shift",
    }
    json.dump(res_out, open(os.path.join(D, "E1.06.json"), "w"), indent=1)
    print(f"{'cell':>10} {'fps':>5} {'res':>4} {'AUROC':>7} {'CI':>17} {'dBase':>7} {'bytes':>8} {'ms':>6} budget")
    for r in rows:
        print(f"{r['cell']:>10} {r['fps']:>5} {r['resolution_px']:>4} {r['auroc']:>7.4f} "
              f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] {r['delta_vs_baseline']:>+7.4f} "
              f"{r['bytes_per_window']:>8d} {r['total_ms_per_window']:>6.0f} "
              f"{'OK' if r['within_budget'] else '--'}")
    if cheapest:
        print(f"cheapest within {BUDGET}: {cheapest['cell']} "
              f"AUROC={cheapest['auroc']:.4f} d={cheapest['delta_vs_baseline']:+.4f} "
              f"bytes={cheapest['bytes_per_window']} ms={cheapest['total_ms_per_window']:.0f}")


if __name__ == "__main__":
    main()
