"""LEAD B / LB1 — INDEPENDENT reproduction of E1.02 (EG2 window-embedding fps subsample).

Written from scratch (not a copy of eval/exp_fps/run_eg2.py) so that the E1.02
numbers are verified by a second implementation:
  * 2 fps  = all cached rows, 1 fps = rows[::2], 0.5 fps = rows[::4]
  * per-window mean over the subsampled rows, L2-normalised
  * leave-one-clip-out logistic regression C=0.01, class_weight=balanced
  * clip-bootstrap 95% CI (2000 resamples of CLIPS, seed 0)

Data: /workspace/jadal_pod_bundle/caches/frames10k/segments.npz (READ-ONLY cache)
Labels: human gold IndEgo windows carried in the cache (labels array).
Protocol: LOCO over the 150 IndEgo windows, 7 clips, clip-bootstrap 95% CI.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEG = "/workspace/jadal_pod_bundle/caches/frames10k/segments.npz"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "pod_verify", "e1_02_eg2.json")
SEED = 0
NBOOT = 2000


def loco_proba(X, y, clips, C=0.01):
    p = np.zeros(len(y), dtype=float)
    for c in sorted(set(clips)):
        tr, te = clips != c, clips == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def clip_bootstrap_ci(y, p, clips, n=NBOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    uc = np.array(sorted(set(clips)))
    aucs = []
    for _ in range(n):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([np.where(clips == c)[0] for c in pick])
        yy, pp = y[idx], p[idx]
        if len(set(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bal_acc(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    pos, neg = y == 1, y == 0
    tpr = (pred[pos] == 1).mean() if pos.any() else float("nan")
    tnr = (pred[neg] == 0).mean() if neg.any() else float("nan")
    return float((tpr + tnr) / 2)


def main():
    z = np.load(SEG, allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    clips = np.array([str(c) for c in z["clips"]])
    y = np.array(z["labels"]).astype(int)
    E = z["F"].astype(np.float32)
    cnt = np.array(z["counts"]).astype(int)
    n_frames_total = int(cnt.sum())

    rows = []
    for tag, step in [("2fps", 1), ("1fps", 2), ("0.5fps", 4)]:
        Xs = []
        used = 0
        for k in range(len(ids)):
            n = max(1, int(cnt[k]))
            r = E[k, :n:step]
            if len(r) == 0:
                r = E[k, :1]
            used += len(r)
            Xs.append(r.mean(axis=0))
        X = np.stack(Xs).astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        p = loco_proba(X, y, clips, C=0.01)
        lo, hi = clip_bootstrap_ci(y, p, clips)
        rows.append({
            "fps": tag, "step": step,
            "mean_frames_per_window": round(used / len(ids), 2),
            "auroc": round(float(roc_auc_score(y, p)), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "bal_acc_at_0.5": round(bal_acc(y, p), 4),
        })
    base = rows[0]["auroc"]
    for r in rows:
        r["delta_vs_2fps"] = round(r["auroc"] - base, 4)

    out = {
        "experiment": "E1.02",
        "verified_by": "lead-b (LB1, independent re-implementation)",
        "n": int(len(y)),
        "n_clips": int(len(set(clips))),
        "n_bad": int((y == 0).sum()),
        "label_source": "human gold IndEgo windows (labels carried in segments.npz cache)",
        "protocol": ("LOCO logistic C=0.01 class_weight=balanced on L2-normalised mean of "
                     "subsampled EG2 rows; clip-bootstrap 95% CI, 2000 resamples, seed 0"),
        "data": SEG,
        "cache_frames_total": n_frames_total,
        "rows": rows,
        "claimed_reference": {"2fps": 0.848, "1fps": 0.860, "0.5fps": 0.870},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for r in rows:
        print(f"{r['fps']:>6s} frames/window {r['mean_frames_per_window']:5.2f} "
              f"AUROC {r['auroc']:.4f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
              f"bal.acc {r['bal_acc_at_0.5']:.3f}")
    print("wrote", os.path.realpath(OUT))


if __name__ == "__main__":
    sys.exit(main())
