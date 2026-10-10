"""Recompute fair single-logistic baselines (class_weight=balanced) and patch M1 JSONs."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from jadal.gpuheads.data import load_frame_eg2, load_spectral

eg2 = load_frame_eg2()
spec = load_spectral()
jobs = [
    ("eval/results/gpu_heads/m1_eg2_hands.json", eg2["emb"].astype(np.float64),
     (eg2["hand"] >= 1).astype(int)),
    ("eval/results/gpu_heads/m1_spec_hands.json", spec["X"].astype(np.float64),
     (spec["hand"] >= 1).astype(int)),
    ("eval/results/gpu_heads/m1_spec_work.json", spec["X"].astype(np.float64),
     (spec["work"] == 1).astype(int)),
]
for fn, X, y in jobs:
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    P = np.zeros(len(y))
    for tri, tei in skf.split(X, y):
        mu = X[tri].mean(0)
        sd = X[tri].std(0)
        sd[sd < 1e-8] = 1
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        clf.fit((X[tri] - mu) / sd, y[tri])
        P[tei] = clf.predict_proba((X[tei] - mu) / sd)[:, 1]
    a = float(roc_auc_score(y, P))
    r = json.load(open(fn))
    r["single_logistic_balanced_auroc"] = a
    r["delta_many_minus_balanced_single"] = float(r["auroc"] - a)
    json.dump(r, open(fn, "w"), indent=1)
    print(f"{fn}: balanced-LR={a:.4f} many={r['auroc']:.4f} "
          f"delta={r['auroc'] - a:+.4f} n={len(y)}", flush=True)
