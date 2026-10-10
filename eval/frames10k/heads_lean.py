"""Lean frame heads on 8k AI-labelled frames (CPU, thread-capped).

Fixed C=1 (no tuning whatsoever — operating point fixed a priori), dim 768
only (truncation already measured on the 2k subset), stratified 5-fold CV,
OOF AUROC + 1000-rep bootstrap CI, balanced accuracy. Run with
OMP/MKL/OPENBLAS_NUM_THREADS=2. Writes frame_cv.lean_8k.json +
frame_heads.lean_8k.json next to the input arrays.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parents[2] / "eval" / "results" / "frames10k" / "partial_8k"
    E = np.load(d / "embeddings.fp16.npy").astype(np.float32)  # float32: RAM-safe
    yh = (np.load(d / "hand.npy") >= 1).astype(int)
    yw = np.load(d / "work.npy").astype(int)
    print(f"n={len(E)} hands>=1 frac={yh.mean():.3f} work=yes frac={yw.mean():.3f}", flush=True)

    out = {}
    heads = {}
    for target, y in (("active_labor", yw), ("hands_ok", yh)):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        oof = np.zeros(len(y))
        for tr, te in skf.split(E, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(E[tr], y[tr])
            oof[te] = clf.predict_proba(E[te])[:, 1]
        auc = roc_auc_score(y, oof)
        rng = np.random.RandomState(0)
        vals = []
        for _ in range(1000):
            idx = rng.randint(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            vals.append(roc_auc_score(y[idx], oof[idx]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ba = balanced_accuracy_score(y, oof > 0.5)
        acc = accuracy_score(y, oof > 0.5)
        out[target] = {"cv_note": "stratified 5-fold, C=1 fixed (no tuning)",
                       "oof_auroc": float(auc), "oof_auroc_ci95": [float(lo), float(hi)],
                       "balacc": float(ba), "acc": float(acc),
                       "label_source": "AI (gemini-2.5-flash prompt; NOT human)",
                       "n": int(len(y)), "pos_frac": float(y.mean())}
        print(f"{target}: AUROC={auc:.4f} [{lo:.4f},{hi:.4f}] balacc={ba:.4f}", flush=True)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(E, y)
        sc, lr = clf[0], clf[1]
        heads[target] = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                         "coef": lr.coef_.reshape(-1).tolist(),
                         "intercept": float(lr.intercept_[0])}
    (d / "frame_cv.lean_8k.json").write_text(json.dumps(out, indent=1))
    (d / "frame_heads.lean_8k.json").write_text(json.dumps({"heads": heads, "dim": 768}))
    print("wrote lean_8k outputs", flush=True)


if __name__ == "__main__":
    main()
