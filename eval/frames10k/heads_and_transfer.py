"""Frame-level heads on frozen EG2 embeddings + Matryoshka truncation test (CPU).

Inputs: eval/results/frames10k/<ds>/embeddings.fp16.npy + hand.npy + work.npy
Labels are AI-produced (gemini-2.5-flash prompt; NOT human) — stated in outputs.
Protocol: stratified 5-fold CV, C in {0.01,0.1,1,10} picked by inner 3-fold
on train folds only (no test tuning). Metrics: AUROC + 95% bootstrap CI,
balanced accuracy, accuracy. Outputs JSON under eval/results/frames10k/<ds>/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT_ROOT = Path(__file__).resolve().parents[2] / "eval" / "results" / "frames10k"


def boot_ci(y, s, fn, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    y = np.asarray(y)
    s = np.asarray(s)
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--dims", default="768,256,128")
    ap.add_argument("--data-dir", default="",
                    help="override embedding dir (default eval/results/frames10k/<ds>)")
    ap.add_argument("--tag", default="",
                    help="output filename tag, e.g. partial_6k -> frame_cv.partial_6k.json")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows; else first N rows")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = Path(args.data_dir) if args.data_dir else OUT_ROOT / args.ds
    E = np.load(d / "embeddings.fp16.npy").astype(np.float64)
    hand = np.load(d / "hand.npy")
    work = np.load(d / "work.npy")
    if args.max_rows:
        E, hand, work = E[:args.max_rows], hand[:args.max_rows], work[:args.max_rows]
    yh = (hand >= 1).astype(int)  # hands visible (any)
    yw = work.astype(int)         # active_labor yes
    print(f"{args.ds}: n={len(E)} hands>=1 frac={yh.mean():.3f} work=yes frac={yw.mean():.3f}", flush=True)

    out = {}
    for target, y in (("active_labor", yw), ("hands_ok", yh)):
        out[target] = {}
        for dim in [int(x) for x in args.dims.split(",")]:
            X = E[:, :dim]  # Matryoshka truncation: leading dims
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            aucs, baccs, accs, oof = [], [], [], np.zeros(len(y))
            for tr, te in skf.split(X, y):
                # inner 3-fold C selection on train only
                best_c, best = 1.0, -1.0
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
                for c in (0.01, 0.1, 1.0, 10.0):
                    i_aucs = []
                    for itr, ite in inner.split(X[tr], y[tr]):
                        clf = make_pipeline(StandardScaler(),
                                            LogisticRegression(C=c, max_iter=2000))
                        clf.fit(X[tr][itr], y[tr][itr])
                        try:
                            i_aucs.append(roc_auc_score(y[tr][ite],
                                                        clf.predict_proba(X[tr][ite])[:, 1]))
                        except ValueError:
                            pass
                    m = float(np.mean(i_aucs)) if i_aucs else 0.0
                    if m > best:
                        best, best_c = m, c
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(C=best_c, max_iter=2000))
                clf.fit(X[tr], y[tr])
                s = clf.predict_proba(X[te])[:, 1]
                oof[te] = s
                aucs.append(roc_auc_score(y[te], s))
                baccs.append(balanced_accuracy_score(y[te], s > 0.5))
                accs.append(accuracy_score(y[te], s > 0.5))
            m_auc, lo, hi = boot_ci(y, oof, roc_auc_score)
            out[target][str(dim)] = {
                "cv_auroc_mean": float(np.mean(aucs)), "cv_auroc_folds": [float(a) for a in aucs],
                "oof_auroc": m_auc, "oof_auroc_ci95": [lo, hi],
                "cv_balacc_mean": float(np.mean(baccs)), "cv_acc_mean": float(np.mean(accs)),
                "label_source": "AI (gemini-2.5-flash prompt; NOT human)",
                "protocol": "stratified 5-fold CV; C in {0.01,0.1,1,10} via inner 3-fold on train only",
                "n": int(len(y)), "pos_frac": float(y.mean()),
            }
            print(f"{target} dim={dim}: AUROC={np.mean(aucs):.4f} "
                  f"(oof {m_auc:.4f} [{lo:.4f},{hi:.4f}]) balacc={np.mean(baccs):.4f}", flush=True)
    # save full-dim heads (coef for reuse in segment transfer)
    heads = {}
    for target, y in (("active_labor", yw), ("hands_ok", yh)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(E, y)
        sc, lr = clf[0], clf[1]
        heads[target] = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                         "coef": lr.coef_.reshape(-1).tolist(),
                         "intercept": float(lr.intercept_[0])}
    stem = f".{args.tag}" if args.tag else ""
    (d / f"frame_heads{stem}.json").write_text(json.dumps({"heads": heads, "dim": 768}))
    (d / f"frame_cv{stem}.json").write_text(json.dumps(out, indent=1))
    print("wrote", d / f"frame_cv{stem}.json", d / f"frame_heads{stem}.json")


if __name__ == "__main__":
    main()
