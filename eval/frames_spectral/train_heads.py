"""Frame-level heads: hand_count (0/1/2 + hands-visible binary) and active_labor.

Labels are AI labels (gemini-2.5-flash per dataset README), NOT human.
Trains on egocentric_10k.npz, evaluates out-of-domain on ego4d/epic_kitchens.

Models (fixed seeds): regularised logistic (L2, C grid in inner fold) and
HistGradientBoostingClassifier. 5-fold CV, two grouping protocols:
  (a) deterministic random split (seed 0) — frames carry no worker id;
  (b) split by frame_id hash (md5 bucket % 5).
Near-duplicate frames from the same video can leak across folds: stated here,
and quantified by the out-of-domain (ego4d/epic_kitchens) evaluation.
Class imbalance: active_labor 'no' is only ~8% -> class_weight='balanced'
for logistic, balancing sample_weight for HGB, plus a trivial majority
baseline. Metrics: AUROC / balanced accuracy / F1 with bootstrap CIs (B=1000).
Family ablation: drop-one-family logistic AUROC deltas.

Usage: python -m eval.frames_spectral.train_heads
Writes eval/results/frames_spectral/frame_heads.json
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from eval.frames_spectral.frame_features import FAMILIES, FEATURES

IN = "eval/results/frames_spectral"
OUT = "eval/results/frames_spectral/frame_heads.json"
SEED = 0
import os
B = int(os.environ.get("FS_BOOT", "1000"))


def boot_ci_safe(y, p, fn, rng, b=B):
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, len(y), len(y))
        try:
            v = fn(y[idx], p[idx])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if not len(vals):
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def oof_predict(X, y, model_fn, groups, seed=0):
    """Out-of-fold probabilities for a binary target with group folds."""
    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # groups only used for the hash protocol via precomputed fold ids
    uniq = np.unique(groups)
    fold_of = {g: i % 5 for i, g in enumerate(uniq)}
    folds = np.asarray([fold_of[g] for g in groups])
    for f in range(5):
        tr = folds != f
        te = folds == f
        sc = StandardScaler()
        Xa = sc.fit_transform(X[tr])
        m = model_fn()
        m.fit(Xa, y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def metrics_binary(y, p, rng):
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    pred = (p >= 0.5).astype(int)
    ba = float(balanced_accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, zero_division=0))
    return {
        "auroc": auc,
        "auroc_ci": list(boot_ci_safe(y, p, roc_auc_score, rng)),
        "bal_acc": ba,
        "f1": f1,
        "f1_ci": list(boot_ci_safe(
            y, pred, lambda a, b: f1_score(a, b, zero_division=0), rng)),
    }


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    z = np.load(f"{IN}/egocentric_10k.npz", allow_pickle=True)
    X = z["X"].astype(np.float64)
    hand = z["hand"].astype(int)
    work = z["work"].astype(int)
    ids = [str(s) for s in z["ids"].tolist()]
    feats = [str(s) for s in z["features"].tolist()]
    assert feats == FEATURES, f"feature order drift: {len(feats)} vs {len(FEATURES)}"
    n, d = X.shape
    out: dict = {"n": n, "d": d, "label_source": "AI (gemini-2.5-flash per dataset README), NOT human",
                 "label_counts": {"hand_count": {str(k): int((hand == k).sum()) for k in (0, 1, 2)},
                                  "active_labor": {"yes": int(work.sum()), "no": int((work == 0).sum())}}}
    # group protocols
    grp_rand = np.random.default_rng(SEED).integers(0, 5, n)
    grp_hash = np.asarray([int(hashlib.md5(i.encode()).hexdigest(), 16) % 5 for i in ids])

    log_fn = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
    hgb_fn = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                    max_leaf_nodes=31, random_state=SEED)

    tasks = {"hands_visible": (hand > 0).astype(int), "active_labor": work}
    oof_store = {}
    for tname, y in tasks.items():
        out[tname] = {}
        # majority baseline
        maj = int(y.mean() >= 0.5)
        majpred = np.full(n, maj)
        out[tname]["majority_baseline"] = {
            "bal_acc": float(balanced_accuracy_score(y, majpred)),
            "f1": float(f1_score(y, majpred, zero_division=0))}
        for pname, grp in (("random_split", grp_rand), ("frameid_hash", grp_hash)):
            out[tname][pname] = {}
            for mname, mfn in (("logreg", log_fn), ("hgb", hgb_fn)):
                if mname == "hgb":
                    # HGB handles raw scale; still standardise for comparability
                    p = oof_predict(X, y, mfn, grp)
                else:
                    p = oof_predict(X, y, mfn, grp)
                out[tname][pname][mname] = metrics_binary(y, p, rng)
                out[tname][pname][mname]["oof_prob_mean"] = float(p.mean())
                if mname == "logreg" and pname == "random_split":
                    oof_store[tname] = p.astype(np.float64)
        # 3-class hand_count (logreg multinomial), macro F1 + per-class AUROC ovr
        if tname == "hands_visible":
            yc = hand
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            P = np.zeros((n, 3))
            for tr, te in skf.split(X, yc):
                sc = StandardScaler()
                m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                       random_state=SEED)
                m.fit(sc.fit_transform(X[tr]), yc[tr])
                P[te] = m.predict_proba(sc.transform(X[te]))
            pred = P.argmax(1)
            aucs = {}
            for k in (0, 1, 2):
                try:
                    aucs[str(k)] = float(roc_auc_score((yc == k).astype(int), P[:, k]))
                except Exception:
                    aucs[str(k)] = float("nan")
            out["hand_count_3class"] = {
                "model": "logreg-multinomial-C1-balanced-random_split",
                "macro_f1": float(f1_score(yc, pred, average="macro", zero_division=0)),
                "bal_acc": float(balanced_accuracy_score(yc, pred)),
                "ovr_auroc": aucs}

    # family ablation (logreg, random_split protocol, both binary tasks)
    idx = {f: i for i, f in enumerate(FEATURES)}
    out["ablation"] = {}
    for tname, y in tasks.items():
        base = out[tname]["random_split"]["logreg"]["auroc"]
        out["ablation"][tname] = {"full_logreg_auroc": base, "drop_one_family": {}}
        for fam, members in FAMILIES.items():
            keep = [i for i in range(d) if FEATURES[i] not in members]
            p = oof_predict(X[:, keep], y, log_fn, grp_rand)
            try:
                a = float(roc_auc_score(y, p))
            except Exception:
                a = float("nan")
            out["ablation"][tname]["drop_one_family"][fam] = {
                "auroc": a, "delta": float(a - base)}

    # out-of-domain: train on ALL ego10k, test on ego4d + epic_kitchens
    out["out_of_domain"] = {}
    for other in ("ego4d", "epic_kitchens"):
        try:
            zo = np.load(f"{IN}/{other}.npz", allow_pickle=True)
        except FileNotFoundError:
            out["out_of_domain"][other] = {"status": "missing-npz"}
            continue
        Xo = zo["X"].astype(np.float64)
        ho = zo["hand"].astype(int)
        wo = zo["work"].astype(int)
        res = {}
        for tname, ytr, yte in (("hands_visible", (hand > 0).astype(int), (ho > 0).astype(int)),
                                ("active_labor", work, wo)):
            sc = StandardScaler()
            m = log_fn()
            m.fit(sc.fit_transform(X), ytr)
            p = m.predict_proba(sc.transform(Xo))[:, 1]
            res[tname] = {"n_test": int(len(yte)), **metrics_binary(yte, p, rng)}
        out["out_of_domain"][other] = res

    out["elapsed_s"] = time.time() - t0
    np.savez_compressed(f"{IN}/frame_oof_logreg.npz",
                        hands_visible=oof_store["hands_visible"],
                        active_labor=oof_store["active_labor"],
                        hand=hand.astype(np.int8), work=work.astype(np.int8),
                        ids=np.asarray(ids))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("label_counts",)}, indent=1))
    for tname in ("hands_visible", "active_labor"):
        for pname in ("random_split", "frameid_hash"):
            for mname in ("logreg", "hgb"):
                m = out[tname][pname][mname]
                print(f"{tname}/{pname}/{mname}: AUROC={m['auroc']:.3f} "
                      f"CI={m['auroc_ci']} balAcc={m['bal_acc']:.3f} F1={m['f1']:.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
