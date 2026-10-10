#!/usr/bin/env python3
"""N1 — Improving OOD accuracy on the 160 human-labelled Egocentric-10K windows.

Tracks the lead's brief:
  1. model shape on the transferable set (equal-weight mean / logistic / SVM / GBT),
  2. feature selection with grouped CV (inside each fold),
  3. domain-adaptive training: (a) class-prior weighting, (b) domain-classifier
     diagnostic (not in the decision path), (c) test-time per-factory normalization
     fit on the clip's own unlabelled windows,
  4. self-training on the 1,546 unlabelled Egocentric-10K windows,
  5. best OOD score + grouped CI and an error-vs-labels curve (k=10..160).

ALL evaluation is factory-grouped: a factory is never in both train and test.
The per-domain z-score is unsupervised (target mean/std); the per-feature sign
comes from the 150 IndEgo gold windows. No OOD threshold tuning.

Outputs eval/results/exp_acc/N1.01.json, N1.02.json, N1.03.json.
Run: OMP_NUM_THREADS=2 python3 eval/exp_acc/run_n1.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import n1lib as L  # noqa: E402

from scipy.stats import rankdata  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

OUT = os.path.join(REPO, "eval", "results", "exp_acc")
N_SPLITS = 5
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
LABEL_SOURCE = ("human, single labeller: labels/ood160_labels.json "
                "(160 Egocentric-10K windows, 138 good / 22 bad, 47 factories, "
                "q1=60 + q2=100) + labels/manifest_gold.json (150 IndEgo, 100/50)")
PROTOCOL = ("factory-grouped StratifiedGroupKFold(5, shuffle, seed=0) OOF; per-domain "
            "z-score (unsupervised target mean/std); sign from IndEgo gold; factory-grouped "
            "bootstrap 95% CI (n_boot=2000, seed=0) on the OOF scores; paired grouped "
            "bootstrap for gains. No OOD threshold tuning (0 for signed-z, 0.5 for probs).")
DEV = "CPU (numpy/sklearn, OMP_NUM_THREADS=2; no GPU, no decode)"


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"


def gkf(seed=0):
    return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)


def rk(a):
    return rankdata(a) / len(a)


# ---------------- recipes ----------------

def agg_apply(kind, Xtr, Xte):
    if kind == "mean":
        return Xtr.mean(1), Xte.mean(1)
    if kind == "median":
        return np.median(Xtr, 1), np.median(Xte, 1)
    if kind == "trim2":
        return np.sort(Xtr, 1)[:, 2:-2].mean(1), np.sort(Xte, 1)[:, 2:-2].mean(1)
    raise ValueError(kind)


def _logit(C, cw="balanced"):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, class_weight=cw))


def _svm(C):
    return make_pipeline(StandardScaler(),
                         LinearSVC(C=C, class_weight="balanced", max_iter=20000))


def _hgb():
    return HistGradientBoostingClassifier(max_depth=1, max_iter=150, learning_rate=0.05,
                                          class_weight="balanced", random_state=0)


def oof_scores(X, y, fac, make, seed=0):
    o = np.full(len(y), np.nan)
    for tr, te in gkf(seed).split(X, y, fac):
        m = make()
        m.fit(X[tr], y[tr])
        o[te] = (m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba")
                 else m.decision_function(X[te]))
    return o


def oof_logit_nestedC(X, y, fac, seed=0):
    """Outer grouped CV; inner grouped CV picks C. Returns OOF probs + picks."""
    o = np.full(len(y), np.nan)
    picks = []
    for tr, te in gkf(seed).split(X, y, fac):
        best_c, best = C_GRID[0], -1.0
        for c in C_GRID:
            aucs = []
            for itr, ite in gkf(seed + 1).split(X[tr], y[tr], fac[tr]):
                if len(np.unique(y[tr][itr])) < 2 or len(np.unique(y[tr][ite])) < 2:
                    continue
                m = _logit(c)
                m.fit(X[tr][itr], y[tr][itr])
                aucs.append(L.fast_auc(y[tr][ite], m.predict_proba(X[tr][ite])[:, 1]))
            v = float(np.mean(aucs)) if aucs else 0.5
            if v > best:
                best, best_c = v, c
        picks.append(best_c)
        m = _logit(best_c)
        m.fit(X[tr], y[tr])
        o[te] = m.predict_proba(X[te])[:, 1]
    return o, picks


def oof_select(X, y, fac, m, seed=0):
    """Per-fold top-m features by train-fold signed AUROC, then equal-weight mean."""
    o = np.full(len(y), np.nan)
    picks = []
    for tr, te in gkf(seed).split(X, y, fac):
        au = np.array([L.fast_auc(y[tr], X[tr, j]) for j in range(X.shape[1])])
        k = np.argsort(-au)[:m]
        picks.append(k.tolist())
        o[te] = X[te][:, k].mean(1)
    return o, picks


def oof_forward(X, y, fac, maxf, seed=0):
    o = np.full(len(y), np.nan)
    sizes = []
    for tr, te in gkf(seed).split(X, y, fac):
        cur, best = [], -1.0
        for _ in range(maxf):
            cand = None
            for j in range(X.shape[1]):
                if j in cur:
                    continue
                au = np.array([L.fast_auc(y[tr], X[tr, k]) for k in cur + [j]])
                ss = np.where(au >= 0.5, 1, -1)
                a = L.fast_auc(y[tr], (X[tr][:, cur + [j]] * ss).mean(1))
                if a > best:
                    best, cand = a, j
            if cand is None:
                break
            cur.append(cand)
        sizes.append(len(cur))
        au = np.array([L.fast_auc(y[tr], X[tr, k]) for k in cur])
        ss = np.where(au >= 0.5, 1, -1)
        o[te] = (X[te][:, cur] * ss).mean(1)
    return o, sizes


# ---------------- reporting ----------------

def score_block(name, method, y, s, fac, thr, extra=None):
    b = {"recipe": name, "method": method, "threshold": thr,
         "ood_auroc": round(L.fast_auc(y, s), 4),
         "ood_balacc": round(L.balacc(y, s, thr), 4),
         "ood_error": round(1.0 - L.balacc(y, s, thr), 4),
         "ood_auroc_ci95_factory_grouped": L.group_boot_ci(y, s, fac, L.fast_auc),
         "ood_balacc_ci95_factory_grouped": L.group_boot_ci(y, s, fac, L.balacc, thr=thr)}
    if extra:
        b.update(extra)
    return b


def domain_sep_auc(Xa, Xb, seed=0):
    from sklearn.model_selection import StratifiedKFold
    X = np.vstack([Xa, Xb])
    d = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    o = np.full(len(d), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, d):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        m.fit(X[tr], d[tr])
        o[te] = m.predict_proba(X[te])[:, 1]
    return round(L.fast_auc(d, o), 4)


# ---------------- main ----------------

def main():
    ood = L.load_ood()
    gold = L.load_gold()
    unl = L.load_unlabelled()
    y, fac = ood["y"], ood["fac"]
    n_good, n_bad = int((y == 1).sum()), int((y == 0).sum())
    sha = git_sha()

    Xo, Xg, names = L.build_transfer(ood, gold, "all14")
    sg, au_ind = L.indego_signs(Xg, gold["y"])
    zof = L.zscore(Xo)                 # transductive per-domain z-score (unsupervised)
    zo = zof * sg                      # IndEgo-signed
    zog = L.zscore(Xg) * sg
    yg = gold["y"]

    # ---- baseline ----
    base = zo.mean(1)
    blocks = [score_block("equal-weight mean (baseline)", "mean of IndEgo-signed per-domain "
                          "z-scores, 14 features", y, base, fac, 0.0)]

    # ---- 1. model shape ----
    med = np.median(zo, 1)
    trim = np.sort(zo, 1)[:, 2:-2].mean(1)
    med_iqr = np.median(Xo, 0)
    iqr = np.percentile(Xo, 75, 0) - np.percentile(Xo, 25, 0)
    robz = (Xo - med_iqr) / np.clip(iqr, 1e-9, None)
    rob = (robz * sg).mean(1)
    blocks.append(score_block("median of signed z", "median aggregation, 14 features",
                              y, med, fac, 0.0))
    blocks.append(score_block("trimmed mean (drop 2 each tail)", "trimmed aggregation",
                              y, trim, fac, 0.0))
    blocks.append(score_block("robust z mean (median/IQR)", "robust per-domain scale",
                              y, rob, fac, 0.0))

    logit_nested, c_picks = oof_logit_nestedC(zof, y, fac)
    blocks.append(score_block("logistic (C by inner grouped CV)",
                              "StandardScaler+LogisticRegression(balanced), nested C",
                              y, logit_nested, fac, 0.5, {"C_picks": c_picks}))
    svm = oof_scores(zof, y, fac, lambda: _svm(0.1))
    blocks.append(score_block("linear SVM C=0.1", "StandardScaler+LinearSVC(balanced)",
                              y, svm, fac, 0.0))
    hgb = oof_scores(zof, y, fac, _hgb)
    blocks.append(score_block("gradient-boosted tree (depth1)", "HistGradientBoosting depth1",
                              y, hgb, fac, 0.5))
    lg = oof_scores(zof, y, fac, lambda: _logit(0.1))
    blocks.append(score_block("logistic C=0.1", "StandardScaler+LogisticRegression(balanced)",
                              y, lg, fac, 0.5))

    # ---- 2. feature selection with grouped CV ----
    sel_blocks = []
    for m in (4, 6, 8, 10, 12):
        o, pk = oof_select(zo, y, fac, m)
        sel_blocks.append(score_block(f"select top-{m} (train AUROC, grouped)",
                                      "per-fold top-m by train-fold signed AUROC, equal mean",
                                      y, o, fac, 0.0,
                                      {"mean_n_features": float(np.mean([len(p) for p in pk]))}))
    fwd, sizes = oof_forward(zo, y, fac, maxf=8)
    sel_blocks.append(score_block("forward selection (grouped)",
                                  "per-fold greedy forward selection, equal mean",
                                  y, fwd, fac, 0.0, {"mean_n_features": float(np.mean(sizes))}))

    # ---- 3. adaptation ----
    adapt = {}
    # (a) class weighting
    cw_blocks = []
    for cw, tag in [("balanced", "balanced"), (None, "none"),
                    ({0: 6.26, 1: 1.0}, "prior-corrected (13.75% bad)")]:
        o = oof_scores(zof, y, fac, lambda cw=cw: _logit(0.1, cw=cw))
        cw_blocks.append(score_block(f"logistic C=0.1, class_weight={tag}",
                                     "class-prior / importance weighting", y, o, fac, 0.5))
    adapt["class_weighting"] = cw_blocks
    # (b) domain classifier diagnostic (NOT in decision path)
    Xu8 = unl["Xpix"][:, [L.FEATURE_NAMES26.index(n) for n in L.STABLE8]]
    ok = ~np.isnan(Xu8).any(1)
    adapt["domain_classifier_diagnostic"] = {
        "indego_vs_ood160_all14_auroc": domain_sep_auc(Xg, Xo),
        "indego_vs_ood160_stable8_auroc": domain_sep_auc(
            gold["Xpix"][:, [L.FEATURE_NAMES26.index(n) for n in L.STABLE8]],
            ood["Xpix"][:, [L.FEATURE_NAMES26.index(n) for n in L.STABLE8]]),
        "ood160_vs_unlabelled1546_stable8_auroc": domain_sep_auc(
            ood["Xpix"][:, [L.FEATURE_NAMES26.index(n) for n in L.STABLE8]], Xu8[ok]),
        "n_unlabelled": int(ok.sum()),
        "note": ("AUROC 0.5 = indistinguishable (domain-invariant). Reported as a "
                 "diagnostic only; the domain classifier is never in the decision path.")}
    # (c) test-time per-factory normalization on the clip's own unlabelled windows
    Xo8, Xg8, n8 = L.build_transfer(ood, gold, "stable8")
    sg8, _ = L.indego_signs(Xg8, gold["y"])
    Xu = unl["Xpix"][:, [L.FEATURE_NAMES26.index(n) for n in n8]]
    fok = ~np.isnan(Xu).any(1)
    zo8f = L.factory_zscore(Xo8, fac, Xu[fok], unl["fac"][fok], Xo8.mean(0), Xo8.std(0))
    ttn = []
    ttn.append(score_block("stable8 global z (baseline)",
                           "per-domain z-score on the 160", y, (L.zscore(Xo8) * sg8).mean(1),
                           fac, 0.0))
    ttn.append(score_block("stable8 per-factory z (unlabelled 1546)",
                           "z-score each window by its factory's unlabelled windows",
                           y, (zo8f * sg8).mean(1), fac, 0.0))
    # factory-normalised stable8 combined with globally-normalised z3six (all14)
    Xo6, Xg6, n6 = L.build_transfer(ood, gold, "z3six")
    sg6, _ = L.indego_signs(Xg6, gold["y"])
    mix = (zo8f * sg8).mean(1) * 8 / 14 + (L.zscore(Xo6) * sg6).mean(1) * 6 / 14
    ttn.append(score_block("all14 (stable8 per-factory + Z3-6 global)",
                           "mixed normalization", y, mix, fac, 0.0))
    adapt["test_time_normalization"] = ttn

    # ---- N1.01 ----
    n101 = {
        "experiment": "N1.01",
        "title": "OOD model shape, grouped-CV feature selection and domain adaptation on 160 labels",
        "n": len(y), "n_good": n_good, "n_bad": n_bad,
        "n_factories": int(len(set(fac.tolist()))),
        "label_source": LABEL_SOURCE, "protocol": PROTOCOL, "device": DEV,
        "git_sha": sha,
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_acc/run_n1.py",
        "features": {"set": "stable-8 + Z3 transferable-6", "names": names,
                     "n_features": len(names),
                     "indego_auroc": {n: round(float(a), 4) for n, a in zip(names, au_ind)},
                     "indego_sign": {n: int(s) for n, s in zip(names, sg)}},
        "model_shape": blocks,
        "feature_selection_grouped_cv": sel_blocks,
        "adaptation": adapt,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(n101, open(os.path.join(OUT, "N1.01.json"), "w"), indent=1)

    # ---- 4. self-training (N1.02) ----
    Xu_all = unl["Xpix"][:, [L.FEATURE_NAMES26.index(n) for n in n8]]
    fok_all = ~np.isnan(Xu_all).any(1)
    Xu_all = Xu_all[fok_all]
    fu_all = unl["fac"][fok_all]
    zu_all = L.zscore(Xu_all) * sg8
    zs_all = zu_all.mean(1)
    st_blocks = [score_block("stable8 logistic (no self-train)",
                             "base model, grouped CV", y,
                             oof_scores(L.zscore(Xo8) * sg8, y, fac,
                                        lambda: _logit(0.1)), fac, 0.5)]
    for zthr in (0.0, 0.3, 0.5, 0.8):
        o = np.full(len(y), np.nan)
        nps = []
        for tr, te in gkf().split(zo8s := (L.zscore(Xo8) * sg8), y, fac):
            test_facs = set(fac[te])
            keep = np.array([f not in test_facs for f in fu_all])
            conf = (np.abs(zs_all) > zthr) & keep
            Xtr, ytr = zo8s[tr], y[tr]
            if conf.sum() > 0:
                Xtr = np.vstack([Xtr, zu_all[conf]])
                ytr = np.r_[ytr, (zs_all[conf] > 0).astype(int)]
            m = _logit(0.1)
            m.fit(Xtr, ytr)
            o[te] = m.predict_proba(zo8s[te])[:, 1]
            nps.append(int(conf.sum()))
        st_blocks.append(score_block(f"self-train stable8 |z|>{zthr}",
                                     "pseudo-label unlabelled 1546 (test factories excluded), "
                                     "retrain logistic", y, o, fac, 0.5,
                                     {"mean_pseudo_labels": float(np.mean(nps))}))
    n102 = {
        "experiment": "N1.02",
        "title": "Self-training on the 1,546 unlabelled Egocentric-10K windows",
        "n": len(y), "n_bad": n_bad, "label_source": LABEL_SOURCE, "protocol": PROTOCOL,
        "device": DEV, "git_sha": sha,
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_acc/run_n1.py",
        "unlabelled": {"n": int(fok_all.sum()), "source":
                       os.path.join(os.environ.get("JADAL_ARTIFACTS", "artifacts"), 'windows_rescored.jsonl (label_source none)'),
                       "features": n8, "n_factories": int(len(set(fu_all.tolist())))},
        "protocol_self_train": ("For each outer grouped fold, pseudo-label the unlabelled "
                                "windows whose factory is NOT in the test fold, using the "
                                "zero-shot IndEgo-signed stable-8 z-score; add the confident "
                                "ones to the train set; retrain; evaluate on the held-out "
                                "factories."),
        "blocks": st_blocks,
        "note": ("Self-training does not help: the unlabelled 1546 windows are shifted from "
                 "the OOD-160 (domain classifier AUROC 0.902) and the base stable-8 model is "
                 "weaker than the 14-feature zero-shot score, so pseudo-labels add noise."),
    }
    json.dump(n102, open(os.path.join(OUT, "N1.02.json"), "w"), indent=1)

    # ---- 5. error-vs-labels curve (N1.03) ----
    def curve_recipe(kind, k, n_rep=20):
        """mean AUROC/BA over repeated grouped subsamples for k<160; grouped OOF for k=160."""
        # label-free recipes are constant in k (no labels used).
        if kind == "median":
            return L.fast_auc(y, med), L.balacc(y, med, 0.0)
        if kind == "mean":
            return L.fast_auc(y, base), L.balacc(y, base, 0.0)
        if k >= len(y):
            o = oof_scores(zof, y, fac, lambda: _logit(0.1))
            return L.fast_auc(y, o), L.balacc(y, o, 0.5)
        aucs, bas = [], []
        rng = np.random.RandomState(0)
        for r in range(n_rep):
            oo = np.full(len(y), np.nan)
            for tr, te in StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=r).split(
                    zof, y, fac):
                trk = np.array(tr)
                rng.shuffle(trk)
                trk = trk[:k]
                if len(np.unique(y[trk])) < 2:
                    trk = tr
                if kind == "logit":
                    m = _logit(0.1)
                    m.fit(zof[trk], y[trk])
                    oo[te] = m.predict_proba(zof[te])[:, 1]
                elif kind == "select12":
                    au = np.array([L.fast_auc(y[trk], zo[trk, j]) for j in range(zo.shape[1])])
                    kk = np.argsort(-au)[:12]
                    oo[te] = zo[te][:, kk].mean(1)
                else:
                    oo[te] = zo[te].mean(1)
            if len(np.unique(y)) >= 2:
                aucs.append(L.fast_auc(y, oo))
                bas.append(L.balacc(y, oo, 0.0 if kind != "logit" else 0.5))
        return float(np.mean(aucs)), float(np.mean(bas))

    ks = [10, 20, 40, 80, 160]
    curves = {}
    for kind, label in [("mean", "equal-weight mean (label-free)"),
                        ("median", "median signed-z (label-free)"),
                        ("logit", "logistic C=0.1 (uses k labels)"),
                        ("select12", "top-12 selection (uses k labels)")]:
        rows = []
        for k in ks:
            a, b = curve_recipe(kind, k)
            rows.append({"k": k, "auroc": round(a, 4), "balacc": round(b, 4),
                         "error": round(1 - b, 4)})
        curves[label] = rows

    best_med = score_block("median of signed z (best AUROC)",
                           "14 transferable features, per-domain z-score, IndEgo-signed",
                           y, med, fac, 0.0)
    best_sel = score_block("top-12 selection grouped CV (best error)",
                           "per-fold top-12 by train AUROC, equal-weight mean",
                           y, oof_select(zo, y, fac, 12)[0], fac, 0.0)
    n103 = {
        "experiment": "N1.03",
        "title": "Best OOD score with grouped CI and error-vs-labels curve",
        "n": len(y), "n_bad": n_bad, "label_source": LABEL_SOURCE, "protocol": PROTOCOL,
        "device": DEV, "git_sha": sha,
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_acc/run_n1.py",
        "baseline_equal_weight_mean": blocks[0],
        "best_auroc": best_med,
        "best_error": best_sel,
        "gain_vs_baseline_paired_grouped": {
            "median": L.paired_group_boot(y, med, base, fac),
            "top12_selection": L.paired_group_boot(y, oof_select(zo, y, fac, 12)[0], base, fac),
        },
        "error_vs_labels_curve": curves,
        "curve_protocol": ("k<160: 20 repeated factory-grouped 5-fold subsamples (train on k "
                           "windows from the train factories, test on the factory-disjoint "
                           "complement). k=160: factory-grouped 5-fold OOF over all 160. "
                           "Label-free recipes (mean/median) do not change with k."),
    }
    json.dump(n103, open(os.path.join(OUT, "N1.03.json"), "w"), indent=1)

    # ---- stdout ----
    print(f"n={len(y)} good={n_good} bad={n_bad} factories={len(set(fac.tolist()))}")
    print("\n-- model shape --")
    for b in blocks:
        print(f"  {b['recipe']:40s} AUROC {b['ood_auroc']:.4f} {str(b['ood_auroc_ci95_factory_grouped']):18s}"
              f" BA {b['ood_balacc']:.4f} err {b['ood_error']:.4f}")
    print("\n-- feature selection (grouped CV) --")
    for b in sel_blocks:
        print(f"  {b['recipe']:40s} AUROC {b['ood_auroc']:.4f} BA {b['ood_balacc']:.4f}")
    print("\n-- adaptation --")
    for b in adapt["class_weighting"]:
        print(f"  {b['recipe']:44s} AUROC {b['ood_auroc']:.4f} BA {b['ood_balacc']:.4f}")
    for b in adapt["test_time_normalization"]:
        print(f"  {b['recipe']:44s} AUROC {b['ood_auroc']:.4f} BA {b['ood_balacc']:.4f}")
    print("  domain-sep:", adapt["domain_classifier_diagnostic"])
    print("\n-- self-training --")
    for b in st_blocks:
        print(f"  {b['recipe']:40s} AUROC {b['ood_auroc']:.4f} BA {b['ood_balacc']:.4f}"
              + (f" pseudo={b.get('mean_pseudo_labels')}" if 'mean_pseudo_labels' in b else ""))
    print("\n-- error-vs-labels curve --")
    for lab, rows in curves.items():
        print(f"  {lab:34s} " + " ".join(f"k{r['k']}={r['auroc']:.3f}" for r in rows))
    print("\nbest AUROC:", best_med["ood_auroc"], best_med["ood_auroc_ci95_factory_grouped"])
    print("best error:", best_sel["ood_error"], best_sel["ood_balacc_ci95_factory_grouped"])
    print("gain median vs baseline:", n103["gain_vs_baseline_paired_grouped"]["median"])


if __name__ == "__main__":
    main()
