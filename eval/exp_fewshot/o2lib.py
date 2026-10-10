"""O2 shared helpers for E5.04-final + the 160-label few-shot adaptation curve.

CPU only, numpy + sklearn, fixed seeds (determinism rule 7).
Every number carries n + label_source + protocol + CI.

Data sources (no labels are ever edited):
  * labels   : $JADAL_BUNDLE/labels/ood160_labels.json   (160 human windows,
               138 good / 22 bad, 47 factories; one labeller)
  * scores   : $JADAL_BUNDLE/caches/ensemble/ood_scores.json (60)
               $JADAL_BUNDLE/caches/ensemble/ood_scores2.json (100)
               -> frozen ensemble_v1 per-window {usable_prob, p_eg2, p_pixel}
  * pixels   : eval/results/exp_fewshot/inputs/ood160_pixel.npz
               (26 per-pixel features, 10 fps-effective 10 s window; built by
                prep_inputs.py from the frozen decode recipe)
  * IndEgo   : $JADAL_BUNDLE/caches/gpu/features.json + manifest_gold.json
"""
from __future__ import annotations

import json
import os

import numpy as np

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "eval", "results", "exp_fewshot")
INPUTS = os.path.join(OUT, "inputs")

FEATURE_NAMES = [
    "temp_b0", "temp_b1", "temp_b2", "temp_b3",
    "frozen_frac", "flicker_idx",
    "radial_slope", "hf_ratio", "dct_energy",
    "pca_resid", "pca_top1_share",
    "luma_mean", "luma_p05", "luma_p50", "luma_p95",
    "dark_frac", "bright_frac",
    "motion_energy", "global_shift",
    "skin_frac", "illum_r", "illum_g", "illum_b",
    "rppg_pulse", "dup_hash_mean", "zncc_self",
]

OOD160_LABELS = os.path.join(BUNDLE, "labels", "ood160_labels.json")
OOD60_LABELS = os.path.join(BUNDLE, "labels", "ood_labels.json")
OOD_LABELS2 = os.path.join(BUNDLE, "labels", "ood_labels2.json")
SCORES1 = os.path.join(BUNDLE, "caches", "ensemble", "ood_scores.json")
SCORES2 = os.path.join(BUNDLE, "caches", "ensemble", "ood_scores2.json")


# ---------------- labels + scores ----------------

def load_labels160():
    """160 merged OOD windows: ids, y (1=good), factory, batch, reason."""
    d = json.load(open(OOD160_LABELS))
    wins = d["windows"]
    return {
        "ids": [w["segment_id"] for w in wins],
        "y": np.asarray([1 if w["verdict"] == "good" else 0 for w in wins], int),
        "factory": np.asarray([w["factory"] for w in wins]),
        "batch": np.asarray([w.get("batch", "?") for w in wins]),
        "reason": np.asarray([w.get("reason") or "" for w in wins]),
        "n": len(wins),
    }


def load_labels60():
    """Old 60-window batch-1 labels (for the E5.03 old-column comparison)."""
    d = json.load(open(OOD60_LABELS))
    latest = d.get("latest", d.get("items", {}))
    out = {}
    if isinstance(latest, dict):
        for sid, e in latest.items():
            v = e.get("verdict") if isinstance(e, dict) else e
            out[str(sid)] = 1 if str(v).lower() in ("good", "1", "usable", "true") else 0
    return out


def load_frozen_scores():
    """id -> {usable_prob, p_eg2, p_pixel, verdict, batch} for all 160 windows."""
    out = {}
    for path, b in ((SCORES1, "q1"), (SCORES2, "q2")):
        for s in json.load(open(path))["scores"]:
            out[s["id"]] = {
                "usable_prob": float(s["usable_prob"]),
                "p_eg2": float(s.get("p_eg2", np.nan)),
                "p_pixel": float(s.get("p_pixel", np.nan)),
                "verdict": s.get("verdict"),
                "batch": b,
            }
    return out


def load_ood160_pixel():
    return np.load(os.path.join(INPUTS, "ood160_pixel.npz"), allow_pickle=True)


def load_gold_indego():
    """150 IndEgo human gold windows + their 26 frozen per-pixel features."""
    m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    items = m["items"]
    ids = [it["id"] for it in items]
    fj = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    X = np.asarray([[float(fj[i]["full"][n]) for n in FEATURE_NAMES] for i in ids], float)
    return {"ids": ids,
            "clips": np.asarray([it["clip"] for it in items]),
            "y": np.asarray([int(it["label"]) for it in items]),
            "X26": X}


# ---------------- numeric / stats ----------------

def sigmoid(z):
    z = np.asarray(z, float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def boot_ci(y, s, n_boot=2000, seed=0):
    """Plain window bootstrap 95% CI (optimistic when ~1 window/factory)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    y = np.asarray(y); s = np.asarray(s)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def group_boot_ci(y, s, groups, n_boot=2000, seed=0, metric="auroc"):
    """Cluster bootstrap over `groups` (factories). The honest CI here.

    metric: 'auroc' | 'balacc' (balanced accuracy at 0.5).
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    fn = roc_auc_score if metric == "auroc" else balanced_accuracy_score
    rng = np.random.RandomState(seed)
    y = np.asarray(y); s = np.asarray(s); groups = np.asarray(groups)
    ug = sorted(set(groups.tolist()))
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ug), len(ug))
        idx = np.concatenate([idx_by[ug[p]] for p in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        ss = s[idx]
        try:
            vals.append(fn(yy, ss if metric == "auroc" else ss > 0.5))
        except ValueError:
            pass
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bal_acc(y, p, t=0.5):
    y = np.asarray(y); p = np.asarray(p)
    return float(((p[y == 1] > t).mean() + (p[y == 0] <= t).mean()) / 2)


def make_clf(C=1.0, balanced=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=4000,
                                            class_weight="balanced" if balanced else None))


def classify_transfer(auroc_a, auroc_b, tol=0.05):
    """E5.03 rule (unchanged): chance iff BOTH |AUROC-0.5|<tol; else stable/flip by sign."""
    da, db = abs(auroc_a - 0.5), abs(auroc_b - 0.5)
    if max(da, db) < tol:
        return "chance"
    return "stable" if (auroc_a - 0.5) * (auroc_b - 0.5) > 0 else "flip"


def git_sha(short=True):
    import subprocess
    try:
        args = ["git", "-C", REPO, "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------- grouped CV / learning curve ----------------

def grouped_cv_oof(X, y, groups, C_grid=(0.01, 0.1, 1.0, 10.0), n_splits=5, seed=0):
    """StratifiedGroupKFold OOF probs; inner grouped CV on train picks C.

    Groups = factory: a factory is never split across train/test. Returns
    (oof probs, folds meta).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.asarray(X, float); y = np.asarray(y, int); groups = np.asarray(groups)
    oof = np.full(len(y), np.nan)
    folds = []
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            folds.append({"fold": k, "skipped": True, "n_train": int(len(tr)),
                          "n_bad_train": int((y[tr] == 0).sum()), "n_test": int(len(te))})
            continue
        n_inner = min(3, max(2, len(set(groups[tr].tolist())) // 2))
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)
        best_c, best = C_grid[0], -1.0
        for c in C_grid:
            aucs = []
            for itr, ite in inner.split(X[tr], y[tr], groups[tr]):
                if len(np.unique(y[tr][itr])) < 2 or len(np.unique(y[tr][ite])) < 2:
                    continue
                clf = make_clf(c); clf.fit(X[tr][itr], y[tr][itr])
                try:
                    aucs.append(roc_auc_score(y[tr][ite], clf.predict_proba(X[tr][ite])[:, 1]))
                except ValueError:
                    pass
            m = float(np.mean(aucs)) if aucs else 0.5
            if m > best:
                best, best_c = m, c
        clf = make_clf(best_c); clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        folds.append({"fold": k, "skipped": False, "C": best_c, "n_train": int(len(tr)),
                      "n_bad_train": int((y[tr] == 0).sum()), "n_test": int(len(te)),
                      "n_bad_test": int((y[te] == 0).sum())})
    return oof, folds


def grouped_subsample_curve(X, y, groups, ks, repeats=20, C_grid=(0.01, 0.1, 1.0, 10.0), seed=0):
    """Learning curve: train on a factory-grouped subsample of ~k windows, test on
    the factory-disjoint complement (grouped CV, never a shared factory).

    For each k and each repeat: shuffle factories, add them until >= k windows,
    fit the recipe (inner grouped CV picks C), score the complement. Repeats with
    a single-class train/test side are dropped (and counted).
    """
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, float); y = np.asarray(y, int); groups = np.asarray(groups)
    facs = sorted(set(groups.tolist()))
    fac_count = {f: int((groups == f).sum()) for f in facs}
    rng = np.random.RandomState(seed)
    out = []
    for k in ks:
        aucs, bas, ntrs, nbs, dropped = [], [], [], [], 0
        for _ in range(repeats):
            order = rng.permutation(facs)
            sel, nsel = [], 0
            for f in order:
                sel.append(f); nsel += fac_count[f]
                if nsel >= k:
                    break
            selset = set(sel)
            tr = np.where(np.asarray([g in selset for g in groups]))[0]
            te = np.where(np.asarray([g not in selset for g in groups]))[0]
            if len(np.unique(y[tr])) < 2 or len(te) == 0 or len(np.unique(y[te])) < 2:
                dropped += 1
                continue
            best_c, best = C_grid[0], -1.0
            tgroups = groups[tr]
            for c in C_grid:
                aucs_c = []
                for f in sorted(set(tgroups.tolist())):
                    itr = tr[tgroups != f]; ite = tr[tgroups == f]
                    if len(np.unique(y[itr])) < 2 or len(np.unique(y[ite])) < 2:
                        continue
                    clf = make_clf(c); clf.fit(X[itr], y[itr])
                    try:
                        aucs_c.append(roc_auc_score(y[ite], clf.predict_proba(X[ite])[:, 1]))
                    except ValueError:
                        pass
                m = float(np.mean(aucs_c)) if aucs_c else 0.5
                if m > best:
                    best, best_c = m, c
            clf = make_clf(best_c); clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            try:
                aucs.append(roc_auc_score(y[te], p))
            except ValueError:
                dropped += 1; continue
            bas.append(bal_acc(y[te], p))
            ntrs.append(int(len(tr))); nbs.append(int((y[tr] == 0).sum()))
        aucs = np.asarray(aucs); bas = np.asarray(bas)
        out.append({
            "k_target": int(k),
            "k_train_mean": float(np.mean(ntrs)) if ntrs else None,
            "repeats": int(len(aucs)), "dropped": int(dropped),
            "auroc_mean": float(aucs.mean()) if len(aucs) else None,
            "auroc_std": float(aucs.std()) if len(aucs) else None,
            "balacc_mean": float(bas.mean()) if len(bas) else None,
            "balacc_std": float(bas.std()) if len(bas) else None,
            "error_mean": float(1.0 - bas.mean()) if len(bas) else None,
            "min_n_bad_train": int(min(nbs)) if nbs else None,
        })
    return out
