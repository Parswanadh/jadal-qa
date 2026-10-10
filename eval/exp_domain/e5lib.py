"""E5 shared helpers: loaders, bootstrap CIs, LOCO / grouped CV.

CPU only, numpy + sklearn. Fixed seeds everywhere (determinism rule).
Every number produced here must carry n + label_source + protocol + CI.
"""
from __future__ import annotations

import json
import os

import numpy as np

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "eval", "results", "exp_domain")
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


# ---------------- numeric helpers ----------------

def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def boot_ci(y, s, fn, n_boot=2000, seed=0):
    """Plain (window) bootstrap 95% CI. Optimistic when ~1 window/factory."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def group_boot_ci(y, s, groups, n_boot=2000, seed=0):
    """Cluster bootstrap over `groups` (clips / factories)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    groups = np.asarray(groups)
    ug = np.asarray(sorted(set(groups.tolist())))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.where(np.isin(groups, pick))[0]
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pool_feats(P):
    """P: (n, T) nan-padded frame probs -> (n, 6): mean,max,p25,p75,frac>0.5,var."""
    feats = []
    for row in P:
        v = row[~np.isnan(row)]
        if len(v) == 0:
            feats.append([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
            continue
        feats.append([v.mean(), v.max(), np.percentile(v, 25), np.percentile(v, 75),
                      (v > 0.5).mean(), v.var()])
    return np.asarray(feats, dtype=np.float64)


def make_clf(C=1.0, balanced=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000, class_weight="balanced" if balanced else None))


def loco_cv(X, y, clips, C_grid=(0.01, 0.1, 1.0, 10.0), seed=0):
    """Leave-one-clip-out OOF probs; inner LOCO on train clips picks C."""
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    clips = np.asarray(clips)
    oof = np.full(len(y), np.nan)
    for held in sorted(set(clips.tolist())):
        te = np.where(clips == held)[0]
        tr = np.where(clips != held)[0]
        best_c, best = C_grid[0], -1.0
        for c in C_grid:
            aucs = []
            tclips = clips[tr]
            for h2 in sorted(set(tclips.tolist())):
                itr = tr[tclips != h2]
                ite = tr[tclips == h2]
                if len(np.unique(y[ite])) < 2 or len(np.unique(y[itr])) < 2:
                    continue
                clf = make_clf(c)
                clf.fit(X[itr], y[itr])
                try:
                    aucs.append(roc_auc_score(y[ite], clf.predict_proba(X[ite])[:, 1]))
                except ValueError:
                    pass
            m = float(np.mean(aucs)) if aucs else 0.5
            if m > best:
                best, best_c = m, c
        clf = make_clf(best_c)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def grouped_cv(X, y, groups, C_grid=(0.01, 0.1, 1.0, 10.0), n_splits=5, seed=0):
    """Grouped CV (groups = factory): returns (oof probs, fold meta).

    Inner grouped CV on the train groups picks C. 5 splits as far as the
    class balance allows; reports the smallest n_bad seen in any train fold.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    oof = np.full(len(y), np.nan)
    folds = []
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            folds.append({"fold": k, "skipped": True,
                          "n_train": int(len(tr)), "n_bad_train": int((y[tr] == 0).sum()),
                          "n_test": int(len(te)), "n_bad_test": int((y[te] == 0).sum())})
            continue
        best_c, best = C_grid[0], -1.0
        inner = StratifiedGroupKFold(n_splits=min(3, max(2, len(set(groups[tr].tolist())) // 2)),
                                     shuffle=True, random_state=seed + 1)
        for c in C_grid:
            aucs = []
            for itr, ite in inner.split(X[tr], y[tr], groups[tr]):
                if len(np.unique(y[tr][itr])) < 2 or len(np.unique(y[tr][ite])) < 2:
                    continue
                clf = make_clf(c)
                clf.fit(X[tr][itr], y[tr][itr])
                try:
                    aucs.append(roc_auc_score(y[tr][ite], clf.predict_proba(X[tr][ite])[:, 1]))
                except ValueError:
                    pass
            m = float(np.mean(aucs)) if aucs else 0.5
            if m > best:
                best, best_c = m, c
        clf = make_clf(best_c)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        folds.append({"fold": k, "skipped": False, "C": best_c,
                      "n_train": int(len(tr)), "n_bad_train": int((y[tr] == 0).sum()),
                      "n_test": int(len(te)), "n_bad_test": int((y[te] == 0).sum())})
    return oof, folds


# ---------------- loaders ----------------

def load_gold():
    """150 IndEgo human windows: ids, clips, labels(1=usable), reasons + 26 feats."""
    m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    items = m["items"]
    ids = [it["id"] for it in items]
    clips = np.asarray([it["clip"] for it in items])
    y = np.asarray([int(it["label"]) for it in items])
    reasons = np.asarray([it.get("reason") or "" for it in items])
    fj = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    X = np.asarray([[float(fj[i]["full"][n]) for n in FEATURE_NAMES] for i in ids], dtype=np.float64)
    return {"ids": ids, "clips": clips, "y": y, "reasons": reasons, "X26": X}


def load_gold_segments():
    z = np.load(os.path.join(BUNDLE, "caches", "frames10k", "segments.npz"), allow_pickle=True)
    return {"F": z["F"].astype(np.float64), "ids": [str(x) for x in z["ids"]],
            "clips": np.asarray([str(x) for x in z["clips"]]),
            "y": z["labels"].astype(int), "counts": z["counts"]}


def load_10k_frames():
    d = os.path.join(BUNDLE, "caches", "frames10k", "egocentric_10k_frame_embeddings")
    return {"E": np.load(os.path.join(d, "embeddings.fp16.npy")).astype(np.float64),
            "hand": np.load(os.path.join(d, "hand.npy")),
            "work": np.load(os.path.join(d, "work.npy")),
            "meta": json.load(open(os.path.join(d, "meta.json")))}


def load_ood():
    """60 Egocentric-10K human-labelled windows: y, factory, ids."""
    lab = json.load(open(os.path.join(BUNDLE, "labels", "ood_labels.json")))["latest"]
    q = json.load(open(os.path.join(BUNDLE, "queues", "label_queue_random.json")))["segments"]
    qmap = {s["segment_id"]: s for s in q}
    ids, y, fac = [], [], []
    for sid in qmap:
        v = lab.get(sid)
        if v is None:
            continue
        ids.append(sid)
        y.append(1 if v["verdict"] == "good" else 0)
        fac.append(qmap[sid].get("factory"))
    return {"ids": ids, "y": np.asarray(y, int), "factory": np.asarray(fac)}


def load_ood_pixel_selfdec():
    """OOD 26 features decoded from the label_windows transcodes (frozen recipe).

    Two fps views per window: native (~30fps for the transcodes) and 10fps
    effective (idx = arange(0,T,3), cpu_window_features(fps=10)).
    """
    return np.load(os.path.join(INPUTS, "ood_pixel_selfdec.npz"), allow_pickle=True)


def load_ood_eg2_means():
    """L2-normalised EG2 window-mean embeddings for the 60 OOD windows."""
    return np.load(os.path.join(INPUTS, "ood_eg2_means.npz"), allow_pickle=True)


def load_frozen_ood_scores():
    """Frozen ensemble scores for the 60 batch-1 OOD windows."""
    return json.load(open(os.path.join(BUNDLE, "caches", "ensemble", "ood_scores.json")))["scores"]


def load_labels_file(path):
    """Handles {latest:{id:{verdict}}} and {items:[{id,label}]} schemas."""
    d = json.load(open(path))
    out = {}
    if isinstance(d, dict) and isinstance(d.get("latest"), dict):
        for wid, e in d["latest"].items():
            v = str(e.get("verdict", e.get("label", ""))).lower() if isinstance(e, dict) else str(e).lower()
            if v in ("good", "1", "1.0", "true"):
                out[str(wid)] = 1
            elif v in ("bad", "0", "0.0", "false"):
                out[str(wid)] = 0
        return out
    items = d.get("items", d.get("segments")) if isinstance(d, dict) else d
    if isinstance(items, dict):
        items = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "label": v} for k, v in items.items()]
    for it in items or []:
        if not isinstance(it, dict):
            continue
        wid = it.get("id", it.get("segment_id"))
        v = str(it.get("verdict", it.get("label", ""))).lower()
        if wid is None:
            continue
        if v in ("good", "1", "1.0", "true", "usable"):
            out[str(wid)] = 1
        elif v in ("bad", "0", "0.0", "false"):
            out[str(wid)] = 0
    return out


def git_sha():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"
