"""Z1 domain-adaptation helpers: loaders, transfer heads, alignment, metrics.

CPU only (numpy + sklearn). Fixed seeds. Reuses the frozen pod bundle
(``$JADAL_BUNDLE``, read-only) and the E5 OOD inputs (``$JADAL_E5_INPUTS``,
default the sibling E5 worktree; see ``README.md`` for how to regenerate them).

Every number carries n + label_source + protocol + CI (AGENTS.md rule 1).
"""
from __future__ import annotations

import json
import os

import numpy as np

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
# E5-derived OOD inputs (ood_pixel_selfdec.npz, ood_eg2_means.npz); the *.npz are
# git-ignored in E5. Regenerate with eval/exp_domain/prep_inputs.py (see E5 report).
E5_INPUTS = os.environ.get(
    "JADAL_E5_INPUTS",
    "/workspace/wt/exp-domain-adapt/eval/results/exp_domain/inputs")
E8_JSONL = os.environ.get("JADAL_E8_JSONL", "/workspace/artifacts/windows_rescored.jsonl")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "eval", "results", "exp_adapt")
FIGS = os.path.join(REPO, "docs", "experiments", "exp_adapt", "images")

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

# E5.03 sign-stable set (IndEgo vs OOD-10fps same side of 0.5).
STABLE = ["radial_slope", "hf_ratio", "dct_energy", "luma_p05", "global_shift",
          "illum_r", "illum_b", "zncc_self"]
# the sharpness / spatial-spectrum group only (the mechanistically-motivated core).
SHARPNESS = ["radial_slope", "hf_ratio", "dct_energy"]
CHANCE = ["temp_b3", "flicker_idx", "dark_frac", "dup_hash_mean"]

C_GRID = (0.01, 0.1, 1.0, 10.0)


def idx_of(names):
    return [FEATURE_NAMES.index(n) for n in names]


# ---------------- metrics ----------------

def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def balacc(y, s, thr=0.5):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, (np.asarray(s) > thr).astype(int)))


def boot_ci(y, s, n_boot=2000, seed=0):
    """Window bootstrap 95% CI (optimistic when ~1 window/factory)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    y = np.asarray(y); s = np.asarray(s)
    vals = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def clip_boot_ci(y, s, clips, n_boot=2000, seed=0):
    """Cluster bootstrap over clips/factories."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    y = np.asarray(y); s = np.asarray(s); clips = np.asarray(clips)
    ug = np.asarray(sorted(set(clips.tolist())))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        i = np.where(np.isin(clips, pick))[0]
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------- loaders ----------------

def load_gold():
    """150 IndEgo human windows: X26, y(1=usable), clips."""
    m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    items = m["items"]
    ids = [it["id"] for it in items]
    clips = np.asarray([it["clip"] for it in items])
    y = np.asarray([int(it["label"]) for it in items])
    fj = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    X = np.asarray([[float(fj[i]["full"][n]) for n in FEATURE_NAMES] for i in ids])
    return {"ids": ids, "clips": clips, "y": y, "X26": X}


def _mean_l2(F):
    """nan-aware L2-normalised mean over frames -> (n, d)."""
    out = np.empty((F.shape[0], F.shape[2]), dtype=np.float64)
    for i, row in enumerate(F.astype(np.float64)):
        v = row[~np.isnan(row)]
        m = v.mean(axis=0) if len(v) else np.zeros(F.shape[2])
        out[i] = m / max(np.linalg.norm(m), 1e-12)
    return out


def load_gold_eg2():
    z = np.load(os.path.join(BUNDLE, "caches", "frames10k", "segments.npz"), allow_pickle=True)
    return {"F": z["F"].astype(np.float64), "means": _mean_l2(z["F"]),
            "ids": [str(x) for x in z["ids"]], "y": z["labels"].astype(int)}


def load_ood_pixel():
    z = np.load(os.path.join(E5_INPUTS, "ood_pixel_selfdec.npz"), allow_pickle=True)
    return {"ids": [str(x) for x in z["ids"]], "y": z["y"].astype(int),
            "factory": np.asarray([str(x) for x in z["factory"]]),
            "names": [str(x) for x in z["names"]],
            "X_native": z["X_native"].astype(np.float64),
            "X_10fps": z["X_10fps"].astype(np.float64)}


def load_ood_eg2():
    z = np.load(os.path.join(E5_INPUTS, "ood_eg2_means.npz"), allow_pickle=True)
    return {"ids": [str(x) for x in z["ids"]], "y": z["y"].astype(int),
            "factory": np.asarray([str(x) for x in z["factory"]]),
            "means": z["means"].astype(np.float64)}


def load_frozen_ood_scores():
    d = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", "ood_scores.json")))["scores"]
    return {s["id"]: float(s["usable_prob"]) for s in d}


def load_e8_corpus(feature_names=None):
    """1546 unlabelled Egocentric-10K windows (E8 rescored), 26 pixel feats @ native 30fps."""
    feature_names = feature_names or FEATURE_NAMES
    ids, X, fac = [], [], []
    base = {"cluster": [], "worker": []}
    for line in open(E8_JSONL):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "header":
            continue
        f = d.get("features", {})
        if not all(k in f for k in feature_names):
            continue
        ids.append(d.get("window_id"))
        X.append([float(f[k]) for k in feature_names])
        fac.append(d.get("factory"))
    return {"ids": ids, "X": np.asarray(X, dtype=np.float64),
            "factory": np.asarray(fac)}


# ---------------- plain classifier helpers ----------------

def make_clf(C=1.0, balanced=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000, class_weight="balanced" if balanced else None))


def fit_head(X, y, C, balanced=True):
    clf = make_clf(C, balanced)
    clf.fit(X, y)
    return clf


def pick_C(X, y, clips_or_groups, inner="loco", seed=0):
    """Choose C by inner CV on the training set only (no test leakage)."""
    from sklearn.metrics import roc_auc_score
    g = np.asarray(clips_or_groups)
    best_c, best = C_GRID[0], -1.0
    ug = sorted(set(g.tolist()))
    for c in C_GRID:
        aucs = []
        for held in ug:
            itr = np.where(g != held)[0]
            ite = np.where(g == held)[0]
            if len(np.unique(y[itr])) < 2 or len(np.unique(y[ite])) < 2:
                continue
            clf = fit_head(X[itr], y[itr], c)
            try:
                aucs.append(roc_auc_score(y[ite], clf.predict_proba(X[ite])[:, 1]))
            except ValueError:
                pass
        m = float(np.mean(aucs)) if aucs else 0.5
        if m > best:
            best, best_c = m, c
    return best_c


# ---------------- unsupervised alignment transforms ----------------

def _cov(X, shrink=0.1):
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(0)
    n = len(Xc)
    S = (Xc.T @ Xc) / max(n - 1, 1)
    d = S.shape[0]
    tr = np.trace(S) / d
    return (1 - shrink) * S + shrink * tr * np.eye(d)


def _sqrtm(A, inv=False, eps=1e-8):
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    w = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return (V * w) @ V.T


class Identity:
    def fit(self, Xs, Xt=None):
        return self

    def transform(self, X):
        return np.asarray(X, float)


class DomainStd:
    """z-score each domain by its own (unlabelled) mean/std."""
    def fit(self, Xs, Xt=None):
        self.ms, self.ss = Xs.mean(0), np.clip(Xs.std(0), 1e-9, None)
        if Xt is not None:
            self.mt, self.st = Xt.mean(0), np.clip(Xt.std(0), 1e-9, None)
        return self

    def transform(self, X, domain="s"):
        X = np.asarray(X, float)
        m, s = (self.ms, self.ss) if domain == "s" else (self.mt, self.st)
        return (X - m) / s


class DomainWhiten:
    """ZCA-whiten each domain by its own (shrunk) covariance -> both unit-cov."""
    def fit(self, Xs, Xt=None, shrink=0.1):
        self.ms = Xs.mean(0)
        self.Ws = _sqrtm(_cov(Xs, shrink), inv=True)
        self.mt = self.Wt = None
        if Xt is not None:
            self.mt = np.asarray(Xt).mean(0)
            self.Wt = _sqrtm(_cov(Xt, shrink), inv=True)
        return self

    def transform(self, X, domain="s"):
        X = np.asarray(X, float)
        m, W = (self.ms, self.Ws) if domain == "s" else (self.mt, self.Wt)
        return (X - m) @ W


class Coral:
    """CORAL (Sun & Saenko 2016): map source onto target (or vice-versa)."""
    def fit(self, Xs, Xt, reg=0.01, shrink=0.1, direction="s2t"):
        self.direction = direction
        self.mu_s, self.mu_t = Xs.mean(0), Xt.mean(0)
        self.Ws = _sqrtm(_cov(Xs, shrink) + reg * np.eye(Xs.shape[1]))
        self.Wt = _sqrtm(_cov(Xt, shrink) + reg * np.eye(Xt.shape[1]))
        return self

    def transform(self, X, domain="s"):
        X = np.asarray(X, float)
        if self.direction == "s2t":
            return (X - self.mu_s) @ self.Ws @ self.Wt + self.mu_t
        return (X - self.mu_t) @ self.Wt @ self.Ws + self.mu_s


# ---------------- per-clip / per-group normalisation ----------------

def group_norm_fit(X, groups):
    """Per-group z-score stats (mean/std within each group)."""
    groups = np.asarray(groups)
    stats = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        stats[g] = (X[m].mean(0), np.clip(X[m].std(0), 1e-9, None))
    return stats


def group_norm_apply(X, groups, stats, fallback=None):
    groups = np.asarray(groups)
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if g in stats:
            mu, sd = stats[g]
        elif fallback is not None:
            mu, sd = fallback
        else:
            mu, sd = X[m].mean(0), np.clip(X[m].std(0), 1e-9, None)
        out[m] = (X[m] - mu) / sd
    return out


def group_rank_apply(X, groups):
    """Within-group rank-normalise each feature to (0,1) (no params to leak)."""
    from scipy.stats import rankdata
    groups = np.asarray(groups)
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for g in sorted(set(groups.tolist())):
        m = groups == g
        sub = X[m]
        if len(sub) == 1:
            out[m] = 0.5
            continue
        for j in range(X.shape[1]):
            out[m, j] = (rankdata(sub[:, j]) - 1) / (len(sub) - 1)
    return out


# ---------------- transfer evaluation ----------------

def _fit_transformers(method, Xsrc, Xtgt):
    """Fit an unsupervised transform. Return (fs, ft) callables source/target.

    Target = unlabelled target-domain features (OOD-60 and/or Ego10K corpus);
    no target labels are ever used. Domain methods keep source and target in
    their own spaces; CORAL maps one onto the other.
    """
    if method == "raw":
        return (lambda X: np.asarray(X, float)), (lambda X: np.asarray(X, float))
    if method == "domain_std":
        t = DomainStd().fit(Xsrc, Xtgt)
        return (lambda X: t.transform(X, "s")), (lambda X: t.transform(X, "t"))
    if method == "domain_whiten":
        t = DomainWhiten().fit(Xsrc, Xtgt)
        return (lambda X: t.transform(X, "s")), (lambda X: t.transform(X, "t"))
    if method == "coral_s2t":
        t = Coral().fit(Xsrc, Xtgt, direction="s2t")
        return (lambda X: t.transform(X, "s")), (lambda X: np.asarray(X, float))
    if method == "coral_t2s":
        t = Coral().fit(Xsrc, Xtgt, direction="t2s")
        return (lambda X: np.asarray(X, float)), (lambda X: t.transform(X, "t"))
    raise ValueError(method)


def loco_transfer(X, y, clips, Xood, method="raw", feat_idx=None, ood_norm="none"):
    """LOCO OOF on IndEgo (source) + OOD score (head fit on all source).

    Alignment transforms are refit inside each LOCO fold on train clips only;
    the target distribution is the full unlabelled target set. ``ood_norm``:
      none | global_z | global_rank  (unsupervised target normalisation).
    """
    X = np.asarray(X, float); y = np.asarray(y); clips = np.asarray(clips)
    Xood = np.asarray(Xood, float)
    if feat_idx is not None:
        X = X[:, feat_idx]; Xood = Xood[:, feat_idx]
    oof = np.full(len(y), np.nan)
    for held in sorted(set(clips.tolist())):
        te = np.where(clips == held)[0]; tr = np.where(clips != held)[0]
        fs, _ = _fit_transformers(method, X[tr], Xood)
        Xtr = fs(X[tr]); Xte = fs(X[te])
        c = pick_C(Xtr, y[tr], clips[tr])
        clf = fit_head(Xtr, y[tr], c)
        oof[te] = clf.predict_proba(Xte)[:, 1]
    fs, ft = _fit_transformers(method, X, Xood)
    Xtr = fs(X); Xt = ft(Xood)
    if ood_norm == "global_z":
        Xt = (Xt - Xt.mean(0)) / np.clip(Xt.std(0), 1e-9, None)
    elif ood_norm == "global_rank":
        from scipy.stats import rankdata
        Xt = np.column_stack([(rankdata(Xt[:, j]) - 1) / (len(Xt) - 1) for j in range(Xt.shape[1])])
    c = pick_C(Xtr, y, clips)
    clf = fit_head(Xtr, y, c)
    return oof, clf.predict_proba(Xt)[:, 1]


def per_clip_transfer(X, y, clips, Xood, feat_idx=None, mode="z", ood_norm="global"):
    """Per-clip norm on IndEgo (removes clip-level shift, e.g. pallet/pallet2).

    OOD has ~1 window per factory, so per-factory normalisation is degenerate;
    ``ood_norm`` global = z-score / rank the whole unlabelled OOD block instead.
    """
    X = np.asarray(X, float); Xood = np.asarray(Xood, float); clips = np.asarray(clips)
    if feat_idx is not None:
        X = X[:, feat_idx]; Xood = Xood[:, feat_idx]
    if mode == "z":
        Xn = group_norm_apply(X, clips, group_norm_fit(X, clips))
    else:
        Xn = group_rank_apply(X, clips)
    Xon = Xood.copy()
    if ood_norm == "global":
        if mode == "z":
            Xon = (Xon - Xon.mean(0)) / np.clip(Xon.std(0), 1e-9, None)
        else:
            from scipy.stats import rankdata
            Xon = np.column_stack([(rankdata(Xon[:, j]) - 1) / (len(Xon) - 1)
                                   for j in range(Xon.shape[1])])
    oof = np.full(len(y), np.nan)
    for held in sorted(set(clips.tolist())):
        tr = np.where(clips != held)[0]; te = np.where(clips == held)[0]
        c = pick_C(Xn[tr], y[tr], clips[tr])
        clf = fit_head(Xn[tr], y[tr], c)
        oof[te] = clf.predict_proba(Xn[te])[:, 1]
    c = pick_C(Xn, y, clips)
    clf = fit_head(Xn, y, c)
    return oof, clf.predict_proba(Xon)[:, 1]


def grouped_cv_fewshot(X, y, groups, k, n_repeats=40, seed=0):
    """Few-shot: train on exactly ~k labelled windows chosen by whole factories.

    Returns list of test AUROCs over repeats (test = held-out factories). Uses all
    remaining factories as the held-out test so no factory appears on both sides.
    """
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, float); y = np.asarray(y); groups = np.asarray(groups)
    facs = np.asarray(sorted(set(groups.tolist())))
    rng = np.random.RandomState(seed)
    res = []
    for _ in range(n_repeats):
        order = rng.permutation(facs)
        sel, n = [], 0
        for f in order:
            sel.append(f); n = int(np.isin(groups, sel).sum())
            if n >= k:
                break
        tr = np.where(np.isin(groups, sel))[0]
        te = np.where(~np.isin(groups, sel))[0]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        c = pick_C(X[tr], y[tr], groups[tr])
        clf = fit_head(X[tr], y[tr], c)
        try:
            res.append({"auroc": float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])),
                        "n_train": int(len(tr)), "n_bad_train": int((y[tr] == 0).sum())})
        except ValueError:
            pass
    return res


def git_sha():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"
