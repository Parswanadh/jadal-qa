"""N4 shared helpers: label-budget curves + certification budget.

CPU only (numpy/sklearn), fixed seeds. Never edits labels (read-only).

Data (read-only):
  labels/ood160_labels.json        160 human windows, 138 good / 22 bad, 47 factories
  eval/results/exp_fewshot/inputs/ood160_pixel.npz  frozen 26 per-pixel features (160)
  caches/ensemble/ood_scores{,2}.json  frozen ensemble_v1 per-window scores (160)
  $JADAL_POD_ROOT/artifacts/windows_rescored.jsonl  1,546 unlabelled Egocentric-10K windows
                                  (same 47 factories; 26 per-pixel features + view scores)

Label convention: y = 1 good/usable, y = 0 bad. error = 1 - balanced accuracy @0.5.
All label-budget curves are evaluated with FACTORY-GROUPED folds: a factory is never
in both train and test (a held-out factory is a new deployment cluster).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

BUNDLE = os.environ.get("JADAL_BUNDLE", "jadal_pod_bundle")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "eval", "results", "exp_acc")
# frozen 160-window 26-feature input (copied from the O2 few-shot worktree; input cache,
# not a label). Search a few known locations so the script is robust across worktrees.
FEWSHOT_CANDIDATES = [
    os.path.join(REPO, "eval", "results", "exp_acc", "inputs"),
    os.path.join(REPO, "eval", "results", "exp_fewshot", "inputs"),
    os.path.join(os.environ.get("JADAL_WT", "wt"), 'm-fewshot160/eval/results/exp_fewshot/inputs'),
]
RESCORED = os.path.join(os.environ.get("JADAL_ARTIFACTS", "artifacts"), 'windows_rescored.jsonl')


def _find_input(name):
    for d in FEWSHOT_CANDIDATES:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in {FEWSHOT_CANDIDATES}")

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


# --------------------------------------------------------------- data loading
def load_labelled160():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    wins = d["windows"]
    pix = np.load(_find_input("ood160_pixel.npz"), allow_pickle=True)
    pidx = {str(i): k for k, i in enumerate(pix["ids"])}
    # frozen scores
    sc = {}
    for path in (os.path.join(BUNDLE, "caches", "ensemble", "ood_scores.json"),
                 os.path.join(BUNDLE, "caches", "ensemble", "ood_scores2.json")):
        for s in json.load(open(path))["scores"]:
            sc[s["id"]] = s
    ids, y, fac, X, puse, peg, ppix = [], [], [], [], [], [], []
    for w in wins:
        sid = w["segment_id"]
        if sid not in pidx or sid not in sc:
            continue
        ids.append(sid)
        y.append(1 if w["verdict"] == "good" else 0)
        fac.append(w["factory"])
        X.append(pix["X10"][pidx[sid]])
        puse.append(float(sc[sid]["usable_prob"]))
        peg.append(float(sc[sid].get("p_eg2", np.nan)))
        ppix.append(float(sc[sid].get("p_pixel", np.nan)))
    return {
        "ids": ids, "y": np.asarray(y, int), "factory": np.asarray(fac),
        "X": np.asarray(X, float), "p_use": np.asarray(puse, float),
        "p_eg2": np.asarray(peg, float), "p_pixel": np.asarray(ppix, float),
    }


def load_unlabelled():
    """1,546 unlabelled windows: id, factory, clip(video), 26 features, view scores."""
    rows = []
    with open(RESCORED) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "header":
                continue
            feats = d["features"]
            rows.append({
                "window_id": d["window_id"],
                "factory": d["factory"],
                "clip": d["clip"],
                "worker": d.get("worker"),
                "start_s": d.get("start_s"),
                "end_s": d.get("end_s"),
                "score": float(d.get("score", np.nan)),
                "p_eg2": float(feats.get("p_eg2", np.nan)),
                "p_pixel": float(feats.get("p_pixel", np.nan)),
                "X": [float(feats.get(n, np.nan)) for n in FEATURE_NAMES],
            })
    return rows


# --------------------------------------------------------------- strategies
def _zscore(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z)


def order_random(X, y, fac, seed=0):
    rng = np.random.RandomState(seed)
    return rng.permutation(len(y))


def order_breadth_new_factories(X, y, fac):
    """Round-robin: one window from each new factory before a second from any.
    Maximises the number of independent clusters per label."""
    from collections import Counter
    idx_by = {f: list(np.where(fac == f)[0]) for f in sorted(set(fac.tolist()))}
    counts = Counter()
    order = []
    remaining = {f: list(v) for f, v in idx_by.items()}
    while remaining:
        f = min(remaining, key=lambda g: (counts[g], -len(remaining[g]), g))
        order.append(remaining[f].pop(0))
        counts[f] += 1
        if not remaining[f]:
            del remaining[f]
    return np.asarray(order, int)


def order_depth_existing_factories(X, y, fac):
    """Exhaust one factory before the next; factories ordered by descending count.
    Maximises windows-per-factory (the opposite extreme)."""
    from collections import Counter
    cnt = Counter(fac.tolist())
    facs = sorted(cnt, key=lambda f: (-cnt[f], f))
    order = []
    for f in facs:
        order += list(np.where(fac == f)[0])
    return np.asarray(order, int)


def order_diversity_maximin(X, y, fac):
    """Greedy farthest-first (maximin) over z-scored 26-D features: each new label
    is the window furthest (min-distance) from the already-chosen set. Coverage of
    the feature/regime space, blind to the model."""
    Z = _zscore(X)
    n = len(Z)
    start = int(np.argmax(np.linalg.norm(Z - Z.mean(0), axis=1)))
    chosen = [start]
    mind = np.linalg.norm(Z - Z[start], axis=1)
    mind[start] = -1
    while len(chosen) < n:
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        d = np.linalg.norm(Z - Z[nxt], axis=1)
        mind = np.minimum(mind, d)
        mind[chosen] = -1
    return np.asarray(chosen, int)


def order_uncertainty_boundary(X, y, fac, p=None):
    """Z4-style: closest to the decision boundary (max 4p(1-p)). Negative control
    (Z4: model uncertainty is ANTI-correlated with true OOD errors, AUROC 0.349)."""
    if p is None:
        p = 0.5 * (X[:, 0] + X[:, 1])  # fallback; callers pass p_use
    U = 4.0 * np.asarray(p, float) * (1.0 - np.asarray(p, float))
    return np.argsort(-U)


STRATEGIES = {
    "random": "seeded uniform random permutation (averaged over seeds)",
    "breadth_new_factories": "round-robin: one window per new factory first (maximise clusters)",
    "depth_existing_factories": "exhaust factories in descending size (maximise windows/factory)",
    "diversity_maximin": "greedy farthest-first over z-scored 26-D regime features (coverage)",
    "uncertainty_boundary": "closest to p=0.5 (Z4 uncertainty; negative control)",
}


def build_order(name, X, y, fac, p=None, seed=0):
    if name == "random":
        return order_random(X, y, fac, seed=seed)
    if name == "breadth_new_factories":
        return order_breadth_new_factories(X, y, fac)
    if name == "depth_existing_factories":
        return order_depth_existing_factories(X, y, fac)
    if name == "diversity_maximin":
        return order_diversity_maximin(X, y, fac)
    if name == "uncertainty_boundary":
        return order_uncertainty_boundary(X, y, fac, p=p)
    raise ValueError(name)


# --------------------------------------------------------------- evaluation
def make_clf(C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def bal_acc(y, p, t=0.5):
    y = np.asarray(y); p = np.asarray(p)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(((p[y == 1] > t).mean() + (p[y == 0] <= t).mean()) / 2)


def _folds(y, fac, n_splits=5, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, fac))


def label_budget_curve(strategy, data, ks, n_splits=5, seed=0, seeds=(0, 1, 2, 3, 4),
                       C=1.0):
    """Factory-grouped label-budget curve for one acquisition strategy.

    For each fold (held-out factories = test), acquire labels from the pool in the
    strategy's order; at each k fit a 26-feature logistic on the first k and score
    the held-out factories. `random` is averaged over `seeds`; deterministic
    strategies run once per fold. Returns per-k mean/std over folds (and seeds).
    """
    from sklearn.metrics import roc_auc_score
    X, y, fac, p = data["X"], data["y"], data["factory"], data["p_use"]
    folds = _folds(y, fac, n_splits=n_splits, seed=seed)
    per_k = {int(k): {"err": [], "auc": [], "nbad_tr": [], "nbad_te": [], "ntr": [], "nte": []}
             for k in ks}
    runs = seeds if strategy == "random" else (0,)
    for sd in runs:
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            order = build_order(strategy, X[tr], y[tr], fac[tr],
                                p=p[tr] if p is not None else None, seed=sd)
            for k in ks:
                kk = min(int(k), len(tr))
                sel = order[:kk]
                if len(np.unique(y[tr][sel])) < 2:
                    continue
                clf = make_clf(C)
                clf.fit(X[tr][sel], y[tr][sel])
                pr = clf.predict_proba(X[te])[:, 1]
                ba = bal_acc(y[te], pr)
                if np.isnan(ba):
                    continue
                per_k[int(k)]["err"].append(1.0 - ba)
                per_k[int(k)]["auc"].append(float(roc_auc_score(y[te], pr)))
                per_k[int(k)]["nbad_tr"].append(int((y[tr][sel] == 0).sum()))
                per_k[int(k)]["nbad_te"].append(int((y[te] == 0).sum()))
                per_k[int(k)]["ntr"].append(int(len(sel)))
                per_k[int(k)]["nte"].append(int(len(te)))
    curve = []
    for k in ks:
        d = per_k[int(k)]
        curve.append({
            "k_target": int(k),
            "n_runs": len(d["err"]),
            "error_mean": float(np.mean(d["err"])) if d["err"] else None,
            "error_std": float(np.std(d["err"])) if d["err"] else None,
            "auroc_mean": float(np.mean(d["auc"])) if d["auc"] else None,
            "auroc_std": float(np.std(d["auc"])) if d["auc"] else None,
            "min_n_bad_train": int(min(d["nbad_tr"])) if d["nbad_tr"] else None,
            "mean_n_bad_train": float(np.mean(d["nbad_tr"])) if d["nbad_tr"] else None,
            "mean_n_test": float(np.mean(d["nte"])) if d["nte"] else None,
        })
    return curve


# --------------------------------------------------------------- certification
def label_budget_paired(data, ks, strategies=None, n_splits=5, seed=0,
                        seeds=(0, 1, 2, 3, 4), C=1.0):
    """Paired per-fold errors for every strategy vs `random` (same held-out factories).

    Returns per-strategy mean error, mean paired delta vs random, its std over folds,
    and the fraction of folds where the strategy beats random. Paired comparisons are
    the honest way to read this data: the folds are shared and n_bad is small, so the
    independent-seed std above overstates the noise.
    """
    from sklearn.metrics import roc_auc_score
    strategies = strategies or list(STRATEGIES)
    X, y, fac, p = data["X"], data["y"], data["factory"], data["p_use"]
    folds = _folds(y, fac, n_splits=n_splits, seed=seed)
    per = {nm: {int(k): [] for k in ks} for nm in strategies}
    rand = {int(k): [] for k in ks}
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        # random: per-fold mean over seeds
        rvals = {int(k): [] for k in ks}
        for sd in seeds:
            order = build_order("random", X[tr], y[tr], fac[tr], seed=sd)
            for k in ks:
                sel = order[:min(int(k), len(tr))]
                if len(np.unique(y[tr][sel])) < 2:
                    continue
                clf = make_clf(C); clf.fit(X[tr][sel], y[tr][sel])
                ba = bal_acc(y[te], clf.predict_proba(X[te])[:, 1])
                if not np.isnan(ba):
                    rvals[int(k)].append(1.0 - ba)
        for k in ks:
            rand[int(k)].append(float(np.mean(rvals[int(k)])) if rvals[int(k)] else np.nan)
        for nm in strategies:
            order = build_order(nm, X[tr], y[tr], fac[tr],
                                p=p[tr] if p is not None else None, seed=0)
            for k in ks:
                sel = order[:min(int(k), len(tr))]
                if len(np.unique(y[tr][sel])) < 2:
                    per[nm][int(k)].append(np.nan); continue
                clf = make_clf(C); clf.fit(X[tr][sel], y[tr][sel])
                ba = bal_acc(y[te], clf.predict_proba(X[te])[:, 1])
                per[nm][int(k)].append(1.0 - ba if not np.isnan(ba) else np.nan)
    out = {}
    for nm in strategies:
        rows = []
        for k in ks:
            a = np.asarray(per[nm][int(k)], float)
            r = np.asarray(rand[int(k)], float)
            m = ~np.isnan(a) & ~np.isnan(r)
            d = a[m] - r[m]
            rows.append({
                "k_target": int(k),
                "n_folds": int(m.sum()),
                "error_mean": float(np.nanmean(a)) if m.any() else None,
                "random_mean": float(np.nanmean(r)) if m.any() else None,
                "paired_delta_vs_random_mean": float(d.mean()) if m.any() else None,
                "paired_delta_std": float(d.std(ddof=1)) if m.sum() > 1 else None,
                "frac_folds_better_than_random": float((d < 0).mean()) if m.any() else None,
            })
        out[nm] = rows
    return out


def cert_budget_table(delta=0.05, grid=46, rhat=0.0):
    """K (exchangeable clusters) to certify a marginal false-accept bound r at
    confidence 1-delta, via the cluster-Hoeffding UCB
        UCB = Rhat + sqrt( ln(1/delta') / (2K) ) <= r
    =>  K >= ln(1/delta') / (2 (r - Rhat)^2).
    delta' = delta (single pre-registered threshold) or delta/grid (union bound
    over a threshold grid, as Z2/LTT does)."""
    out = {}
    for r in (0.05, 0.10, 0.20):
        row = {}
        for label, dp in (("single_threshold", delta), ("grid_union", delta / grid)):
            denom = 2.0 * (r - rhat) ** 2
            K = math.ceil(math.log(1.0 / dp) / denom) if denom > 0 else None
            row[label] = {"K_clusters": K, "delta_prime": dp}
        out[f"{r:.2f}"] = row
    return out


def certifiable_r(K, delta=0.05, grid=46, rhat=0.0):
    """Largest r certifiable with K clusters (invert the bound)."""
    dp = delta / grid
    return rhat + math.sqrt(math.log(1.0 / dp) / (2.0 * K))


# --------------------------------------------------------------- misc
def git_sha(short=True):
    import subprocess
    try:
        args = ["git", "-C", REPO, "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"
