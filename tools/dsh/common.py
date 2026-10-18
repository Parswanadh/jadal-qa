"""Shared loaders, metrics and results-JSON schema for the GRID/AUTO/TASK tracks.

One owner: stream S0 (`dsh/infra`). Other streams import, never fork.

Conventions enforced here (brief sections 3 and 7.4):
  * every CI that touches factory footage is *grouped* by factory; the window
    bootstrap is kept for continuity but is marked optimistic;
  * clip-level bootstrap for in-domain IndEgo (7 clips);
  * every results JSON carries n, n_bad, label_source, protocol, ci95 + ci95_type,
    device, cost, command, git_sha, seed, config_sha256.

CPU only, stdlib + numpy + scikit-learn.  No network.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: pod bundle (labels + caches).  Override with JADAL_BUNDLE.
BUNDLE = os.environ.get("JADAL_BUNDLE") or os.path.join(
    os.path.dirname(REPO), "handover", "pod_bundle")

#: the 26 shipped per-pixel feature names, in the canonical order
FEATURE_NAMES = [
    "temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac", "flicker_idx",
    "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "pca_top1_share",
    "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac",
    "motion_energy", "global_shift", "skin_frac", "illum_r", "illum_g", "illum_b",
    "rppg_pulse", "dup_hash_mean", "zncc_self",
]

DEVICE_CPU = "CPU (numpy/sklearn, this laptop: RTX 4070 Laptop 8GB / 22 cores / 15GB RAM)"


# --------------------------------------------------------------------- io ----
def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO, timeout=20).stdout.strip()
    except Exception:
        return "unknown"


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def load_gold(bundle: str | None = None):
    """The 150 human-labelled IndEgo windows + their cached 26 features.

    Returns dict with ids, X (n,26 float64), y (1=good), clip, reason, path,
    start_s, names.  Label source: ONE human labeller (documented).
    """
    b = bundle or BUNDLE
    man = json.load(open(os.path.join(b, "labels", "manifest_gold.json")))
    feats = json.load(open(os.path.join(b, "caches", "gpu", "features.json")))["windows"]
    ids, X, y, clip, reason, path, start = [], [], [], [], [], [], []
    for it in man["items"]:
        k = it["id"]
        if k not in feats or "full" not in feats[k]:
            continue
        ids.append(k)
        X.append([feats[k]["full"][n] for n in FEATURE_NAMES])
        y.append(int(it["label"]))
        clip.append(it["clip"])
        reason.append(it.get("reason"))
        path.append(it["path"])
        start.append(float(it["start_s"]))
    return {"ids": ids, "X": np.nan_to_num(np.asarray(X, float)), "y": np.asarray(y),
            "clip": np.asarray(clip), "reason": np.asarray(reason, dtype=object),
            "path": path, "start_s": np.asarray(start), "names": FEATURE_NAMES,
            "label_source": "human gold, ONE labeller (150 IndEgo windows: 100 good / 50 bad)"}


def load_ood160(bundle: str | None = None):
    """The 160 human-labelled Egocentric-10K windows (138 good / 22 bad, 47 factories).

    Labels come from the merged manifest (`ood160_labels.json`); the per-window
    frozen 26-feature scores come from the harden gate run
    (`ood160_heads_old26_vs_new22.json`), which scored the 480p window clips
    through the ORIGINAL portal.  `video` paths are 480p window clips.
    """
    b = bundle or BUNDLE
    lab = json.load(open(os.path.join(b, "labels", "ood160_labels.json")))
    win = lab["windows"]
    scores_p = os.environ.get("JADAL_OOD160_HEADS") or os.path.join(
        "/home/parshu/projects/awing/interview/dsh_pack/ood160_heads_old26_vs_new22.json")
    heads = json.load(open(scores_p))
    ids = [w["window"] for w in win]
    y = np.asarray([1 if w["verdict"] == "good" else 0 for w in win])
    factory = np.asarray([w["factory"] for w in win], dtype=object)
    reason = np.asarray([w.get("reason") or "" for w in win], dtype=object)
    seg = [w["segment_id"] for w in win]
    p26 = np.asarray([heads["old26"].get(i, {}).get("score") for i in ids], float)
    v26 = [heads["old26"].get(i, {}).get("verdict") for i in ids]
    p22 = np.asarray([heads["new22"].get(i, {}).get("score") for i in ids], float)
    return {"ids": ids, "segment_id": seg, "y": y, "factory": factory,
            "reason": reason, "p26": p26, "v26": v26, "p22": p22, "heads": heads,
            "n_good": int((y == 1).sum()), "n_bad": int((y == 0).sum()),
            "label_source": "human labels, ONE labeller (160 Egocentric-10K windows: "
                            "138 good / 22 bad; 19 of 22 bad are NO_ACTIVE_WORK)"}


# ---------------------------------------------------------------- metrics ----
def fast_auc(y, s) -> float:
    """Rank-based AUROC (ties averaged), no sklearn dependency."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _index_by(groups):
    gs = list(groups)
    uniq = sorted(set(gs))
    m = {g: i for i, g in enumerate(uniq)}
    gi = np.asarray([m[g] for g in gs])
    return uniq, [np.where(gi == k)[0] for k in range(len(uniq))]


def group_boot_ci(y, s, groups, n_boot=2000, seed=0):
    """Resample GROUPS with replacement; the honest CI when groups are correlated
    (clips in-domain, factories out-of-domain).  Skips draws with one class."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    uniq, idx_by = _index_by(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[k] for k in pick])
        d = y[idx]
        if d.min() == d.max():
            continue
        vals.append(fast_auc(d, s[idx]))
    if not vals:
        return (float("nan"), float("nan"), 0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi), len(vals))


def window_boot_ci(y, s, n_boot=2000, seed=0):
    """OPTIMISTIC: resamples windows, ignoring within-clip/within-factory
    correlation.  Kept for continuity with earlier reports; never used alone
    to call a difference significant."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        d = y[idx]
        if d.min() == d.max():
            continue
        vals.append(fast_auc(d, s[idx]))
    if not vals:
        return (float("nan"), float("nan"), 0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi), len(vals))


def paired_group_boot_diff(y, s_a, s_b, groups, n_boot=2000, seed=0):
    """Paired group-bootstrap of AUROC(a) - AUROC(b) on the SAME windows.
    Significant only if the interval excludes zero."""
    y = np.asarray(y)
    s_a = np.asarray(s_a, float)
    s_b = np.asarray(s_b, float)
    uniq, idx_by = _index_by(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[k] for k in pick])
        d = y[idx]
        if d.min() == d.max():
            continue
        vals.append(fast_auc(d, s_a[idx]) - fast_auc(d, s_b[idx]))
    if not vals:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    delta = fast_auc(y, s_a) - fast_auc(y, s_b)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"delta": float(delta), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": len(vals)}


# --------------------------------------------------------- decision rules ----
def loco_proba(X, y, clip, C=0.3, class_weight="balanced", groups=None):
    """Leave-one-GROUP-out out-of-fold probabilities (group = clip by default).

    StandardScaler + LogisticRegression fitted on the training groups only:
    every hyperparameter is fixed here, nothing is chosen on the held-out group.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    groups = np.asarray(clip if groups is None else groups)
    p = np.zeros(len(y), float)
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000, C=C,
                               class_weight=class_weight).fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def youden_threshold(y, p):
    """Threshold maximising balanced accuracy (Youden's J).  Choose on TRAIN only."""
    y = np.asarray(y)
    p = np.asarray(p, float)
    cand = np.unique(np.round(p, 6))
    best, bt = -1.0, 0.5
    for t in cand:
        ba = balanced_accuracy(y, (p >= t).astype(int))
        if ba > best:
            best, bt = ba, float(t)
    return bt, best


def balanced_accuracy(y, pred) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    tpr = float((pred[pos] == 1).mean())
    tnr = float((pred[neg] == 0).mean())
    return 0.5 * (tpr + tnr)


def rates_at(y, p, thr):
    """accept = p >= thr.  Returns the four rates the memo needs."""
    y = np.asarray(y)
    acc = np.asarray(p, float) >= thr
    good, bad = y == 1, y == 0
    return {
        "threshold": float(thr),
        "false_accept_rate": float(acc[bad].mean()) if bad.sum() else float("nan"),
        "false_reject_rate": float((~acc[good]).mean()) if good.sum() else float("nan"),
        "bal_acc": balanced_accuracy(y, acc.astype(int)),
        "coverage": 1.0,
        "n_good": int(good.sum()), "n_bad": int(bad.sum()),
    }


# ---------------------------------------------------------------- results ----
REQUIRED = ("n", "n_bad", "label_source", "protocol", "ci95", "ci95_type",
            "device", "cost", "command", "git_sha", "seed", "config_sha256")


def save_results(path: str, payload: dict) -> str:
    """Validate the schema, then write.  Missing keys are a hard error: the
    whole point is that no number reaches a report without its provenance."""
    missing = [k for k in REQUIRED if k not in payload]
    if missing:
        raise ValueError(f"results JSON missing required keys: {missing}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    return path
