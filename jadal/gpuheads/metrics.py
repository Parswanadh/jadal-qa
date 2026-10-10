"""Metrics: AUROC / balanced accuracy / F1 + bootstrap CIs (deterministic RNG)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s))


def balanced_acc(y, s, thr=0.5) -> float:
    return float(balanced_accuracy_score(y, (s >= thr).astype(int)))


def f1(y, s, thr=0.5) -> float:
    return float(f1_score(y, (s >= thr).astype(int), zero_division=0))


def bootstrap_ci_auroc(y, s, n_boot=1000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(vals, (1 - level) / 2))
    hi = float(np.quantile(vals, 1 - (1 - level) / 2))
    return (lo, hi)
