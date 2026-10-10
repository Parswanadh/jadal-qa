"""Core deterministic helpers for the Y2 review-gate analysis.

Everything here is pure numpy / stdlib and deterministic (fixed seeds, stable
sort). No decoding, no model forward, no network: it consumes the frozen
caches listed in the experiment README.

Decision-theoretic gating
-------------------------
A window with positive-class probability ``p`` is decided ACCEPT when
``p >= p*`` and REJECT otherwise, where ``p* = C_fa / (C_fa + C_fr)`` is the
cost-optimal threshold for a false-accept cost ``C_fa`` and a false-reject
cost ``C_fr``.  The review region is where the expected losses of the two
decisions are closest, i.e. where ``|p - p*|`` is smallest: we decide the
``coverage`` fraction of windows with the largest margin and REVIEW the rest.
Coverage is therefore a direct budget knob on the review share
(``review = 1 - coverage``).

Stage-0 quality rules
---------------------
Deterministic rules over the cached per-window *descriptor* features (the same
feature names the frozen re-score wrote into ``windows_rescored.jsonl``).
Thresholds are fixed a priori (see ``QUALITY_RULES``) and validated on the
synthetic degradation set and on the 150 human-labelled IndEgo windows; they
are NEVER fit on the 60 human OOD windows.
"""
from __future__ import annotations

import numpy as np

SEED = 7
_EPS = 1e-6


# --------------------------------------------------------------------------
# basic metrics
# --------------------------------------------------------------------------
def auroc(p, y) -> float:
    """AUROC with proper average ranks for ties (Mann-Whitney U)."""
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    sr = np.arange(1, len(p) + 1, dtype=float)
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            sr[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(p))
    ranks[order] = sr
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list:
    """Wilson score interval for a binomial share."""
    if n == 0:
        return [float("nan"), float("nan")]
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    half = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((c - half) / d, 4), round((c + half) / d, 4)]


def cls_metrics(y, pred) -> dict:
    """Confusion + TPR/TNR/BA + cost-weighted error. pred/accept == 1."""
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    ba = 0.5 * (tpr + tnr)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "tpr": round(tpr, 4), "tnr": round(tnr, 4),
            "balanced_accuracy": round(ba, 4),
            "acc_n": tp + fp, "rej_n": tn + fn}


# --------------------------------------------------------------------------
# calibration (temperature scaling, monotone)
# --------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nll(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


TEMP_GRID = np.linspace(0.2, 5.0, 481)


def fit_temperature(p, y, grid=TEMP_GRID) -> float:
    """Single-parameter temperature minimising NLL (deterministic grid)."""
    nlls = [_nll(_sigmoid(_logit(p) / t), y) for t in grid]
    return float(grid[int(np.argmin(nlls))])


def apply_temperature(p, T) -> np.ndarray:
    return _sigmoid(_logit(p) / float(T))


def loco_temperature(p, y, groups, grid=TEMP_GRID) -> np.ndarray:
    """Honest leave-one-clip-out temperature: T for group g is fit on all
    OTHER groups only, then applied to g. Returns calibrated probs."""
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    out = np.empty(len(p))
    for g in sorted(set(groups.tolist())):
        m = groups == g
        T = fit_temperature(p[~m], y[~m], grid)
        out[m] = apply_temperature(p[m], T)
    return out


# --------------------------------------------------------------------------
# decision-theoretic gating
# --------------------------------------------------------------------------
def gate(p, coverage: float, cost_ratio: float, decide="cost"):
    """Assign ACCEPT/REJECT/REVIEW.

    Parameters
    ----------
    p : positive-class (good/usable) probability.
    coverage : fraction decided (0-1); review = 1 - coverage.
    cost_ratio : C_fa : C_fr expressed as C_fa / C_fr.
    decide : 'cost' ranks by the cost-aware margin |p - p*|;
             'uncertainty' ranks by |p - 0.5| (review = most uncertain).

    Returns (decided_idx, review_idx, pred) where pred is 1=accept / 0=reject
    for ALL windows (only valid where decided).
    """
    p = np.asarray(p, float)
    pstar = cost_ratio / (cost_ratio + 1.0)
    marg = np.abs(p - pstar) if decide == "cost" else np.abs(p - 0.5)
    order = np.argsort(-marg, kind="mergesort")  # stable -> deterministic
    k = int(round(float(coverage) * len(p)))
    decided = order[:k]
    review = order[k:]
    pred = (p >= pstar).astype(int)
    return decided, review, pred


def evaluate_gate(p, y, coverage, cost_ratio, decide="cost"):
    """Metrics over the DECIDED subset for one config."""
    decided, review, pred = gate(p, coverage, cost_ratio, decide)
    met = cls_metrics(np.asarray(y)[decided], pred[decided])
    met.update({"coverage": round(float(coverage), 4),
                "review": round(1 - float(coverage), 4),
                "cost_ratio_fa_fr": cost_ratio,
                "p_star": round(cost_ratio / (cost_ratio + 1.0), 4),
                "n_decided": int(len(decided)),
                "weighted_error": round(
                    (cost_ratio * met["FP"] + 1.0 * met["FN"]) / len(decided), 4)
                if len(decided) else float("nan")})
    return met, decided, review, pred


# --------------------------------------------------------------------------
# cached stage-0 quality rules (descriptor feature space)
# --------------------------------------------------------------------------
# Thresholds fixed a priori from the synthetic degradation set + the frozen
# L0 rule concepts (dark / over-exposed / frozen / static / undecodable).
# They are calibration-independent and use only cached per-window features.
QUALITY_RULES = [
    # (code, feature, op, threshold, direction-of-fail)
    ("TOO_DARK", "luma_mean", "<", 0.06),
    ("TOO_DARK", "dark_frac", ">", 0.85),
    ("OVEREXPOSED", "luma_mean", ">", 0.80),
    ("OVEREXPOSED", "bright_frac", ">", 0.25),
    ("FROZEN", "frozen_frac", ">", 0.50),
    ("FROZEN", "zncc_self", ">", 0.90),
    ("NO_ACTIVE_WORK", "motion_energy", "<", 1e-4),
]


def quality_fires(features: dict) -> list:
    """Deterministic stage-0 evidence: list of reason codes that fire."""
    out = []
    for code, feat, op, thr in QUALITY_RULES:
        v = features.get(feat)
        if v is None:
            continue
        if op == "<" and v < thr:
            out.append(code)
        elif op == ">" and v > thr:
            out.append(code)
    return out


def quality_fires_undecodable(failed: bool) -> list:
    """Integrity rule: an undecodable window is a hard reject."""
    return ["UNDECODABLE"] if failed else []


# --------------------------------------------------------------------------
# clip bootstrap
# --------------------------------------------------------------------------
def clip_bootstrap_indices(groups, rng):
    """Resample the SET of groups with replacement, concatenate their rows."""
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    pick = rng.integers(0, len(uniq), size=len(uniq))
    idx = np.concatenate([np.where(groups == uniq[i])[0] for i in pick])
    return idx
