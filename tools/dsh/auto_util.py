#!/usr/bin/env python3
"""EXTRA helpers for the AUTO stream (S4).  Extends tools/dsh/common.py, never forks it.

Everything here is built on `common`'s metrics (fast_auc, balanced_accuracy,
rates_at) and on the same group-resampling convention: CIs that touch factory
footage resample FACTORIES, in-domain CIs resample CLIPS.  The window bootstrap
is kept only as a labelled-optimistic contrast.

Added because `common` ships AUROC-only grouped intervals and the AUTO policies
are rate/policy statistics: a generic grouped bootstrap that can take ANY
statistic (including one that re-selects a set inside each resample, which is
what a coverage sweep needs).

CPU only, stdlib + numpy + scikit-learn.  No network.
"""
from __future__ import annotations

import numpy as np

import common as CM

#: label every CI this module produces
CI_TYPE_CLIP = "clip-level bootstrap (resample 7 IndEgo clips, 2000 draws, seed 0)"
CI_TYPE_FACTORY = "factory-grouped bootstrap (resample 47 factories, 2000 draws, seed 0)"
CI_TYPE_WINDOW_OPTIMISTIC = ("window bootstrap (2000 draws, seed 0) - OPTIMISTIC: "
                            "ignores within-clip/within-factory correlation")


# ---------------------------------------------------------------- grouping ----
def group_index(groups):
    """(unique groups sorted, list of index arrays) - same convention as common."""
    gs = list(groups)
    uniq = sorted(set(gs))
    m = {g: i for i, g in enumerate(uniq)}
    gi = np.asarray([m[g] for g in gs])
    return uniq, [np.where(gi == k)[0] for k in range(len(uniq))]


def group_boot_stat(y, groups, stat_fn, n_boot=2000, seed=0, min_classes=True):
    """Percentile CI of an arbitrary statistic under GROUP resampling.

    stat_fn(y_sub, idx_sub) -> float.  `idx_sub` is passed so a statistic can
    re-derive per-window arrays (scores, thresholds) for the resampled set --
    needed because a coverage sweep must re-select inside each resample.

    Draws where the resampled set has one class only are skipped (an AUROC or a
    balanced accuracy is undefined there); the number of usable draws is
    reported so a wide interval can be read as "few usable draws", not "narrow".
    """
    y = np.asarray(y)
    uniq, idx_by = group_index(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[k] for k in pick])
        d = y[idx]
        if min_classes and d.min() == d.max():
            continue
        v = stat_fn(d, idx)
        if v is None or not np.isfinite(v):
            continue
        vals.append(float(v))
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0, "point": float("nan")}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_boot": len(vals)}


def paired_group_boot_diff(y, groups, stat_a, stat_b, n_boot=2000, seed=0):
    """Paired group bootstrap of stat_a - stat_b on the SAME resampled groups.

    stat_x(y_sub, idx_sub) -> float.  A difference is "significant" only if the
    interval excludes zero (brief rule).  Both statistics see identical
    resamples, so the interval is the paired one, not the difference of two
    independent intervals.
    """
    y = np.asarray(y)
    uniq, idx_by = group_index(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[k] for k in pick])
        d = y[idx]
        if d.min() == d.max():
            continue
        a, b = stat_a(d, idx), stat_b(d, idx)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        vals.append(float(a) - float(b))
    if not vals:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "excludes_zero": False}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"delta": float(stat_a(y, np.arange(len(y))) - stat_b(y, np.arange(len(y)))),
            "lo": float(lo), "hi": float(hi), "n_boot": len(vals),
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ----------------------------------------------------------------- rates ----
def policy_rates(y, pred):
    """accept = pred==1.  The four numbers the AUTO memo needs."""
    y = np.asarray(y)
    pred = np.asarray(pred).astype(int)
    good, bad = y == 1, y == 0
    return {
        "n": int(len(y)), "n_good": int(good.sum()), "n_bad": int(bad.sum()),
        "false_accept_rate": float((pred[bad] == 1).mean()) if bad.sum() else float("nan"),
        "false_reject_rate": float((pred[good] == 0).mean()) if good.sum() else float("nan"),
        "bal_acc": float(CM.balanced_accuracy(y, pred)),
        "accept_share": float((pred == 1).mean()),
        "error_rate": float((pred != y).mean()),
        "coverage": 1.0,
    }


def rates_ci(y, groups, pred, n_boot=2000, seed=0):
    """Grouped CIs for bal_acc / FAR / FRR / accept_share of a FIXED prediction."""
    y = np.asarray(y)
    pred = np.asarray(pred).astype(int)
    out = {}
    for name, fn in (("bal_acc", lambda yy, pp: float(CM.balanced_accuracy(yy, pp))),
                     ("false_accept_rate", _far),
                     ("false_reject_rate", _frr),
                     ("accept_share", lambda yy, pp: float(np.mean(pp == 1))),
                     ("error_rate", lambda yy, pp: float(np.mean(pp != yy)))):
        # both classes are required in every resample: a draw without bad
        # windows has no false-accept rate, and reporting 0 there would be wrong
        r = group_boot_stat(y, groups, lambda d, idx, f=fn: f(d, pred[idx]),
                            n_boot=n_boot, seed=seed)
        out[name + "_ci95"] = [round(r["lo"], 4), round(r["hi"], 4)]
        out[name + "_n_boot_used"] = r["n_boot"]
    return out


def _far(y, pred):
    bad = y == 0
    return float((pred[bad] == 1).mean()) if bad.sum() else float("nan")


def _frr(y, pred):
    good = y == 1
    return float((pred[good] == 0).mean()) if good.sum() else float("nan")


# ------------------------------------------------------------ rank helpers ----
def rank01(x):
    """Average ranks scaled to [0,1]; ties share the mean rank.  Label-free."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, float)
    sr = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j + 2)  # 1-based average rank
        i = j + 1
    return (ranks - 1.0) / max(n - 1, 1)


def rank_average(*scores):
    """Mean of per-score rank01 -- the Y3-style combination, checked here."""
    return np.mean([rank01(s) for s in scores], axis=0)


# ------------------------------------------------- threshold selection ----
def cost_threshold(y, p, c_fa=1.0, c_fr=1.0):
    """Threshold minimising c_fa*FA + c_fr*FR on the given (training) labels.

    Empirical, not the analytic Bayes rule: the scores are not calibrated under
    class_weight='balanced', so the empirical minimiser is the honest choice.
    Ties break towards the threshold closest to 0.5.
    """
    y = np.asarray(y)
    p = np.asarray(p, float)
    cand = np.unique(np.round(p, 6))
    best, bt = np.inf, 0.5
    for t in cand:
        acc = p >= t
        fa = float(((acc == 1) & (y == 0)).sum())
        fr = float(((acc == 0) & (y == 1)).sum())
        c = c_fa * fa + c_fr * fr
        if c < best - 1e-12 or (abs(c - best) <= 1e-12 and abs(t - 0.5) < abs(bt - 0.5)):
            best, bt = c, float(t)
    return bt, float(best)


def loco_fold_thresholds(X, y, clip, kind="youden", C=0.3, c_fa=1.0, c_fr=1.0):
    """Per-held-out-clip thresholds chosen ONLY on the other clips.

    For held-out clip g: inner leave-one-clip-out probabilities on the 6
    training clips, then the rule on those inner-OOF training probabilities.
    kind='youden' -> balanced-accuracy maximiser; kind='cost' -> cost minimiser.
    Returns (threshold per window, per-fold dict).
    """
    y = np.asarray(y)
    clip = np.asarray(clip)
    thr = np.full(len(y), np.nan)
    folds = {}
    for g in sorted(set(clip)):
        tr, te = clip != g, clip == g
        p_in = CM.loco_proba(X[tr], y[tr], clip[tr], C=C)
        if kind == "youden":
            t, val = CM.youden_threshold(y[tr], p_in)
        elif kind == "cost":
            t, val = cost_threshold(y[tr], p_in, c_fa=c_fa, c_fr=c_fr)
        else:
            raise ValueError(kind)
        thr[te] = t
        folds[str(g)] = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                         "threshold": float(t), "train_objective": float(val)}
    assert not np.isnan(thr).any(), "every window must get its fold's threshold"
    return thr, folds


# --------------------------------------------------- conformal-style bound ----
def crc_lambda_select(cov_grid, risk_hat, alpha, n_cal):
    """Conformal-risk-control style selection on CALIBRATION data only.

    cov_grid / risk_hat: candidate confidence thresholds (lambda) ascending with
    the coverage they yield, and the empirical risk (error among auto-decided)
    on the calibration folds.  Bates et al. CRC: pick the largest coverage
    (smallest lambda) whose finite-sample-corrected risk is <= alpha,

        (n/(n+1)) * risk_hat(lambda) + 1/(n+1) <= alpha.

    Returns (index or None, corrected risk at that index).  No test data and no
    held-out label enters this function.
    """
    ok = None
    for i in range(len(cov_grid)):
        r = (n_cal / (n_cal + 1.0)) * risk_hat[i] + 1.0 / (n_cal + 1.0)
        if r <= alpha:
            ok = i
            break  # grid ascends by coverage: first feasible = largest coverage
    if ok is None:
        return None, float("nan")
    return ok, float((n_cal / (n_cal + 1.0)) * risk_hat[ok] + 1.0 / (n_cal + 1.0))


def wilson_upper(k, n, z=1.96):
    """Clopper-Pearson-free upper bound on a binomial rate (Wilson, 95%)."""
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(min(1.0, (c + h) / d))


# ----------------------------------------------------------- misc helpers ----
def decision_note_ood():
    """The sentence that must sit next to EVERY OOD-160 number."""
    return ("OOD-160: n=160, n_bad=22, and 19 of the 22 bad windows are "
            "NO_ACTIVE_WORK (2 HANDS_NOT_VISIBLE, 1 SHAKY); 47 factories; "
            "ONE human labeller. Intervals are factory-grouped; the window "
            "bootstrap is shown only marked optimistic.")


def summarise_ci(d):
    """['lo','hi'] -> '<point> [lo, hi]' from a rates dict + its CI dict."""
    return d
