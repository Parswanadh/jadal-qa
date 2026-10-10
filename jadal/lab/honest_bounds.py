"""Jadal lab: honest finite-sample error reporting (direction 4 methodology).

Breakthrough-lab direction 4 finding: no deployment-wide error guarantee
is computable from n=150 windows / 7 clips / 2 wearers (conformal
calibration on 4 clips promised <=5% false-reject, measured 46% on 3
held-out clips; ICC=0.58 -> effective n ~= 11 of 100). Evidence:
eval/lab/conformal/REPORT.md.

This module is the adopted practice: every dashboard number MUST be
reported as observed rate + exact CI + effective-n caveat. Nothing here
tightens a bound; it stops us printing bounds we do not have.

Functions are pure/deterministic (no RNG except documented seed param
for the optional cluster bootstrap, which defaults to a fixed seed).
"""

from __future__ import annotations

import math

try:
    from scipy.stats import beta as _beta
except Exception:  # pragma: no cover - scipy is a pinned dependency
    _beta = None


def _cp(k: int, n: int, alpha: float = 0.05) -> tuple:
    if _beta is None:  # normal-approx fallback (flagged, never silent)
        p = k / n
        z = 1.96
        se = math.sqrt(p * (1 - p) / n)
        return (max(0.0, p - z * se), min(1.0, p + z * se), False)
    lo = 0.0 if k == 0 else _beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi), True)


def cp_interval(k: int, n: int, alpha: float = 0.05) -> dict:
    """Observed rate + exact CI + method flag."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= k <= n):
        raise ValueError("k must satisfy 0 <= k <= n")
    lo, hi, exact = _cp(k, n, alpha)
    return {"k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi,
            "alpha": alpha, "exact": exact}


def kish_deff(icc: float, mean_cluster_size: float) -> float:
    """Design effect DEFF = 1 + (m-1)*ICC for clustered windows."""
    return 1.0 + (mean_cluster_size - 1.0) * icc


def effective_n(n: int, deff: float) -> float:
    """Effective independent sample size n / DEFF."""
    return n / deff


def required_n_zero_failure(p: float, delta: float = 0.05, m: int = 1) -> int:
    """Clean samples needed for a zero-failure claim err<=p at 1-delta.

    n = ceil(ln(delta/m) / ln(1-p)) (Bonferroni over m hypotheses).
    E.g. p=0.05, delta=0.05, m=4 -> 86 (the 86-vs-84 arithmetic).
    Only applies to a zero-observed-failure regime on a FRESH sample.
    """
    return math.ceil(math.log(delta / m) / math.log(1 - p))


def required_n_rate(p: float, half_width: float, alpha_ci: float = 0.05) -> int:
    """Samples for +/-half_width normal-approx CI at observed rate p."""
    z = 1.96  # 95%
    _ = alpha_ci
    return math.ceil(p * (1 - p) * (z / half_width) ** 2)


def dashboard_line(name: str, k: int, n: int,
                   icc: float | None = None,
                   mean_cluster_size: float | None = None) -> str:
    """One honest dashboard line: observed rate + CI (+ clustering caveat)."""
    ci = cp_interval(k, n)
    line = (f"{name}: {ci['rate']*100:.1f}% "
            f"(95% CI {ci['lo']*100:.1f}-{ci['hi']*100:.1f}%, "
            f"{k}/{n})")
    if icc is not None and mean_cluster_size is not None:
        deff = kish_deff(icc, mean_cluster_size)
        line += (f" [clustered: ICC={icc:.3f}, DEFF={deff:.2f}, "
                 f"effective n~{effective_n(n, deff):.1f}; "
                 f"pooled CI understates uncertainty]")
    return line
