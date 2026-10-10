"""Jadal lab: permutation entropy of the frame-difference series.

Breakthrough-lab direction 6 finding: normalised Bandt-Pompe PE (m=4) of
the per-window MAD series was the best single information-theoretic cue
(LOCO AUROC 0.62, 95% CI 0.56-0.72); DUPLICATE windows (n=6) had the
lowest median PE — weakly consistent with "looped/periodic -> low PE",
underpowered. Evidence: eval/lab/infotheory/REPORT.md.

Use as a cheap loop/periodicity feature inside the stage-1 score, never
as a standalone gate. Deterministic (positional tie-break, no RNG);
streaming-partial (needs the window's MAD series); hashable (pure
function of the input series).
"""

from __future__ import annotations

import math


def _pattern_index(order: tuple, factorial: list) -> int:
    """Lehmer-code rank of a permutation (0 .. m!-1)."""
    m = len(order)
    used = [False] * m
    rank = 0
    for i, v in enumerate(order):
        smaller = sum(1 for u in range(v) if not used[u])
        rank += smaller * factorial[m - 1 - i]
        used[v] = True
    return rank


def permutation_entropy(series, m: int = 4, tau: int = 1) -> float:
    """Normalised Bandt-Pompe permutation entropy in [0, 1].

    Ties (equal values, e.g. frozen frames) are broken by position order,
    deterministically. Returns 0.0 for constant series (single pattern)
    and nan when the series is too short (< (m-1)*tau + 1 points).
    """
    x = [float(v) for v in series]
    n = len(x)
    if n < (m - 1) * tau + 1:
        return float("nan")
    factorial = [1] * (m + 1)
    for i in range(1, m + 1):
        factorial[i] = factorial[i - 1] * i
    counts = [0] * factorial[m]
    total = 0
    for i in range(n - (m - 1) * tau):
        window = [(x[i + j * tau], j) for j in range(m)]
        # sort by value, ties by position -> deterministic ordinal pattern
        window.sort(key=lambda t: (t[0], t[1]))
        order = [0] * m
        for rank, (_, pos) in enumerate(window):
            order[pos] = rank
        counts[_pattern_index(tuple(order), factorial)] += 1
        total += 1
    if total == 0:
        return float("nan")
    ent = 0.0
    for c in counts:
        if c:
            p = c / total
            ent -= p * math.log(p)
    return ent / math.log(factorial[m])


def pe_loop_score(series, m: int = 4, tau: int = 1) -> dict:
    """Return PE plus a loop-suspicion flag (descriptive, NOT a verdict).

    ``low_pe`` is True when PE < 0.5. The 0.5 cut is a descriptive
    starting point from n=150 (DUPLICATE median m3 0.93 vs GOOD 0.97 —
    overlapping); it MUST be recalibrated and must never gate alone.
    """
    pe = permutation_entropy(series, m=m, tau=tau)
    return {"pe": pe, "m": m, "tau": tau, "n": len(list(series)),
            "low_pe": (not math.isnan(pe)) and pe < 0.5}
