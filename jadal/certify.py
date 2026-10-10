"""L2 certification: calibration, exact error bounds, threshold selection, batch rule.

Determinism: no randomness anywhere in this module. The same inputs always give
the same outputs on the same numpy/scipy versions.

NOT DONE:
- `fit_temperature` fits ONE scalar temperature for all rows/classes. A
  per-question-type temperature map (as Laya does) is not implemented.
- `pick_threshold` assumes confidences are comparable across segments and that
  the calibration set is labelled; it does not cluster or group by step type.
- No confidence interval for the batch rule (it is a fixed rule, see SPEC.md).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.stats import beta

DEFAULT_DELTA = 0.1
DEFAULT_MIN_ACCEPTED = 30
DEFAULT_BINS = 15


def exact_binomial_upper(errors: int, n: int, delta: float = DEFAULT_DELTA) -> float:
    """One-sided Clopper-Pearson upper bound on the error rate.

    Returns the smallest p such that observing ``errors`` failures out of ``n``
    trials is not significant at level ``delta`` (i.e. a ``1 - delta`` upper
    confidence bound on the true error rate).

    With zero errors the bound is ``1 - delta ** (1/n)`` (rule of three family);
    with all errors it is 1.0.
    """
    errors = int(errors)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if errors < 0 or errors > n:
        raise ValueError(f"errors must be in [0, n], got errors={errors}, n={n}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if errors == 0:
        return 1.0 - delta ** (1.0 / n)
    if errors == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))
