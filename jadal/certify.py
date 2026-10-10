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

import math
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta

DEFAULT_DELTA = 0.1
DEFAULT_MIN_ACCEPTED = 30
DEFAULT_BINS = 15
TEMPERATURE_MIN = 0.01
TEMPERATURE_MAX = 100.0


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


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _prepare_logits(logits, labels) -> tuple[np.ndarray, np.ndarray]:
    """Return (n, classes) logits and integer labels, validating the input."""
    z = np.asarray(logits, dtype=float)
    if z.ndim == 1:
        # 1-D input means "logit of the positive class" of a binary problem
        z = np.stack([np.zeros_like(z), z], axis=1)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError("logits must be (n, classes) with at least 2 classes")
    if z.shape[0] == 0:
        raise ValueError("logits must contain at least one row")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite")
    y = np.asarray(labels)
    if y.ndim != 1 or y.shape[0] != z.shape[0]:
        raise ValueError("labels must be a 1-D array with one entry per logit row")
    if np.issubdtype(y.dtype, np.floating) and not np.all(y == np.floor(y)):
        raise ValueError("labels must be integers")
    y = y.astype(int)
    if np.any(y < 0) or np.any(y >= z.shape[1]):
        raise ValueError("labels must index the columns of logits")
    return z, y


def _prepare_probs(probs, labels) -> tuple[np.ndarray, np.ndarray]:
    """Return (n, classes) probabilities and integer labels, validating the input."""
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        # 1-D input means "probability of the positive class" of a binary problem
        if np.any(p < 0.0) or np.any(p > 1.0):
            raise ValueError("1-D probs must lie in [0, 1]")
        p = np.stack([1.0 - p, p], axis=1)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probs must be (n, classes) with at least 2 classes")
    if p.shape[0] == 0:
        raise ValueError("probs must contain at least one row")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("probs must be finite and non-negative")
    if np.any(p.sum(axis=1) <= 0.0):
        raise ValueError("every probs row must have a positive sum")
    y = np.asarray(labels)
    if y.ndim != 1 or y.shape[0] != p.shape[0]:
        raise ValueError("labels must be a 1-D array with one entry per probs row")
    if np.issubdtype(y.dtype, np.floating) and not np.all(y == np.floor(y)):
        raise ValueError("labels must be integers")
    y = y.astype(int)
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("labels must index the columns of probs")
    return p, y


def ece(probs, y, bins: int = DEFAULT_BINS) -> float:
    """Expected calibration error of top-1 predictions.

    ``probs`` is either an (n, classes) array of probabilities or a 1-D array of
    positive-class probabilities for a binary problem. Bins are equal-width over
    [0, 1]; empty bins contribute nothing. Returns a value in [0, 1].
    """
    bins = int(bins)
    if bins < 1:
        raise ValueError("bins must be a positive integer")
    p, labels = _prepare_probs(probs, y)
    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)
    correct = (predicted == labels).astype(float)
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    total = float(labels.shape[0])
    error = 0.0
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
        error += (count / total) * gap
    return float(error)


def fit_temperature(
    logits, y, low: float = TEMPERATURE_MIN, high: float = TEMPERATURE_MAX
) -> float:
    """Fit one scalar temperature T minimising the NLL of ``softmax(logits / T)``.

    The search runs over ``log T`` in ``[log low, log high]`` with a bounded,
    deterministic Brent minimiser (no sampling, no random restarts). A value
    above 1 cools over-confident logits; below 1 sharpens under-confident ones.
    """
    z, labels = _prepare_logits(logits, y)
    if not 0.0 < low < high:
        raise ValueError("require 0 < low < high")
    rows = np.arange(labels.shape[0])

    def objective(log_t: float) -> float:
        temperature = math.exp(log_t)
        log_probs = _softmax(z / temperature)
        chosen = np.clip(log_probs[rows, labels], 1e-12, None)
        return float(-np.log(chosen).mean())

    result = minimize_scalar(
        objective,
        bounds=(math.log(low), math.log(high)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:  # pragma: no cover - bounded Brent is very robust
        raise RuntimeError(f"temperature fit failed: {result.message}")
    return float(math.exp(result.x))
