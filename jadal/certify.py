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
from collections.abc import Mapping
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta

DEFAULT_DELTA = 0.1
DEFAULT_MIN_ACCEPTED = 30
DEFAULT_BINS = 15
DEFAULT_TARGET_ERROR = 0.05  # matches Config.target_error in jadal/config.py
TEMPERATURE_MIN = 0.01
TEMPERATURE_MAX = 100.0
VERDICTS = frozenset({"accept", "reject", "review"})


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


def _prepare_conf_correct(conf, correct) -> tuple[np.ndarray, np.ndarray]:
    """Return (confidence, is_correct) arrays, validating the input."""
    c = np.asarray(conf, dtype=float)
    if c.ndim != 1:
        raise ValueError("conf must be a 1-D array of confidences")
    ok = np.asarray(correct)
    if ok.ndim != 1 or ok.shape[0] != c.shape[0]:
        raise ValueError("correct must be a 1-D array with one entry per confidence")
    ok = _as_bool_mask(ok)
    if not np.all(np.isfinite(c)):
        raise ValueError("conf must be finite")
    if np.any(c < 0.0) or np.any(c > 1.0):
        raise ValueError("conf must lie in [0, 1]")
    return c, ok


def _as_bool_mask(ok: np.ndarray) -> np.ndarray:
    """Interpret `correct` strictly as booleans or 0/1 numbers.

    Plain ``astype(bool)`` would silently turn the string ``"False"`` into
    ``True``; anything that is not a bool or a 0/1 int/float is rejected.
    """
    if ok.dtype == bool:
        return ok
    if ok.dtype.kind in "iu":
        if np.all((ok == 0) | (ok == 1)):
            return ok.astype(bool)
        raise ValueError("correct must contain only 0/1 integers or booleans")
    if ok.dtype.kind == "f":
        if np.all((ok == 0.0) | (ok == 1.0)):
            return ok.astype(bool)
        raise ValueError("correct must contain only 0/1 numbers or booleans")
    raise ValueError("correct must be booleans or 0/1 numbers, not strings/objects")


def pick_threshold(
    conf,
    correct,
    target_err: float = DEFAULT_TARGET_ERROR,
    delta: float = DEFAULT_DELTA,
    min_accepted: int = DEFAULT_MIN_ACCEPTED,
) -> dict:
    """Fixed-sequence threshold search over the confidence distribution.

    Candidates are the distinct confidence values, ordered strictest
    (highest) threshold first. That order is fixed BEFORE looking at any
    outcomes. Each candidate is tested once at level ``delta``: a segment is
    accepted when ``conf >= threshold``, and the candidate is certified only
    when it accepts at least ``min_accepted`` segments AND its exact binomial
    (Clopper-Pearson) upper bound on the error rate is at or below
    ``target_err``. The scan STOPS at the first candidate that fails either
    check and returns the last certified one.

    Why the stopping rule matters: because the order is pre-fixed and the
    decision to test candidate j depends only on candidates 1..j passing,
    this is fixed-sequence testing — the family-wise error over the whole
    scan is controlled at ``delta`` with no multiplicity correction, so the
    returned threshold keeps its ``1 - delta`` certificate. Scanning every
    candidate and then picking the loosest passer would be a
    multiple-comparisons selection and would NOT carry that certificate.

    Returns ``{"threshold", "accepted", "errors", "upper_bound", "coverage"}``.
    When even the strictest candidate fails, ``threshold`` is ``None`` and the
    other fields describe the loosest candidate (accept everything), so
    callers can see how far off the target was.
    """
    c, ok = _prepare_conf_correct(conf, correct)
    target = float(target_err)
    delta = float(delta)
    min_accepted = int(min_accepted)
    if not 0.0 < target < 1.0:
        raise ValueError("target_err must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if min_accepted < 1:
        raise ValueError("min_accepted must be at least 1")
    total = int(c.shape[0])
    if total == 0:
        return {
            "threshold": None,
            "accepted": 0,
            "errors": 0,
            "upper_bound": None,
            "coverage": 0.0,
        }
    best: dict | None = None
    loosest: dict | None = None
    for threshold in np.unique(c)[::-1]:  # strictest first; order fixed upfront
        mask = c >= threshold
        accepted = int(mask.sum())
        errors = int((mask & ~ok).sum())
        upper_bound = exact_binomial_upper(errors, accepted, delta)
        loosest = {
            "threshold": float(threshold),
            "accepted": accepted,
            "errors": errors,
            "upper_bound": upper_bound,
            "coverage": accepted / total,
        }
        if accepted < min_accepted or upper_bound > target:
            break  # fixed-sequence stopping rule: no later candidate is tested
        best = loosest
    if best is not None:
        return best
    # Nothing was certified. The accept-everything point below is a diagnostic
    # summary only: it was never tested, so it carries no certificate, and
    # computing it does not test (or certify) any candidate.
    errors = int((~ok).sum())
    return {
        "threshold": None,
        "accepted": total,
        "errors": errors,
        "upper_bound": exact_binomial_upper(errors, total, delta),
        "coverage": 1.0,
    }
def risk_coverage(conf, correct) -> list[dict]:
    """Risk (observed error) versus coverage for every distinct confidence.

    Rows are ordered from the strictest threshold downward, so coverage grows
    along the curve and the last row is "accept everything". All floats are
    rounded to 4 decimals (SPEC determinism rule); distinct confidences that
    round equal therefore produce duplicate threshold rows — that is kept
    deliberately so every row still maps 1:1 to a distinct raw confidence.
    """
    c, ok = _prepare_conf_correct(conf, correct)
    total = int(c.shape[0])
    rows: list[dict] = []
    for threshold in np.unique(c)[::-1]:  # strictest first, coverage ascending
        mask = c >= threshold
        accepted = int(mask.sum())
        errors = int((mask & ~ok).sum())
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "coverage": round(accepted / total, 4),
                "error": round(errors / accepted, 4),
            }
        )
    return rows


def _verdict_values(verdicts) -> list[str]:
    """Accept a Batch mapping, a list of Verdict mappings, or verdict strings."""
    if isinstance(verdicts, Mapping):
        items = verdicts.get("verdicts", [])
    else:
        items = verdicts
    values: list[str] = []
    for item in items:
        value = item.get("verdict") if isinstance(item, Mapping) else item
        if not isinstance(value, str):
            raise ValueError(f"cannot read a verdict from {item!r}")
        if value not in VERDICTS:
            raise ValueError(f"unknown verdict {value!r}")
        values.append(value)
    return values


def batch_rule(verdicts, k: int) -> str:
    """The FIXED batch rule: reject when the rejection count exceeds ``k``.

    Only ``reject`` verdicts count; ``review`` segments go to a supervisor and do
    not sink the batch. The rule carries no statistical guarantee at n = 5-10
    (see SPEC.md) and must never be described as one.
    """
    k = int(k)
    if k < 0:
        raise ValueError("k must be 0 or more")
    rejected = sum(1 for value in _verdict_values(verdicts) if value == "reject")
    return "reject" if rejected > k else "accept"
