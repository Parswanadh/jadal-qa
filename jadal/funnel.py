"""Deterministic funnel router: accept / reject / review per docs/SPEC.md.

Contract: ``route(signals, decisions, thresholds)`` is a PURE function
(no I/O, no randomness, no clock). Same inputs -> identical output.

Rules (SPEC "Funnel router"):
1. HARD REJECT only from exact, reproducible L0 rules: undecodable,
   too-short segment, normalised blur, too dark / overexposed, static or
   shaky motion, hands-visible fraction below minimum, near-duplicate.
   Every reject records the measured value and the threshold in
   ``evidence`` so a worker can dispute it and anyone can re-verify it.
   Static motion maps to ``NO_ACTIVE_WORK``; too-short/undecodable maps
   to ``UNDECODABLE`` (no new reason codes before baseline-v1).
2. L1 model outputs (``hands_ok.p_yes``, ``active_work.p_yes``, ``step``)
   may NEVER hard-reject; they can only demote a clip to ``review``
   (``LOW_CONFIDENCE`` / ``WRONG_STEP`` / ``NO_ACTIVE_WORK``).
3. AUTO-ACCEPT only when every L0 check passes AND the calibrated L1
   ``answer_conf`` clears ``thresholds['accept_conf']``.
4. Everything else is ``review`` (Awign QA queue), ordered by
   ``queue_priority`` (higher = more uncertain / more worth a human's
   time; ties break by segment order, see :func:`queue_rank_key`).

``answer_conf`` choice: the MINIMUM of the available L1 head
probabilities (``hands_ok.p_yes``, ``active_work.p_yes``,
``step.answer_conf``). Rationale: accept needs confidence from EVERY
head, so the weakest head governs; ``min`` is monotone and needs no
tuning. ``None`` when no L1 head is available (=> review, never accept).

``queue_priority`` choice: mean per-head uncertainty
``u(p) = 1 - |2p - 1|`` (0 = certain, 1 = coin-flip) over the available
L1 heads; missing L1 entirely scores 1.0 (most worth a human look).
Accepted/rejected clips score 0.0 (they never enter the queue). The
score is monotone in uncertainty: a clip closer to the decision
boundary always ranks higher.

Missing fields: a missing/unparseable L0 measurement can never prove a
clip is bad, so it demotes to ``review`` (``LOW_CONFIDENCE``) instead
of rejecting. A missing L1 decision (``None`` or ``{}``) likewise means
``review``, never a crash and never an accept. ``dup_of`` missing or
null simply means "not a duplicate".

NOT DONE (Phase 3+):
- Threshold defaults are PLACEHOLDERS until calibrated on human gold labels.
- :func:`calibrate_two_sided` is implemented but only as trustworthy as its
  input labels: synthetic/Gemini runs are clearly marked and must never be
  presented as human-grade numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from jadal import certify as certify_mod

REASON_ORDER = [
    "UNDECODABLE", "HANDS_NOT_VISIBLE", "BLUR", "TOO_DARK", "OVEREXPOSED",
    "SHAKY", "DUPLICATE", "WRONG_STEP", "NO_ACTIVE_WORK", "LOW_CONFIDENCE",
]
_REASON_RANK = {code: i for i, code in enumerate(REASON_ORDER)}

# Canonical L0 check order. Evidence entries and multi-failure reasons both
# follow this order, so output is stable regardless of input dict ordering.
CHECK_ORDER = [
    "decodable", "duration_s", "blur_var", "brightness_mean",
    "under_exposed_frac", "over_exposed_frac", "motion_mean",
    "hands_visible_frac", "dup_of",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLDS_PATH = _REPO_ROOT / "configs" / "funnel_thresholds.json"


# --------------------------------------------------------------------------
# thresholds I/O
# --------------------------------------------------------------------------
def canonical_json(obj: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, compact separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def thresholds_hash(thresholds: Mapping[str, Any]) -> str:
    """Stable sha256 of the thresholds, excluding the ``sha256`` key itself.

    The stored ``sha256`` field pins the other fields; hashing excludes it
    so the digest is a fixed point (verify: recompute and compare equal).
    """
    body = {k: v for k, v in dict(thresholds).items() if k != "sha256"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def load_thresholds(path: str | Path | None = None) -> dict:
    """Load funnel thresholds from JSON (default: configs/funnel_thresholds.json)."""
    src = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
    with open(src, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"thresholds file {src} must contain a JSON object")
    return data


# --------------------------------------------------------------------------
# helpers (pure)
# --------------------------------------------------------------------------
def _round4(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _as_float(value: Any) -> float | None:
    """Parse a numeric measurement; None when missing/unparseable/NaN."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _as_prob(value: Any) -> float | None:
    number = _as_float(value)
    if number is None or not 0.0 <= number <= 1.0:
        return None
    return number


def _evidence(check: str, value: Any, op: str, threshold: Any,
              code: str) -> dict:
    return {
        "check": check,
        "value": _round4(value),
        "op": op,
        "threshold": _round4(threshold),
        "code": code,
    }


def _head_probs(decisions: Mapping[str, Any] | None) -> list[float]:
    """L1 head probabilities in a fixed order (hands, active, step conf)."""
    decisions = decisions if isinstance(decisions, Mapping) else {}
    probs: list[float] = []
    hands = decisions.get("hands_ok")
    hands_p = _as_prob(hands.get("p_yes")) if isinstance(hands, Mapping) else None
    if hands_p is not None:
        probs.append(hands_p)
    active = decisions.get("active_work")
    active_p = _as_prob(active.get("p_yes")) if isinstance(active, Mapping) else None
    if active_p is not None:
        probs.append(active_p)
    step = decisions.get("step")
    step = step if isinstance(step, Mapping) else {}
    step_conf = _as_prob(step.get("answer_conf"))
    if step_conf is not None:
        probs.append(step_conf)
    return probs


def _uncertainty(prob: float) -> float:
    """u(p) = 1 - |2p - 1|; 0 = certain, 1 = coin-flip."""
    """u(p) = 1 - |2p - 1|; 0 = certain, 1 = coin-flip."""
    return 1.0 - abs(2.0 * prob - 1.0)


def queue_rank_key(segment_id: str, queue_priority: float | None) -> tuple:
    """Deterministic queue ordering: higher priority first, ties by segment_id.

    Use as ``sorted(items, key=lambda it: queue_rank_key(it['segment_id'],
    it['queue_priority']))``. ``None`` priority (should not happen from
    :func:`route`, which always returns a float) sorts last.
    """
    priority = float(queue_priority) if queue_priority is not None else -1.0
    return (-priority, str(segment_id))


# --------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------
def route(signals: Mapping[str, Any] | None,
          decisions: Mapping[str, Any] | None,
          thresholds: Mapping[str, Any]) -> dict:
    """Route one segment to accept | reject | review (pure, deterministic).

    Returns ``{"verdict", "reasons", "evidence", "answer_conf",
    "queue_priority"}``. ``reasons`` uses existing reason codes only, in
    :data:`REASON_ORDER`; ``evidence`` follows :data:`CHECK_ORDER` then
    L1 checks (``hands_ok``, ``active_work``, ``step``); floats are
    rounded to 4 decimals.
    """
    signals = dict(signals) if isinstance(signals, Mapping) else {}
    decisions = dict(decisions) if isinstance(decisions, Mapping) else {}
    thresh = dict(thresholds)

    hard: set[str] = set()
    soft: set[str] = set()
    evidence: list[dict] = []
    missing_l0 = False  # any required L0 measurement missing/unparseable

    def need_float(name: str) -> float | None:
        nonlocal missing_l0
        number = _as_float(signals.get(name))
        if number is None:
            missing_l0 = True
        return number

    # -- L0 hard checks (exact rules only) -------------------------------
    # decodable
    decodable = signals.get("decodable")
    if decodable is False or (isinstance(decodable, (int,))
                              and not isinstance(decodable, bool)
                              and int(decodable) == 0):
        hard.add("UNDECODABLE")
        evidence.append(_evidence("decodable", False, "==", False,
                                  "UNDECODABLE"))
    elif decodable is True or decodable == 1:
        evidence.append(_evidence("decodable", True, "==", True,
                                  "UNDECODABLE"))
    else:
        missing_l0 = True
        evidence.append(_evidence("decodable", "missing", "==", True,
                                  "UNDECODABLE"))

    # duration (optional signal; skipped silently when absent)
    min_duration = _as_float(thresh.get("min_duration_s"))
    duration = _as_float(signals.get("duration_s",
                                    signals.get("segment_duration_s")))
    if min_duration is not None and duration is not None:
        if duration < min_duration:
            hard.add("UNDECODABLE")
        evidence.append(_evidence("duration_s", duration, ">=",
                                  min_duration, "UNDECODABLE"))

    # blur (blur_var is brightness- and resolution-normalised upstream)
    blur_min = _as_float(thresh.get("blur_var_min"))
    blur = need_float("blur_var")
    if blur_min is not None and blur is not None:
        if blur < blur_min:
            hard.add("BLUR")
        evidence.append(_evidence("blur_var", blur, ">=", blur_min, "BLUR"))

    # brightness / exposure
    bright_min = _as_float(thresh.get("brightness_min"))
    bright = need_float("brightness_mean")
    if bright_min is not None and bright is not None:
        if bright < bright_min:
            hard.add("TOO_DARK")
        evidence.append(_evidence("brightness_mean", bright, ">=",
                                  bright_min, "TOO_DARK"))
    under_max = _as_float(thresh.get("under_exposed_frac_max"))
    under = _as_float(signals.get("under_exposed_frac"))
    if under_max is not None and under is not None:
        if under > under_max:
            hard.add("TOO_DARK")
        evidence.append(_evidence("under_exposed_frac", under, "<=",
                                  under_max, "TOO_DARK"))
    over_max = _as_float(thresh.get("over_exposed_frac_max"))
    over = need_float("over_exposed_frac")
    if over_max is not None and over is not None:
        if over > over_max:
            hard.add("OVEREXPOSED")
        evidence.append(_evidence("over_exposed_frac", over, "<=",
                                  over_max, "OVEREXPOSED"))

    # motion: too shaky (SHAKY) or static (NO_ACTIVE_WORK)
    motion_max = _as_float(thresh.get("motion_max"))
    motion_min = _as_float(thresh.get("motion_min"))
    motion = need_float("motion_mean")
    if motion is not None:
        if motion_max is not None:
            if motion > motion_max:
                hard.add("SHAKY")
            evidence.append(_evidence("motion_mean", motion, "<=",
                                      motion_max, "SHAKY"))
        if motion_min is not None:
            if motion < motion_min:
                hard.add("NO_ACTIVE_WORK")
            evidence.append(_evidence("motion_mean_static", motion, ">=",
                                      motion_min, "NO_ACTIVE_WORK"))

    # hands visible fraction
    hands_min = _as_float(thresh.get("hands_visible_frac_min"))
    hands_frac = need_float("hands_visible_frac")
    if hands_min is not None and hands_frac is not None:
        if hands_frac < hands_min:
            hard.add("HANDS_NOT_VISIBLE")
        evidence.append(_evidence("hands_visible_frac", hands_frac, ">=",
                                  hands_min, "HANDS_NOT_VISIBLE"))

    # near-duplicate
    dup_of = signals.get("dup_of")
    if dup_of:
        hard.add("DUPLICATE")
        evidence.append(_evidence("dup_of", str(dup_of), "!=", None,
                                  "DUPLICATE"))

    # -- L1 soft checks (demote to review only, never reject) -------------
    accept_conf = _as_float(thresh.get("accept_conf"))
    if accept_conf is None:
        accept_conf = 1.0  # uncalibrated: nothing auto-accepts

    hands = decisions.get("hands_ok")
    hands_p = _as_prob(hands.get("p_yes")) if isinstance(hands, Mapping) else None
    active = decisions.get("active_work")
    active_p = _as_prob(active.get("p_yes")) if isinstance(active, Mapping) else None
    step = decisions.get("step")
    step = step if isinstance(step, Mapping) else {}
    step_choice = step.get("choice")
    step_conf = _as_prob(step.get("answer_conf"))

    heads = _head_probs(decisions)
    if hands_p is not None:
        if hands_p < accept_conf:
            soft.add("LOW_CONFIDENCE")
        evidence.append(_evidence("hands_ok.p_yes", hands_p, ">=",
                                  accept_conf, "LOW_CONFIDENCE"))
    if active_p is not None:
        if active_p < accept_conf:
            soft.add("NO_ACTIVE_WORK")
        evidence.append(_evidence("active_work.p_yes", active_p, ">=",
                                  accept_conf, "NO_ACTIVE_WORK"))
    if step_choice:
        evidence.append(_evidence("step.choice", str(step_choice), "==",
                                  str(step_choice), "WRONG_STEP"))
    else:
        soft.add("WRONG_STEP")
        evidence.append(_evidence("step.choice", "missing", "==",
                                  "non-empty", "WRONG_STEP"))
    if step_conf is not None:
        if step_conf < accept_conf:
            soft.add("LOW_CONFIDENCE")
        evidence.append(_evidence("step.answer_conf", step_conf, ">=",
                                  accept_conf, "LOW_CONFIDENCE"))

    if missing_l0:
        # A missing L0 measurement proves nothing: review, never reject.
        soft.add("LOW_CONFIDENCE")
    if not heads:
        # No usable L1 signal at all: review, never accept.
        soft.add("LOW_CONFIDENCE")

    answer_conf: float | None = round(min(heads), 4) if heads else None

    # -- verdict -----------------------------------------------------------
    if hard:
        verdict = "reject"
        reasons = sorted(hard | soft,
                         key=lambda c: _REASON_RANK.get(c, len(REASON_ORDER)))
        queue_priority = 0.0
    elif soft or answer_conf is None or answer_conf < accept_conf:
        verdict = "review"
        reasons_set = set(soft)
        if answer_conf is not None and answer_conf < accept_conf \
                and not reasons_set:
            reasons_set.add("LOW_CONFIDENCE")
        if answer_conf is None and not reasons_set:
            reasons_set.add("LOW_CONFIDENCE")
        reasons = sorted(reasons_set,
                         key=lambda c: _REASON_RANK.get(c, len(REASON_ORDER)))
        queue_priority = round(
            sum(_uncertainty(p) for p in heads) / len(heads), 4
        ) if heads else 1.0
    else:
        verdict = "accept"
        reasons = []
        queue_priority = 0.0

    return {
        "verdict": verdict,
        "reasons": reasons,
        "evidence": evidence,
        "answer_conf": answer_conf,
        "queue_priority": queue_priority,
    }


# --------------------------------------------------------------------------
# Phase 2: two-sided calibration
# --------------------------------------------------------------------------
LABEL_SOURCES = ("human_gold", "synthetic", "gemini")

# Tunable hard-reject rules: threshold key, signal field, fire direction,
# reason code. "lt" fires when value < threshold (strictest = smallest t,
# fires the fewest, most-certainly-bad clips first); "gt" fires when
# value > threshold (strictest = largest t). Boolean rules (decodable,
# dup_of) have no tunable threshold and are always on.
REJECT_RULES = [
    {"key": "blur_var_min", "signal": "blur_var", "direction": "lt",
     "code": "BLUR"},
    {"key": "brightness_min", "signal": "brightness_mean", "direction": "lt",
     "code": "TOO_DARK"},
    {"key": "under_exposed_frac_max", "signal": "under_exposed_frac",
     "direction": "gt", "code": "TOO_DARK"},
    {"key": "over_exposed_frac_max", "signal": "over_exposed_frac",
     "direction": "gt", "code": "OVEREXPOSED"},
    {"key": "motion_max", "signal": "motion_mean", "direction": "gt",
     "code": "SHAKY"},
    {"key": "motion_min", "signal": "motion_mean", "direction": "lt",
     "code": "NO_ACTIVE_WORK"},
    {"key": "hands_visible_frac_min", "signal": "hands_visible_frac",
     "direction": "lt", "code": "HANDS_NOT_VISIBLE"},
    {"key": "min_duration_s", "signal": "duration_s", "direction": "lt",
     "code": "UNDECODABLE"},
]

DEFAULT_TARGETS = {
    "target_alpha": 0.05,   # defect rate among accepted
    "delta": 0.1,           # 1 - delta confidence for every bound
    "target_beta": 0.05,    # false-reject rate per reject rule
    "min_support": 10,      # a rule must fire on this many calib clips
    "min_accepted": 30,     # accept side must cover this many calib clips
    "accept_step": 0.05,    # accept-side candidate grid resolution
    "calib_frac": 0.5,      # stratified calibration share
    "seed": 1234,
}


def _validate_records(records: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {i} must be a mapping")
        label = record.get("label")
        if label not in ("good", "bad"):
            raise ValueError(f"record {i} needs label 'good'|'bad'")
        source = record.get("label_source")
        if source not in LABEL_SOURCES:
            raise ValueError(
                f"record {i} needs a known label_source {LABEL_SOURCES}; "
                f"refusing to mix unknown sources silently")
        signals = record.get("signals")
        if not isinstance(signals, Mapping):
            raise ValueError(f"record {i} needs a signals mapping")
        decisions = record.get("decisions")
        if decisions is not None and not isinstance(decisions, Mapping):
            raise ValueError(f"record {i} decisions must be a mapping or None")
        cleaned.append({
            "signals": dict(signals),
            "decisions": dict(decisions) if decisions is not None else None,
            "label": label,
            "reason_gold": record.get("reason_gold"),
            "label_source": source,
        })
    if not cleaned:
        raise ValueError("need at least one labelled record")
    return cleaned


def _record_key(record: dict) -> str:
    return hashlib.sha256(canonical_json({
        "signals": record["signals"], "decisions": record["decisions"],
        "label": record["label"], "reason_gold": record.get("reason_gold"),
        "label_source": record["label_source"],
    })).hexdigest()


def _split_calib_heldout(records: list[dict], calib_frac: float,
                         seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified (by label) calibration/held-out split, fully deterministic.

    Records are first sorted by content hash (input-order independent),
    then shuffled within each label stratum with ``random.Random(seed)``.
    """
    ordered = sorted(records, key=_record_key)
    rng = random.Random(int(seed))
    strata: dict[str, list[dict]] = {"good": [], "bad": []}
    for record in ordered:
        strata[record["label"]].append(record)
    calib: list[dict] = []
    heldout: list[dict] = []
    for stratum in strata.values():
        rng.shuffle(stratum)
        total = len(stratum)
        if total == 0:
            continue
        take = min(max(1, int(round(float(calib_frac) * total))), total - 1) \
            if total >= 2 else 1
        calib.extend(stratum[:take])
        heldout.extend(stratum[take:])
    return calib, heldout


def _rate_bound(fired_labels: list[str], delta: float,
                error_label: str = "good") -> dict:
    """Exact-binomial bound on the fraction of fired clips with an error label.

    Reject side: ``error_label='good'`` (a fired clip that is actually good
    is a false reject). Accept side: ``error_label='bad'`` (an accepted clip
    that is actually bad is a defect).
    """
    total = len(fired_labels)
    errors = sum(1 for label in fired_labels if label == error_label)
    return {
        "fired": total,
        "errors": errors,
        "upper_bound": round(certify_mod.exact_binomial_upper(
            errors, total, delta), 4) if total else None,
    }


def _bound_by_source(fired: list[dict], delta: float,
                     error_label: str = "good") -> dict:
    by_source: dict[str, dict] = {}
    for source in LABEL_SOURCES:
        subset = [r["label"] for r in fired if r["label_source"] == source]
        if subset:
            by_source[source] = _rate_bound(subset, delta, error_label)
    return by_source


def _measure(rule: dict, signals: Mapping[str, Any]) -> float | None:
    value = _as_float(signals.get(rule["signal"]))
    if value is None and rule["signal"] == "duration_s":
        value = _as_float(signals.get("segment_duration_s"))
    return value


def calibrate_two_sided(records: list[dict],
                        targets: dict | None = None) -> dict:
    """Two-sided calibration with exact-binomial certificates (deterministic).

    ``records``: ``{signals, decisions|None, label: 'good'|'bad',
    reason_gold: code|None, label_source}``. ``targets`` overrides any of
    :data:`DEFAULT_TARGETS`.

    (a) ACCEPT side: ``answer_conf`` (= min L1 head prob, same definition
    as :func:`route`) threshold via ``jadal.certify.pick_threshold``
    (fixed-sequence, exact binomial) on the CALIBRATION split so the
    DEFECT rate among accepted clips <= ``target_alpha``. Confs are
    floored to a fixed ``accept_step`` grid BEFORE the scan: this makes
    the candidate set a pre-declared grid (equivalent to scanning the raw
    confs at grid thresholds, since ``floor(c) >= t ⟺ c >= t`` on-grid),
    without which continuous confs would stop the scan at the first
    candidate. The reported bound is re-estimated on the DISJOINT
    held-out split, overall and per label source. Records with no usable
    L1 head can never accept and are counted as ``n_no_conf``.
    (b) REJECT side: per tunable rule, candidates ordered strictest
    (fires fewest, most-certainly-bad clips) to loosest; the SAME
    fixed-sequence logic bounds the FALSE-REJECT rate (fraction of fired
    clips that are actually good) <= ``target_beta``. Candidates firing
    on fewer than ``max(min_support, n_power)`` calib clips are excluded
    BEFORE testing, where ``n_power = ceil(log(delta)/log(1-beta))`` is
    the smallest support at which a clean rule CAN pass (the exact upper
    bound with 0 errors on fewer clips already exceeds beta). Both
    filters use measurements only — never labels — so the certificate
    stands. The scan stops at the first bound failure. A rule with no
    certifiable candidate is DISABLED (threshold None).

    Returns ``{"accept", "reject_rules", "threshold_updates", "meta"}``;
    all floats rounded to 4 decimals. Label sources are NEVER pooled
    silently: every bound is reported per source (``meta`` records the
    source mix and N per source).
    """
    cfg = dict(DEFAULT_TARGETS)
    cfg.update(targets or {})
    alpha = float(cfg["target_alpha"])
    delta = float(cfg["delta"])
    beta = float(cfg["target_beta"])
    min_support = int(cfg["min_support"])
    min_accepted = int(cfg["min_accepted"])
    accept_step = float(cfg["accept_step"])
    seed = int(cfg["seed"])
    if not 0.0 < alpha < 1.0 and alpha != 0.0:
        raise ValueError("target_alpha must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("target_beta must be in (0, 1)")
    if not 0.0 < accept_step <= 1.0:
        raise ValueError("accept_step must be in (0, 1]")
    # Smallest support at which a perfectly clean rule can pass the bound.
    n_power = math.ceil(math.log(delta) / math.log(1.0 - beta))
    support_floor = max(min_support, n_power)

    cleaned = _validate_records(records)
    calib, heldout = _split_calib_heldout(cleaned, float(cfg["calib_frac"]),
                                          seed)

    def conf_of(record: dict) -> float | None:
        heads = _head_probs(record["decisions"])
        return min(heads) if heads else None

    # -- (a) accept side --------------------------------------------------
    calib_conf = [(conf_of(r), r["label"] == "good", r) for r in calib]
    n_no_conf = sum(1 for conf, _, _ in calib_conf if conf is None)
    usable = [(conf, ok) for conf, ok, _ in calib_conf if conf is not None]
    quantised = [round(math.floor(c / accept_step) * accept_step, 10)
                 for c, _ in usable]
    picked = certify_mod.pick_threshold(
        conf=quantised, correct=[ok for _, ok in usable],
        target_err=alpha, delta=delta, min_accepted=min_accepted)
    accept_threshold = (round(float(picked["threshold"]), 4)
                        if picked["threshold"] is not None else None)
    accept_out: dict[str, Any] = {
        "threshold": accept_threshold,
        "n_calib": len(calib),
        "n_no_conf": n_no_conf,
        "calib_certified": {
            "accepted": picked["accepted"],
            "errors": picked["errors"],
            "upper_bound": (round(picked["upper_bound"], 4)
                            if picked["upper_bound"] is not None else None),
            "coverage": round(picked["coverage"], 4),
        },
    }
    if accept_threshold is not None:
        fired = [r for r in heldout
                 if (conf_of(r) is not None and conf_of(r) >= accept_threshold)]
        accept_out["heldout"] = _rate_bound(
            [r["label"] for r in fired], delta, error_label="bad")
        accept_out["heldout"]["n_heldout"] = len(heldout)
        accept_out["heldout_by_source"] = _bound_by_source(
            fired, delta, error_label="bad")
    else:
        accept_out["heldout"] = {"fired": 0, "errors": 0,
                                 "upper_bound": None,
                                 "n_heldout": len(heldout),
                                 "note": "no threshold certified: "
                                         "auto-accept stays disabled"}
        accept_out["heldout_by_source"] = {}

    # -- (b) reject side --------------------------------------------------
    reject_out: dict[str, dict] = {}
    updates: dict[str, Any] = {"accept_conf": accept_threshold}
    for rule in REJECT_RULES:
        key = rule["key"]
        measured = [(m, r) for r in calib
                    if (m := _measure(rule, r["signals"])) is not None]
        values = sorted({_round4(m) for m, _ in measured})
        if rule["direction"] == "lt":
            candidates = values  # strictest (fewest fires) first
        else:
            candidates = values[::-1]
        # Label-independent support pre-filter (keeps the certificate);
        # support_floor also guarantees the first tested candidate CAN pass.
        eligible = [t for t in candidates
                    if sum(1 for m, _ in measured
                           if (m < t if rule["direction"] == "lt"
                               else m > t)) >= support_floor]
        best: float | None = None
        best_stats: dict | None = None
        for threshold in eligible:
            fired = [r for m, r in measured
                     if (m < threshold if rule["direction"] == "lt"
                         else m > threshold)]
            stats = _rate_bound([r["label"] for r in fired], delta)
            if stats["upper_bound"] is not None \
                    and stats["upper_bound"] <= beta:
                best = threshold
                best_stats = stats
            else:
                break  # fixed-sequence stopping rule
        if best is None:
            entry: dict[str, Any] = {
                "threshold": None, "disabled": True,
                "code": rule["code"],
                "reason": ("no candidate certified: the strictest candidate "
                           f"with support >= {support_floor} already exceeds "
                           "target_beta (or no candidate reaches support)"),
                "n_eligible_candidates": len(eligible),
            }
        else:
            entry = {
                "threshold": best, "disabled": False,
                "code": rule["code"],
                "calib": best_stats,
                "n_eligible_candidates": len(eligible),
            }
        # Held-out diagnostic for the selected (or disabled) rule.
        held = [(m, r) for r in heldout
                if (m := _measure(rule, r["signals"])) is not None]
        if best is not None:
            fired_hold = [r for m, r in held
                          if (m < best if rule["direction"] == "lt"
                              else m > best)]
            entry["heldout"] = _rate_bound(
                [r["label"] for r in fired_hold], delta)
            entry["heldout_by_source"] = _bound_by_source(fired_hold, delta)
        else:
            entry["heldout"] = {"fired": 0, "errors": 0,
                                "upper_bound": None,
                                "note": "rule disabled: fires on nothing"}
            entry["heldout_by_source"] = {}
        reject_out[key] = entry
        updates[key] = best

    by_source: dict[str, int] = {}
    for record in cleaned:
        by_source[record["label_source"]] = \
            by_source.get(record["label_source"], 0) + 1
    meta = {
        "n_total": len(cleaned),
        "n_calib": len(calib),
        "n_heldout": len(heldout),
        "n_good": sum(1 for r in cleaned if r["label"] == "good"),
        "n_bad": sum(1 for r in cleaned if r["label"] == "bad"),
        "by_source": by_source,
        "targets": {k: cfg[k] for k in
                    ("target_alpha", "delta", "target_beta", "min_support",
                     "min_accepted", "accept_step", "calib_frac")},
        "support_floor": support_floor,
        "seed": seed,
        "date_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {"accept": accept_out, "reject_rules": reject_out,
            "threshold_updates": updates, "meta": meta}


__all__ = [
    "REASON_ORDER", "CHECK_ORDER", "DEFAULT_THRESHOLDS_PATH",
    "canonical_json", "thresholds_hash", "load_thresholds",
    "queue_rank_key", "route", "calibrate_two_sided",
]
