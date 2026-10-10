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
    WHICH rules may reject is gated by ``thresholds['hard_reject_policy']``
    (round 2, audit L9): ``'integrity_only'`` (the DEFAULT) lets only
    undecodable / too-short / explicit-corruption rules reject — every
    other L0 quality failure goes to ``review`` with its evidence;
    ``'evidence_reject'`` additionally enables the calibrated L0 quality
    rejects. Rationale: a false reject costs a worker pay and a re-shoot
    inside the window, so quality rejects need a certified bound while
    integrity rejects (the clip proves nothing) are always safe.
2. L1 model outputs (``hands_ok.p_yes``, ``active_work.p_yes``, ``step``)
    may NEVER hard-reject; they can only demote a clip to ``review``
    (``LOW_CONFIDENCE`` / ``WRONG_STEP`` / ``NO_ACTIVE_WORK``). NOTE (M4):
    a ``reject`` verdict's ``reasons`` still list any L1 soft codes present
    alongside the hard cause (``sorted(hard | soft)``) — they describe
    additional uncertainty on the same clip, not an additional cause.
3. AUTO-ACCEPT only when every L0 check passes AND the calibrated L1
    ``answer_conf`` clears ``thresholds['accept_conf']`` AND the
    calibration gate opens (round 2, audit N1): ``status == 'calibrated'``
    accepts by default; ``'provisional'`` (synthetic/Gemini labels) routes
    to ``review`` unless the caller passes ``allow_provisional_accept=True``
    (every such accept carries
    ``{check: 'accept_status', value: 'provisional', label_source: ...}``
    evidence); ``'placeholder'``/missing routes to ``review`` unless the
    caller passes ``allow_uncalibrated_accept=True`` (demo pattern).
4. The hands rule is TASK-SCOPED (round 2, audit L8: the 'no hands' rule
    fires on 55/113 REAL segments — 0/46 assembly but 55/67
    logistics/desk work — because walking with a pallet IS the task).
    ``thresholds['profiles']`` maps profile name -> ``{requires_hands}``,
    e.g. ``{'assembly': {requires_hands: True},
    'logistics': {requires_hands: False}, 'default': {requires_hands:
    False}}``; ``route(..., profile='assembly')`` hard-rejects a
    hands violation, any other profile demotes it to ``review`` with
    evidence (never rejects). Unknown profile names fall back to
    ``'default'``. A config with NO ``profiles`` map keeps the legacy
    behaviour (hands rule enforced everywhere).
5. DUPLICATE never rejects on a degenerate 64-bit phash (all-zero hex, or
    popcount <= 4 / >= 60: a flat/blank frame carries no identity) and
    never in ``integrity_only`` mode; duplicates go to ``review`` with
    evidence (audit L5: embedding similarity only routes to review).
6. Everything else is ``review`` (Awign QA queue), ordered by
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

Missing fields: a missing, non-numeric, NaN/inf or garbage REQUIRED L0
measurement (``decodable`` plus every measurement whose rule threshold is
configured) can never prove anything, so it yields ``review`` with
``LOW_CONFIDENCE`` and a ``missing:<field>`` evidence entry — never
reject, never accept. A missing L1 decision (``None`` or ``{}``) likewise
means ``review``, never a crash and never an accept. ``dup_of`` missing or
null simply means "not a duplicate".

NOT DONE (Phase 3+):
- Threshold defaults are PLACEHOLDERS until calibrated on human gold labels.
- :func:`calibrate_two_sided` is implemented but only as trustworthy as its
  input labels: synthetic/Gemini runs are clearly marked and must never be
  presented as human-grade numbers.
- I5: ``min_accepted`` can leave ``accept_conf`` null on low-prevalence
  data: when SELECT holds fewer than ``min_accepted`` high-confidence
  clips (e.g. good segments are rare), no accept candidate is selected
  and auto-accept stays DISABLED (threshold None + note) — absence of a
  threshold is the honest output, not a failure to look.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    """Load funnel thresholds, verifying the pinned ``sha256`` (I1).

    The ``sha256`` field must equal :func:`thresholds_hash` over the rest
    of the file; a mismatch (tampering, hand-edit without rehashing, stale
    placeholder) raises ``ValueError`` instead of silently loading.
    """
    src = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
    with open(src, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"thresholds file {src} must contain a JSON object")
    stored = data.get("sha256")
    computed = thresholds_hash(data)
    if stored != computed:
        raise ValueError(
            f"thresholds file {src} fails hash verification: "
            f"stored sha256={stored!r} != computed {computed!r}. "
            f"Recompute with jadal.funnel.thresholds_hash after editing.")
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
# policy / profiles / phash helpers (pure, round 2)
# --------------------------------------------------------------------------
HARD_REJECT_POLICIES = ("integrity_only", "evidence_reject")

# L0 quality rules: may hard-reject ONLY under hard_reject_policy ==
# 'evidence_reject' (with certified thresholds). Under the default
# 'integrity_only' they demote to review with their evidence. Integrity
# rules (decodable/duration -> UNDECODABLE) always reject: the clip
# proves nothing, so no worker pay is at stake.
QUALITY_RULE_CODES = frozenset({
    "BLUR", "TOO_DARK", "OVEREXPOSED", "SHAKY", "NO_ACTIVE_WORK",
    "HANDS_NOT_VISIBLE", "DUPLICATE",
})


def _phash_degenerate(phash: Any) -> bool | None:
    """Is this 64-bit phash hex degenerate? None when it isn't one.

    All-zero, or popcount <= 4 / >= 60 (a flat, blank or saturated frame
    carries no identity), means a 'duplicate' match is unreliable: route
    to review, never reject. Returns None for non-16-hex-char values
    (no phash to judge — the caller's dup_of stands on its own).
    """
    if not isinstance(phash, str):
        return None
    text = phash.strip().lower()
    if len(text) != 16 or any(c not in "0123456789abcdef" for c in text):
        return None
    value = int(text, 16)
    if value == 0:
        return True
    popcount = bin(value).count("1")
    return popcount <= 4 or popcount >= 60


def _requires_hands(thresholds: Mapping[str, Any], profile: str) -> bool:
    """Does this task profile require visible hands? (audit L8).

    No ``profiles`` map in the config => legacy behaviour (enforced).
    Unknown profile name => falls back to ``'default'``; a ``'default'``
    entry missing ``requires_hands`` => False (hands demote to review).
    """
    profiles = thresholds.get("profiles")
    if profiles is None:
        return True
    if not isinstance(profiles, Mapping):
        raise ValueError("thresholds['profiles'] must be a mapping")
    entry = profiles.get(profile, profiles.get("default", {}))
    if not isinstance(entry, Mapping):
        raise ValueError(f"profiles[{profile!r}] must be a mapping")
    return bool(entry.get("requires_hands", False))


# --------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------
def route(signals: Mapping[str, Any] | None,
          decisions: Mapping[str, Any] | None,
          thresholds: Mapping[str, Any],
          allow_uncalibrated_accept: bool = False,
          allow_provisional_accept: bool = False,
          profile: str = "default") -> dict:
    """Route one segment to accept | reject | review (pure, deterministic).

    Returns ``{"verdict", "reasons", "evidence", "answer_conf",
    "queue_priority"}``. ``reasons`` uses existing reason codes only: on
    ``reject`` they are ``sorted(hard | soft)`` in :data:`REASON_ORDER`
    (M4: L1 soft codes ride along as extra uncertainty context, never as
    the cause); ``evidence`` follows :data:`CHECK_ORDER` then L1 checks
    (``hands_ok``, ``active_work``, ``step``); floats are rounded to 4
    decimals.

    Safety (C1): every REQUIRED L0 field — ``decodable`` plus each
    measurement whose rule threshold is configured — goes through the
    strict ``need_float`` path. A missing, non-numeric, NaN/inf or
    garbage value can never prove anything, so it yields ``review`` with
    ``LOW_CONFIDENCE`` and an evidence entry
    ``{check: 'missing:<field>', ...}`` — never reject, never accept.
    Only ``decodable`` explicitly false (or 0) rejects (``UNDECODABLE``).
    ``dup_of`` is not required: missing/null simply means "not a duplicate".

    Policy gate (round 2, audit L9): ``thresholds['hard_reject_policy']``
    is ``'integrity_only'`` (DEFAULT when absent) or ``'evidence_reject'``
    (anything else raises ``ValueError``). Under ``integrity_only`` only
    UNDECODABLE integrity rules reject; every other L0 quality failure
    (blur/dark/motion/hands/duplicate) demotes to ``review`` with its
    evidence.

    Hands scope (round 2, audit L8): the hands rule hard-rejects only
    when ``profile`` requires hands (see :func:`_requires_hands`);
    otherwise a low hands fraction demotes to ``review`` with evidence
    (never rejects) — walking with a pallet IS the task.

    Duplicate guard (round 2, audit L5): never reject on a degenerate
    phash (see :func:`_phash_degenerate`) and never on duplicate in
    ``integrity_only`` mode; duplicates go to ``review`` with evidence.

    Calibration gate (round 2, audit N1): auto-accept additionally
    requires ``thresholds['calibration']['status'] == 'calibrated'``.
    ``'provisional'`` routes to ``review`` unless the caller passes
    ``allow_provisional_accept=True`` (every such accept carries
    ``{check: 'accept_status', value: 'provisional', label_source: ...}``
    evidence); ``'placeholder'``/missing routes to ``review`` unless the
    caller passes ``allow_uncalibrated_accept=True`` (the demo sets it;
    the evidence then carries the flag).
    """
    signals = dict(signals) if isinstance(signals, Mapping) else {}
    decisions = dict(decisions) if isinstance(decisions, Mapping) else {}
    thresh = dict(thresholds)

    policy = thresh.get("hard_reject_policy", "integrity_only")
    if policy not in HARD_REJECT_POLICIES:
        raise ValueError(
            f"thresholds['hard_reject_policy'] must be one of "
            f"{HARD_REJECT_POLICIES}, got {policy!r}")
    quality_rejects = (policy == "evidence_reject")
    hands_required = _requires_hands(thresh, profile)

    hard: set[str] = set()
    soft: set[str] = set()
    evidence: list[dict] = []
    missing_l0 = False  # any required L0 measurement missing/unparseable

    def _file(code: str) -> None:
        """File a quality-rule failure: reject iff policy enables it."""
        if quality_rejects:
            hard.add(code)
        else:
            soft.add(code)

    def need_float(name: str) -> float | None:
        """Strict required-field read: records missing-evidence on failure."""
        nonlocal missing_l0
        raw = signals.get(name, signals.get("segment_duration_s")
                          if name == "duration_s" else None)
        number = _as_float(raw)
        if number is None:
            missing_l0 = True
            soft.add("LOW_CONFIDENCE")
            evidence.append({
                "check": f"missing:{name}",
                "value": raw if raw is not None else "missing",
                "op": "==",
                "threshold": "valid-number",
                "code": "LOW_CONFIDENCE",
            })
        return number

    # -- L0 hard checks (exact rules only) -------------------------------
    # decodable
    decodable = signals.get("decodable")
    if decodable is False or (isinstance(decodable, int)
                              and not isinstance(decodable, bool)
                              and int(decodable) == 0):
        hard.add("UNDECODABLE")
        evidence.append(_evidence("decodable", False, "==", False,
                                  "UNDECODABLE"))
    elif decodable is True or decodable == 1:
        evidence.append(_evidence("decodable", True, "==", True,
                                  "UNDECODABLE"))
    else:
        need_float("decodable")

    # duration (required whenever min_duration_s is configured)
    min_duration = _as_float(thresh.get("min_duration_s"))
    if min_duration is not None:
        duration = need_float("duration_s")
        if duration is not None:
            if duration < min_duration:
                hard.add("UNDECODABLE")
            evidence.append(_evidence("duration_s", duration, ">=",
                                      min_duration, "UNDECODABLE"))

    # blur (blur_var is brightness- and resolution-normalised upstream)
    blur_min = _as_float(thresh.get("blur_var_min"))
    blur = need_float("blur_var") if blur_min is not None else None
    if blur_min is not None and blur is not None:
        if blur < blur_min:
            _file("BLUR")
        evidence.append(_evidence("blur_var", blur, ">=", blur_min, "BLUR"))

    # brightness / exposure
    bright_min = _as_float(thresh.get("brightness_min"))
    bright = need_float("brightness_mean") if bright_min is not None else None
    if bright_min is not None and bright is not None:
        if bright < bright_min:
            _file("TOO_DARK")
        evidence.append(_evidence("brightness_mean", bright, ">=",
                                  bright_min, "TOO_DARK"))
    under_max = _as_float(thresh.get("under_exposed_frac_max"))
    under = (need_float("under_exposed_frac")
             if under_max is not None else None)
    if under_max is not None and under is not None:
        if under > under_max:
            _file("TOO_DARK")
        evidence.append(_evidence("under_exposed_frac", under, "<=",
                                  under_max, "TOO_DARK"))
    over_max = _as_float(thresh.get("over_exposed_frac_max"))
    over = need_float("over_exposed_frac") if over_max is not None else None
    if over_max is not None and over is not None:
        if over > over_max:
            _file("OVEREXPOSED")
        evidence.append(_evidence("over_exposed_frac", over, "<=",
                                  over_max, "OVEREXPOSED"))

    # motion: too shaky (SHAKY) or static (NO_ACTIVE_WORK)
    motion_max = _as_float(thresh.get("motion_max"))
    motion_min = _as_float(thresh.get("motion_min"))
    motion = (need_float("motion_mean")
              if motion_max is not None or motion_min is not None else None)
    if motion is not None:
        if motion_max is not None:
            if motion > motion_max:
                _file("SHAKY")
            evidence.append(_evidence("motion_mean", motion, "<=",
                                      motion_max, "SHAKY"))
        if motion_min is not None:
            if motion < motion_min:
                _file("NO_ACTIVE_WORK")
            evidence.append(_evidence("motion_mean_static", motion, ">=",
                                      motion_min, "NO_ACTIVE_WORK"))

    # hands visible fraction: task-scoped (audit L8). Where the profile
    # requires hands this is a quality reject like the rest; elsewhere a
    # low fraction only demotes to review (never rejects).
    hands_min = _as_float(thresh.get("hands_visible_frac_min"))
    hands_frac = (need_float("hands_visible_frac")
                  if hands_min is not None else None)
    if hands_min is not None and hands_frac is not None:
        if hands_frac < hands_min:
            if hands_required:
                _file("HANDS_NOT_VISIBLE")
            else:
                soft.add("HANDS_NOT_VISIBLE")
        evidence.append(_evidence("hands_visible_frac", hands_frac, ">=",
                                  hands_min, "HANDS_NOT_VISIBLE"))

    # near-duplicate: never on a degenerate phash, never in
    # integrity_only mode (audit L5) — review with evidence instead.
    dup_of = signals.get("dup_of")
    if dup_of:
        degenerate = _phash_degenerate(signals.get("phash"))
        if degenerate:
            soft.add("DUPLICATE")
            evidence.append({
                "check": "dup_phash_degenerate",
                "value": str(signals.get("phash")),
                "op": "==",
                "threshold": "non-degenerate-phash",
                "code": "DUPLICATE",
            })
            evidence.append(_evidence("dup_of", str(dup_of), "!=", None,
                                      "DUPLICATE"))
        elif not quality_rejects:
            soft.add("DUPLICATE")
            evidence.append(_evidence("dup_of", str(dup_of), "!=", None,
                                      "DUPLICATE"))
        else:
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
        # (need_float already added LOW_CONFIDENCE + missing evidence.)
        soft.add("LOW_CONFIDENCE")
    if not heads:
        # No usable L1 signal at all: review, never accept.
        soft.add("LOW_CONFIDENCE")

    answer_conf: float | None = round(min(heads), 4) if heads else None

    # Calibration gate (round 2, audit N1): auto-accept ONLY on
    # 'calibrated'. 'provisional' (synthetic/Gemini labels) needs the
    # explicit allow_provisional_accept opt-in; 'placeholder'/missing
    # needs the allow_uncalibrated_accept demo flag.
    calib_block = thresh.get("calibration")
    calib_status = calib_block.get("status") if isinstance(
        calib_block, Mapping) else None
    calib_source = calib_block.get("label_source") if isinstance(
        calib_block, Mapping) else None
    if calib_status == "calibrated":
        accept_open, provisional_accept = True, False
    elif calib_status == "provisional":
        accept_open, provisional_accept = allow_provisional_accept, True
    else:
        accept_open, provisional_accept = allow_uncalibrated_accept, False

    def _review_priority() -> float:
        return round(
            sum(_uncertainty(p) for p in heads) / len(heads), 4
        ) if heads else 1.0

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
        queue_priority = _review_priority()
    elif not accept_open:
        verdict = "review"
        reasons = ["LOW_CONFIDENCE"]
        if provisional_accept:
            evidence.append(_evidence("accept_provisional", answer_conf,
                                      ">=", accept_conf, "LOW_CONFIDENCE"))
        else:
            evidence.append(_evidence("accept_uncalibrated", answer_conf,
                                      ">=", accept_conf, "LOW_CONFIDENCE"))
        queue_priority = _review_priority()
    else:
        verdict = "accept"
        reasons = []
        queue_priority = 0.0
        if provisional_accept:
            evidence.append({
                "check": "accept_status",
                "value": "provisional",
                "op": "==",
                "threshold": "allow_provisional_accept",
                "code": "LOW_CONFIDENCE",
                "label_source": calib_source,
            })
        elif calib_status != "calibrated":
            evidence.append(_evidence("allow_uncalibrated_accept", True,
                                      "==", True, "LOW_CONFIDENCE"))

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
    "delta": 0.1,           # 1 - delta confidence for the family
    "target_beta": 0.05,    # false-reject rate per reject rule
    "min_support": 10,      # CERTIFY support required to write a threshold
    "min_accepted": 30,     # accept side must cover this many SELECT clips
    "accept_step": 0.05,    # accept-side candidate grid resolution
    "seed": 1234,           # salts the deterministic group split
    "bonferroni": True,     # split delta across the m tunable reject
    # rules (each tested at delta/m); False recovers the round-1
    # behaviour (each rule at full delta — NOT family-wise, say so).
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
        group = record.get("group")
        wearer = record.get("wearer")
        if wearer is not None and not isinstance(wearer, (str, list, tuple)):
            raise ValueError(f"record {i} wearer must be a string or a list")
        labeller = record.get("labeller")
        if labeller is not None and not isinstance(labeller, str):
            raise ValueError(f"record {i} labeller must be a string")
        cleaned.append({
            "signals": dict(signals),
            "decisions": dict(decisions) if decisions is not None else None,
            "label": label,
            "reason_gold": record.get("reason_gold"),
            "label_source": source,
            # Optional grouping for the SELECT/CERTIFY split (clip id or
            # near-duplicate component). Absent => each record is its own
            # group (fine for independent synthetic draws).
            "group": str(group) if group is not None else None,
            "segment_id": record.get("segment_id"),
            # Optional provenance: wearer id(s) for this record's clip,
            # labeller id for human labels (synthetic/gemini => none).
            "wearer": (list(wearer) if isinstance(wearer, (list, tuple))
                       else wearer),
            "labeller": labeller,
        })
    if not cleaned:
        raise ValueError("need at least one labelled record")
    return cleaned


def _content_key(record: dict) -> str:
    return hashlib.sha256(canonical_json({
        "signals": record["signals"], "decisions": record["decisions"],
        "label": record["label"], "reason_gold": record.get("reason_gold"),
        "label_source": record["label_source"],
    })).hexdigest()


def _split_select_certify(records: list[dict],
                          seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic SELECT/CERTIFY split by CLIP GROUP, never by segment.

    Records sharing a ``group`` (clip id / near-duplicate component, else
    their content hash) always land on the SAME side: groups are ordered by
    ``sha256(seed + group)`` and dealt alternately to SELECT and CERTIFY,
    so neighbouring segments of one clip cannot leak across the split
    (audit L3: with 7 clips this is a 4-clip SELECT / 3-clip CERTIFY
    partition by hash order). No RNG anywhere; input order is irrelevant.
    """
    groups: dict[str, list[dict]] = {}
    for record in records:
        key = record["group"] if record["group"] is not None \
            else "content:" + _content_key(record)
        groups.setdefault(key, []).append(record)
    ordered = sorted(
        groups, key=lambda g: hashlib.sha256(
            f"{int(seed)}/funnel-select-certify/{g}".encode()).hexdigest())
    select: list[dict] = []
    certify: list[dict] = []
    for i, key in enumerate(ordered):
        (select if i % 2 == 0 else certify).extend(groups[key])
    return select, certify


def _source_provenance(subset: list[dict], source: str) -> dict:
    """Provenance printed and stored beside EVERY bound (audit L3/L7).

    ``n_segments`` / ``n_clips`` (distinct clip groups) / ``n_wearers``
    (distinct ``wearer`` ids parsed from clip names where available, else
    ``'unknown'``) / ``n_labellers`` (distinct ``labeller`` ids from the
    gold file; synthetic/gemini => 0 humans; human_gold with no labeller
    ids recorded => ``'unknown'``) / ``label_source``.
    """
    wearers: set[str] = set()
    for record in subset:
        wearer = record.get("wearer")
        if isinstance(wearer, str):
            wearers.add(wearer)
        elif isinstance(wearer, (list, tuple)):
            wearers.update(str(w) for w in wearer)
    labellers = {r["labeller"] for r in subset
                 if r.get("labeller") is not None}
    if source in ("synthetic", "gemini"):
        n_labellers: Any = 0
    elif labellers:
        n_labellers = len(labellers)
    else:
        n_labellers = "unknown"
    return {
        "label_source": source,
        "n_segments": len(subset),
        "n_clips": len({r["group"] for r in subset
                        if r["group"] is not None}),
        "n_wearers": len(wearers) if wearers else "unknown",
        "n_labellers": n_labellers,
    }


def calibration_content_hash(result: Mapping[str, Any]) -> str:
    """Stable sha256 of a calibration result, excluding ``sidecar`` (M3).

    Wall-clock/provenance stamps (``produced_utc``) live under
    ``result['sidecar']`` and never enter compared or hashed JSON, so two
    runs of the same records hash identically regardless of when they ran.
    """
    body = {k: v for k, v in dict(result).items() if k != "sidecar"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


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


def _measure(rule: dict, signals: Mapping[str, Any]) -> float | None:
    value = _as_float(signals.get(rule["signal"]))
    if value is None and rule["signal"] == "duration_s":
        value = _as_float(signals.get("segment_duration_s"))
    return value


def _conf_of(record: dict) -> float | None:
    heads = _head_probs(record["decisions"])
    return min(heads) if heads else None


def _calibrate_source(subset: list[dict], cfg: dict, delta: float,
                      delta_rule: float, support_floor: int,
                      source: str) -> dict:
    """Select on SELECT, certify on CERTIFY (disjoint by clip group).

    Reject rules are tested at ``delta_rule`` (Bonferroni ``delta/m`` by
    default; the per-rule delta used is reported). The accept side is a
    single pre-registered hypothesis tested at full ``delta`` (stated,
    not family-wise), and the UNION headline is measured once on CERTIFY
    at full ``delta`` — per-rule bounds do NOT bound the union. A rule
    that cannot be certified on held-out clips is DISABLED (threshold
    null, reason says so): certification stopped a bad rule.
    """
    alpha = float(cfg["target_alpha"])
    beta = float(cfg["target_beta"])
    min_support = int(cfg["min_support"])
    min_accepted = int(cfg["min_accepted"])
    accept_step = float(cfg["accept_step"])
    select, certify = _split_select_certify(subset, int(cfg["seed"]))

    # -- (a) accept side: select on SELECT, gate on CERTIFY ----------------
    select_conf = [(_conf_of(r), r["label"] == "good") for r in select]
    n_no_conf = sum(1 for conf, _ in select_conf if conf is None)
    usable = [(conf, ok) for conf, ok in select_conf if conf is not None]
    quantised = [round(math.floor(c / accept_step) * accept_step, 10)
                 for c, _ in usable]
    picked = certify_mod.pick_threshold(
        conf=quantised, correct=[ok for _, ok in usable],
        target_err=alpha, delta=delta, min_accepted=min_accepted)
    candidate = (round(float(picked["threshold"]), 4)
                 if picked["threshold"] is not None else None)
    accept_out: dict[str, Any] = {
        "candidate": candidate,
        "n_select": len(select),
        "n_no_conf": n_no_conf,
        "select_certified": {
            "accepted": picked["accepted"],
            "errors": picked["errors"],
            "upper_bound": (round(picked["upper_bound"], 4)
                            if picked["upper_bound"] is not None else None),
            "coverage": round(picked["coverage"], 4),
        },
    }
    accept_threshold: float | None = None
    if candidate is not None:
        fired = [r for r in certify
                 if (_conf_of(r) is not None and _conf_of(r) >= candidate)]
        stats = _rate_bound([r["label"] for r in fired], delta,
                            error_label="bad")
        accept_out["certify"] = {**stats, "n_certify": len(certify)}
        if stats["upper_bound"] is not None \
                and stats["upper_bound"] <= alpha \
                and stats["fired"] >= min_support:
            accept_threshold = candidate
        else:
            accept_out["gate"] = (
                "candidate rejected on CERTIFY: "
                f"bound={stats['upper_bound']} (target {alpha}), "
                f"support={stats['fired']} (min {min_support})")
    else:
        accept_out["certify"] = {"fired": 0, "errors": 0, "upper_bound": None,
                                 "n_certify": len(certify),
                                 "note": "no candidate selected: "
                                         "auto-accept stays disabled"}
    accept_out["threshold"] = accept_threshold

    # -- (b) reject side: select per rule on SELECT, gate on CERTIFY -------
    reject_out: dict[str, dict] = {}
    updates: dict[str, Any] = {"accept_conf": accept_threshold}
    for rule in REJECT_RULES:
        key = rule["key"]
        measured = [(m, r) for r in select
                    if (m := _measure(rule, r["signals"])) is not None]
        values = sorted({_round4(m) for m, _ in measured})
        if rule["direction"] == "lt":
            candidates = values  # strictest (fewest fires) first
        else:
            candidates = values[::-1]

        def _fires(measured_pairs: list, threshold: float) -> list[dict]:
            return [r for m, r in measured_pairs
                    if (m < threshold if rule["direction"] == "lt"
                        else m > threshold)]

        # Label-independent support pre-filter (keeps the certificate);
        # support_floor also guarantees the first tested candidate CAN pass.
        eligible = [t for t in candidates
                    if sum(1 for m, _ in measured
                           if (m < t if rule["direction"] == "lt"
                               else m > t)) >= support_floor]
        best: float | None = None
        best_stats: dict | None = None
        for threshold in eligible:
            stats = _rate_bound(
                [r["label"] for r in _fires(measured, threshold)], delta_rule)
            if stats["upper_bound"] is not None \
                    and stats["upper_bound"] <= beta:
                best = threshold
                best_stats = stats
            else:
                break  # fixed-sequence stopping rule
        cert_measured = [(m, r) for r in certify
                         if (m := _measure(rule, r["signals"])) is not None]
        if best is not None:
            cert_fired = _fires(cert_measured, best)
            cert_stats = _rate_bound(
                [r["label"] for r in cert_fired], delta_rule)
            gate_ok = cert_stats["upper_bound"] is not None \
                and cert_stats["upper_bound"] <= beta \
                and cert_stats["fired"] >= min_support
        else:
            cert_stats = {"fired": 0, "errors": 0, "upper_bound": None}
            gate_ok = False
        if best is not None and gate_ok:
            entry: dict[str, Any] = {
                "threshold": best, "disabled": False,
                "code": rule["code"],
                "select": best_stats,
                "certify": {**cert_stats, "n_certify": len(certify)},
                "n_eligible_candidates": len(eligible),
            }
        else:
            if best is None:
                reason = ("no candidate certified on SELECT: the strictest "
                          f"candidate with support >= {support_floor} already "
                          "exceeds target_beta (or no candidate reaches "
                          "support)")
            else:
                reason = ("SELECT candidate rejected on CERTIFY: "
                          f"bound={cert_stats['upper_bound']} "
                          f"(target {beta}), support={cert_stats['fired']} "
                          f"(min {min_support})")
            entry = {
                "threshold": None, "disabled": True,
                "code": rule["code"],
                "reason": reason,
                "select_candidate": best,
                "certify": {**cert_stats, "n_certify": len(certify)},
                "n_eligible_candidates": len(eligible),
            }
        reject_out[key] = entry
        updates[key] = entry["threshold"]

    # -- (c) union: measured end-to-end on CERTIFY with selected thresholds.
    # Each rule was tested at delta_rule (Bonferroni delta/m by default);
    # the UNION false-reject rate of route() itself is measured on CERTIFY
    # with its exact bound at full delta. Per-rule bounds do NOT bound
    # the union — only this number describes route().
    route_thresh = {k: v for k, v in updates.items()}
    union_rejected = [
        r for r in certify
        if route(r["signals"], r["decisions"], route_thresh,
                 allow_uncalibrated_accept=True)["verdict"] == "reject"]
    union_stats = _rate_bound([r["label"] for r in union_rejected], delta)
    union = {**union_stats, "n_certify": len(certify),
             "note": ("union of all selected reject rules as route() "
                      "applies them; per-rule bounds do not bound this")}

    select_groups = sorted({r["group"] for r in select
                            if r["group"] is not None})
    certify_groups = sorted({r["group"] for r in certify
                             if r["group"] is not None})
    return {
        "n_total": len(subset),
        "n_select": len(select),
        "n_certify": len(certify),
        "select_groups": select_groups,
        "certify_groups": certify_groups,
        "n_select_clips": len(select_groups),
        "n_certify_clips": len(certify_groups),
        "provenance": {**_source_provenance(subset, source),
                       "delta_per_rule": round(delta_rule, 4)},
        "accept": accept_out,
        "reject_rules": reject_out,
        "union": union,
        "threshold_updates": updates,
    }


def calibrate_two_sided(records: list[dict],
                        targets: dict | None = None) -> dict:
    """Two-sided calibration with exact-binomial certificates (deterministic).

    ``records``: ``{signals, decisions|None, label: 'good'|'bad',
    reason_gold: code|None, label_source}`` plus optional ``group``
    (clip id / near-duplicate component) and ``segment_id``. ``targets``
    overrides any of :data:`DEFAULT_TARGETS`.

    NEVER POOLED: each label source is calibrated independently and the
    result carries ``per_source`` sub-results — no bound is ever computed
    over a mix of sources.

    Per source: records are split into SELECT and CERTIFY parts by CLIP
    GROUP (same clip never spans the split; see
    :func:`_split_select_certify`).
    (a) ACCEPT side: ``answer_conf`` threshold selected with
    ``jadal.certify.pick_threshold`` (fixed-sequence, exact binomial, confs
    floored to the fixed ``accept_step`` grid) on SELECT; the threshold is
    WRITTEN only if the CERTIFY defect bound <= ``target_alpha`` with
    CERTIFY support >= ``min_support``. I5: when SELECT holds fewer than
    ``min_accepted`` high-confidence clips (low-prevalence good data), no
    candidate is selected and ``accept_conf`` stays null — auto-accept
    stays disabled and the output says so. (b) REJECT side: per tunable
    rule, candidates strictest-first on SELECT with the same
    fixed-sequence logic, each rule tested at the Bonferroni
    ``delta/m`` (``m`` = number of tunable rules; the per-rule delta used
    is reported; ``bonferroni: False`` recovers per-rule full delta and
    must then be STATED as not family-wise); WRITTEN only if the CERTIFY
    false-reject bound <= ``target_beta`` with CERTIFY support >=
    ``min_support``, else the rule is DISABLED (threshold None) and the
    reason says so — certification stopped a bad rule. (c) UNION: the
    measured false-reject rate of the full route() union on CERTIFY with
    its exact bound at full ``delta`` (per-rule bounds do not bound the
    union). Beside EVERY bound the output stores ``provenance``:
    ``n_segments`` / ``n_clips`` / ``n_wearers`` / ``n_labellers`` /
    ``label_source``.

    Pure function: no I/O, no randomness, no clock. Returns
    ``{"per_source", "threshold_updates" (per source), "meta"}``; all
    floats rounded to 4 decimals.
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
    bonferroni = bool(cfg["bonferroni"])
    if not 0.0 < alpha < 1.0:
        raise ValueError("target_alpha must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("target_beta must be in (0, 1)")
    if not 0.0 < accept_step <= 1.0:
        raise ValueError("accept_step must be in (0, 1]")
    # Family-wise error (audit L3): the m tunable reject rules share the
    # budget — each is tested at delta/m by default. The support floor is
    # computed from the same per-rule delta, so the first tested candidate
    # CAN pass its certificate.
    m_rules = len(REJECT_RULES)
    delta_rule = delta / m_rules if bonferroni else delta
    # Smallest support at which a perfectly clean rule can pass the bound.
    n_power = math.ceil(math.log(delta_rule) / math.log(1.0 - beta))
    support_floor = max(min_support, n_power)

    cleaned = _validate_records(records)
    per_source: dict[str, dict] = {}
    updates_all: dict[str, dict] = {}
    by_source: dict[str, int] = {}
    for source in LABEL_SOURCES:
        subset = [r for r in cleaned if r["label_source"] == source]
        if not subset:
            continue
        by_source[source] = len(subset)
        per_source[source] = _calibrate_source(subset, cfg, delta,
                                               delta_rule, support_floor,
                                               source)
        updates_all[source] = per_source[source]["threshold_updates"]
    meta = {
        "n_total": len(cleaned),
        "by_source": by_source,
        "targets": {k: cfg[k] for k in
                    ("target_alpha", "delta", "target_beta", "min_support",
                     "min_accepted", "accept_step", "bonferroni")},
        "m_rules": m_rules,
        "delta_per_rule": round(delta_rule, 4),
        "support_floor": support_floor,
        "seed": seed,
        "method": ("per-source independent calibration; clip-group-based "
                    "SELECT/CERTIFY split; Bonferroni delta/m across the m "
                    "tunable reject rules by default (accept side and union "
                    "at full delta, stated); no bound is ever pooled across "
                    "label sources"),
    }
    return {"per_source": per_source, "threshold_updates": updates_all,
            "meta": meta}


__all__ = [
    "REASON_ORDER", "CHECK_ORDER", "DEFAULT_THRESHOLDS_PATH",
    "LABEL_SOURCES", "REJECT_RULES", "DEFAULT_TARGETS",
    "HARD_REJECT_POLICIES", "QUALITY_RULE_CODES",
    "canonical_json", "thresholds_hash", "load_thresholds",
    "queue_rank_key", "route", "calibrate_two_sided",
    "calibration_content_hash",
]
