"""Deterministic physics rule-list model: integer-quantised decision path.

This module is the shipping implementation of the Q1 rule table. It contains NO
learned weights. The **rule decision path compares quantised integers only**:
every feature value and every threshold is converted to an integer at
10**quantisation (Q=4) *before* comparison, the score is the integer count of
crossed thresholds, and ACCEPT / REJECT / REVIEW is a pure function of integers.
Evidence is emitted as ``{rule, feature, value, op, threshold}`` with the integer
value and threshold.

The *optional* logistic fallback (hybrid mode) is the one **float** comparison in
the receipt path: a REVIEW window is resolved with ``usable_prob >= 0.5``. Its
source is bound into the receipt's ``fallback.source_sha256``, and the
deterministic decision is kept in ``decision_deterministic``.

Only the *input extraction* (decode + feature kernels) uses floats, exactly as
in Q1; the scorer never sees them. The canonical receipt binds the result to
the rule table (sha256) and the decision code (sha256).

Public API
----------
load_rules(path=None) -> dict
quantise(value, q) -> int | None
feature_digest(features, rules) -> str
score_window(features, rules) -> dict
decide(scored, rules) -> dict
receipt(window_id, features, rules, fallback_prob=None, fallback_sha256=None) -> dict
"""
from __future__ import annotations

import hashlib
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RULES = os.path.join(HERE, "rules.json")
DEFAULT_Q = 4

# Exit codes for the CLI (deterministic, documented).
EXIT_ACCEPT = 0
EXIT_REJECT = 1
EXIT_REVIEW = 2


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_sha256() -> str:
    """Digest of this decision implementation (the file that computes decisions)."""
    return sha256_file(os.path.abspath(__file__))


def load_rules(path: str | None = None) -> dict:
    """Load the frozen rule table and bind its sha256."""
    path = path or DEFAULT_RULES
    with open(path, "r") as f:
        raw = f.read()
    rules = json.loads(raw)
    rules["_path"] = os.path.abspath(path)
    rules["_sha256"] = sha256_bytes(raw.encode("utf-8"))
    # thresholds_q is authoritative; derive it if a hand-edited table omits it.
    q = int(rules.get("quantisation", DEFAULT_Q))
    for r in rules["rules"]:
        if "thresholds_q" not in r:
            r["thresholds_q"] = [int(round(float(t) * 10 ** q)) for t in r["thresholds"]]
    return rules


def quantise(value, q: int = DEFAULT_Q):
    """Deterministic integer quantisation. Non-finite -> None (never guessed)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    # Half-up rounding, fixed for both signs of the scale factor.
    scaled = v * (10 ** q)
    return int(math.floor(scaled + 0.5)) if scaled >= 0 else int(math.ceil(scaled - 0.5))


def feature_digest(features: dict, rules: dict) -> str:
    """sha256 over the quantised values of every feature the rule table reads."""
    q = int(rules.get("quantisation", DEFAULT_Q))
    needed = sorted({r["feature"] for r in rules["rules"]})
    lines = []
    for name in needed:
        vq = quantise(features.get(name), q)
        lines.append(f"{name}={'' if vq is None else vq}")
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def score_window(features: dict, rules: dict) -> dict:
    """Integer score + fired evidence. Pure function of quantised integers."""
    q = int(rules.get("quantisation", DEFAULT_Q))
    score = 0
    fired = []
    for r in rules["rules"]:
        vq = quantise(features.get(r["feature"]), q)
        if vq is None:
            continue
        crossed = 0
        best_thr = None
        best_margin = None  # smallest integer margin (closest threshold)
        hard = False        # crossing threshold index 0 (the p01/p99 extreme)
        for idx, tq in enumerate(r["thresholds_q"]):
            if r["op"] == "<":
                hit = vq < tq
                margin = tq - vq
            else:
                hit = vq > tq
                margin = vq - tq
            if hit:
                crossed += 1
                if idx == 0:
                    hard = True
                if best_margin is None or margin < best_margin:
                    best_margin = margin
                    best_thr = tq
        if crossed:
            score += crossed
            fired.append({
                "rule": r["rule"], "feature": r["feature"], "value": vq,
                "op": r["op"], "threshold": best_thr,
                "strength": r["strength"], "n_thresholds_crossed": crossed,
                "hard": hard,
            })
    # UNDECODABLE tripwire: any core feature missing / non-finite.
    und = [f for f in rules.get("undecodable_features", [])
           if quantise(features.get(f), q) is None]
    if und:
        score += int(rules.get("undecodable_score", 10 ** 6))
        fired.append({
            "rule": "UNDECODABLE", "feature": und[0], "value": None,
            "op": "is_finite", "threshold": None, "strength": "quality",
            "n_thresholds_crossed": 1, "hard": True,
        })
    # A quality rule is a HARD reject only when it crosses its extreme (p01/p99)
    # threshold; softer crossings only add to the integer score.
    strong = sorted({e["rule"] for e in fired
                     if e["strength"] == "quality" and e["hard"]})
    weak = sorted({e["rule"] for e in fired if e["strength"] == "weak"})
    return {"score": int(score), "fired": fired,
            "fired_rules": sorted({e["rule"] for e in fired}),
            "strong_fired": strong, "weak_fired": weak}


def decide(scored: dict, rules: dict) -> dict:
    """ACCEPT / REJECT / REVIEW from the integer score + fired quality rules."""
    reject_threshold = int(rules["reject_threshold"])
    if scored["strong_fired"]:
        ev = next(e for e in scored["fired"]
                  if e["strength"] == "quality" and e.get("hard"))
        return {"decision": "REJECT", "reason": "quality_rule",
                "evidence": ev, "score": scored["score"]}
    if scored["score"] >= reject_threshold:
        ev = (max(scored["fired"], key=lambda e: e["n_thresholds_crossed"])
              if scored["fired"] else None)
        return {"decision": "REJECT", "reason": "score>=reject_threshold",
                "evidence": ev, "score": scored["score"]}
    if scored["score"] >= 1:
        ev = scored["fired"][0] if scored["fired"] else None
        return {"decision": "REVIEW", "reason": "weak_rule",
                "evidence": ev, "score": scored["score"]}
    return {"decision": "ACCEPT", "reason": "no_rule", "evidence": None,
            "score": scored["score"]}


def _evidence_list(scored: dict) -> list:
    """Canonical evidence: {rule, feature, value, op, threshold}, sorted."""
    out = []
    for e in sorted(scored["fired"], key=lambda x: (x["rule"], x["feature"],
                                                    x["threshold"] if x["threshold"] is not None else -1)):
        out.append({"rule": e["rule"], "feature": e["feature"],
                    "value": e["value"], "op": e["op"], "threshold": e["threshold"]})
    return out


def receipt(window_id: str, features: dict, rules: dict,
            fallback_prob=None, fallback_sha256: str | None = None) -> dict:
    """Canonical JSON receipt for one window.

    If ``fallback_prob`` is given and the deterministic decision is REVIEW, the
    window is resolved by the frozen logistic fallback (ACCEPT iff p >= 0.5);
    the receipt records this in ``fallback`` (including ``source_sha256`` when
    provided) while keeping the deterministic ``decision_deterministic`` field.
    """
    scored = score_window(features, rules)
    det = decide(scored, rules)
    decision = det["decision"]
    fb = None
    if fallback_prob is not None and decision == "REVIEW":
        p = float(fallback_prob)
        decision = "ACCEPT" if p >= 0.5 else "REJECT"
        fb = {"model": "ensemble_v1.usable_prob", "prob": round(p, 6),
              "rule": "accept iff prob>=0.5", "decision": decision,
              "source_sha256": fallback_sha256}
    rec = {
        "code_sha256": code_sha256(),
        "decision": decision,
        "evidence": _evidence_list(scored),
        "feature_digest": feature_digest(features, rules),
        "fired_rules": scored["fired_rules"],
        "rule_table_sha256": rules["_sha256"],
        "score": scored["score"],
        "window_id": window_id,
    }
    if fallback_prob is not None:
        rec["decision_deterministic"] = det["decision"]
        rec["fallback"] = fb
    return rec


def canonical_json(obj) -> str:
    """Deterministic serialisation: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def decision_exit_code(decision: str) -> int:
    return {"ACCEPT": EXIT_ACCEPT, "REJECT": EXIT_REJECT, "REVIEW": EXIT_REVIEW}.get(
        decision, EXIT_REVIEW)
