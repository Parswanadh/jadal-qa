"""Deterministic physics rule model v2 -- integer-only decision path, fixed constants.

Design contract (v2):
  * Every feature value is quantised once, at the boundary, to an integer at Q=4:
        value_q = int(round(value * 10**Q))      # Q = 4
    All comparisons afterwards are INTEGER comparisons: no float ever decides.
  * Each rule carries up to four INTEGER thresholds (Q=4) on one feature and a
    direction (``side``: ``low`` = small is bad, ``high`` = large is bad).
  * ``score = sum over rules of w_rule * (# thresholds crossed)`` with ``w_rule`` a
    fixed non-negative INTEGER derived on training clips only (v1 is the special
    case where every ``w_rule == 1``). No float weight exists in the decision path.
  * ``strength == "quality"`` rules veto: crossing their EXTREME threshold
    (index 0, the outermost quantile) forces REJECT regardless of the score.
  * The verdict is a deterministic function of the integer score and the vetoes:
        score >= t_reject            -> REJECT
        t_accept <= score < t_reject -> REVIEW
        score <  t_accept            -> ACCEPT
  * Every decision emits, per crossed threshold, the evidence record
    ``{rule, feature, value_q, op, threshold_q, weight}`` and a canonical receipt
    ``sha256`` of the sorted-key, fixed-separator JSON of the whole record.

Nothing in this module fits a parameter at scoring time. The only data-derived
quantities are the integer thresholds and the integer weights, both frozen in
``configs/phys_rules_v2.json`` and both rounded from training-clip statistics.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence

Q = 4
SCALE = 10 ** Q


def quantise(value: float) -> int:
    """Quantise one feature value to an integer at Q=4 (round-half-away-from-zero).

    Deterministic: pure arithmetic on a float64, no locale, no RNG.
    """
    x = float(value) * SCALE
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def quantise_row(row: Mapping[str, float]) -> Dict[str, int]:
    """Quantise a whole feature record (finite values only)."""
    out: Dict[str, int] = {}
    for k, v in row.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv == fv and fv not in (float("inf"), float("-inf")):  # not NaN/inf
            out[k] = quantise(fv)
    return out


def crossed(rule: Mapping[str, Any], value_q: int) -> int:
    """How many of this rule's thresholds does ``value_q`` cross? (0..4)"""
    op = rule["op"]
    n = 0
    for t in rule["thresholds_q"]:
        if (op == "<" and value_q < t) or (op == ">" and value_q > t):
            n += 1
    return n


def extreme_crossed(rule: Mapping[str, Any], value_q: int) -> bool:
    """Did the value cross the rule's EXTREME (outermost) threshold?"""
    if not rule.get("thresholds_q"):
        return False
    t = rule["thresholds_q"][0]
    return (value_q < t) if rule["op"] == "<" else (value_q > t)


def _as_int(v: Any) -> int:
    """Accept either an already-quantised int or a raw float; quantise the latter.

    Keeps ``score_window`` correct even if a caller forgets to quantise first --
    comparing a raw float against Q=4 integer thresholds would be meaningless.
    """
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    return quantise(float(v))


def score_window(value_q: Mapping[str, Any], table: Mapping[str, Any]
                 ) -> Dict[str, Any]:
    """Deterministic score + evidence for one window.

    ``table`` is the loaded phys_rules_v2 config dict. Returns a JSON-serialisable
    record; ordering of ``evidence`` is the table's rule order (stable, not
    dict-iteration order of user input).
    """
    evidence: List[Dict[str, Any]] = []
    vetoes: List[Dict[str, Any]] = []
    score = 0
    per_rule: Dict[str, int] = {}
    for rule in table["rules"]:
        feat = rule["feature"]
        if feat not in value_q:
            continue  # missing feature never fires (documented, not a silent pass)
        raw = value_q[feat]
        if not isinstance(raw, (int, float)) or raw != raw:  # NaN guard
            continue
        vq = _as_int(raw)
        n = crossed(rule, vq)
        if n == 0:
            continue
        w = int(rule.get("weight", 1))
        score += w * n
        per_rule[rule["rule"]] = per_rule.get(rule["rule"], 0) + w * n
        first = True
        for t in rule["thresholds_q"]:
            hit = (vq < t) if rule["op"] == "<" else (vq > t)
            if hit:
                evidence.append({"rule": rule["rule"], "feature": feat,
                                 "value_q": vq, "op": rule["op"],
                                 "threshold_q": int(t), "weight": w,
                                 "grade": "extreme" if first else "soft"})
            first = False
        if rule.get("strength") == "quality" and extreme_crossed(rule, vq):
            vetoes.append({"rule": rule["rule"], "feature": feat, "value_q": vq,
                           "op": rule["op"],
                           "threshold_q": int(rule["thresholds_q"][0])})
    t_reject = int(table["t_reject"])
    t_accept = int(table.get("t_accept", 0))
    if vetoes:
        verdict = "REJECT"
    elif score >= t_reject:
        verdict = "REJECT"
    elif score >= t_accept:
        verdict = "REVIEW"
    else:
        verdict = "ACCEPT"
    return {"score": int(score), "verdict": verdict, "vetoed": bool(vetoes),
            "per_rule": {k: int(v) for k, v in sorted(per_rule.items())},
            "evidence": evidence, "vetoes": vetoes,
            "t_accept": t_accept, "t_reject": t_reject}


def canonical_receipt(record: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON: sorted keys, fixed separators, UTF-8."""
    blob = json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def decide(features: Mapping[str, float], table: Mapping[str, Any],
           window_id: str = "", model_name: str | None = None) -> Dict[str, Any]:
    """Full deterministic decision + receipt for one window."""
    vq = quantise_row(features)
    rec = score_window(vq, table)
    out = {"window_id": window_id,
           "model": model_name or table.get("name", "phys_rules_v2"),
           "verdict": rec["verdict"], "score": rec["score"],
           "vetoed": rec["vetoed"], "per_rule": rec["per_rule"],
           "evidence": rec["evidence"], "vetoes": rec["vetoes"],
           "t_accept": rec["t_accept"], "t_reject": rec["t_reject"],
           "Q": Q}
    out["receipt_sha256"] = canonical_receipt(out)
    return out


def load_table(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def score_records(records: Sequence[Mapping[str, Any]], table: Mapping[str, Any]
                  ) -> List[Dict[str, Any]]:
    """Score many windows in the given order; output order == input order."""
    return [decide(r.get("features", r), table, str(r.get("id", "")))
            for r in records]
