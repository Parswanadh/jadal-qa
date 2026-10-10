"""Deterministic rule-scorer entry point for the Jadal receipt proof (Q3).

Scope: the *deterministic rule path* only — the frozen quality tripwires from
Q1's ``jadal/perpixel/scorer.py`` (rule list ``configs/perpixel_v1.json``).
No model, no GPU, no I/O inside the decision, no clock, no randomness.

Guarantees enforced here:
  * Q=4 integer quantisation: every feature is converted to a fixed-point
    integer ``round(value * 10**4)`` using ``decimal`` ROUND_HALF_EVEN, so the
    *decision path contains no floating-point comparisons at all*.
  * canonical JSON: sorted keys, compact separators, UTF-8, ``ensure_ascii=False``.
  * receipts carry no time/random/device fields; only the schema fields
    plus the ``q`` scale.
  * ``rule_table_sha256`` and ``code_sha256`` pin the rule constants and the
    scorer source into every receipt.
  * **fail closed on missing features**: a required feature that is absent /
    non-finite is recorded in ``missing_features`` and the window is rejected
    with ``reject:missing_feature`` (it can never silently pass).

Receipt body (hashed)::

    {window_id, decision, evidence:[{rule,feature,value,op,threshold,missing}],
     missing_features, feature_digest, rule_table_sha256, code_sha256, q}

``receipt_sha256`` is the SHA-256 of the canonical JSON of that body and is the
only envelope field, so it is reproducible byte-for-byte across runs on the same
machine (verified on one x86_64 CPU). This is a same-machine guarantee, not a
cross-machine one.
"""
from __future__ import annotations

import hashlib
import json
import os
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 4
QI = 10 ** Q
MISSING = -99999999  # sentinel for absent/NaN features (never fires a < rule)
FORBIDDEN_KEYS = frozenset({
    "time", "timestamp", "created_utc", "created", "utc", "date", "now",
    "random", "nonce", "uuid", "seed", "elapsed", "latency", "wall_s",
    "device", "gpu", "cuda",
})


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------
def canonical_json(obj) -> bytes:
    """Canonical JSON bytes: sorted keys, compact separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_sha256() -> str:
    """SHA-256 of this scorer's source bytes (pins the decision code)."""
    with open(os.path.abspath(__file__), "rb") as fh:
        return sha256_hex(fh.read())


def rule_table_sha256(rule_table: dict) -> str:
    """SHA-256 of the canonical JSON of the rule table (integer thresholds)."""
    return sha256_hex(canonical_json(rule_table))


def quantise(value) -> int:
    """Exact Q=4 fixed point: ``round(value * 10**4)``, ROUND_HALF_EVEN.

    Accepts Decimal / int / float / numeric string. ``None``/NaN/inf -> MISSING
    so a missing feature can never satisfy a ``<``/``>`` tripwire.
    """
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return MISSING
    if isinstance(value, Decimal):
        d = value
    elif isinstance(value, int):
        d = Decimal(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return MISSING
        d = Decimal(repr(value))
    else:
        try:
            d = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return MISSING
    if not d.is_finite():
        return MISSING
    return int((d * QI).to_integral_value(rounding=ROUND_HALF_EVEN))


# --------------------------------------------------------------------------
# the deterministic rule path
# --------------------------------------------------------------------------
def load_rule_table(path: str | None = None) -> dict:
    src = path or os.path.join(HERE, "rule_table.json")
    with open(src, "r", encoding="utf-8") as fh:
        table = json.load(fh)
    if table.get("q") != Q:
        raise ValueError(f"rule table q={table.get('q')} != {Q}")
    for r in table["rules"]:
        if r["op"] not in ("<", ">"):
            raise ValueError(f"unsupported op {r['op']!r}")
        if not isinstance(r["thr_int"], int):
            raise ValueError(f"threshold {r['name']} must be a Q=4 integer")
    return table


def score_window(window: dict, rule_table: dict, *,
                 code_hash: str | None = None) -> dict:
    """Score one window -> canonical receipt (deterministic, pure).

    ``window`` = ``{"window_id": str, "features": {name: number}}``.
    """
    features = window.get("features") or {}
    evidence: list[dict] = []
    missing: list[str] = []
    hit_rule: str | None = None
    for r in rule_table["rules"]:
        feature = r["feature"]
        value_int = quantise(features.get(feature))
        thr_int = int(r["thr_int"])
        is_missing = value_int == MISSING
        if is_missing:
            missing.append(feature)
            hit = False
        elif r["op"] == "<":
            hit = value_int < thr_int
        else:
            hit = value_int > thr_int
        evidence.append({
            "rule": r["name"],
            "feature": feature,
            "value": value_int,
            "op": r["op"],
            "threshold": thr_int,
            "missing": is_missing,
        })
        if hit and hit_rule is None:
            hit_rule = r["name"]
    # Fail closed: a missing/non-finite required feature rejects the window.
    # It must never silently fall through to "pass".
    if missing:
        decision = "reject:missing_feature"
    elif hit_rule:
        decision = "reject:" + hit_rule
    else:
        decision = "pass"

    feat_q = {r["feature"]: quantise(features.get(r["feature"]))
              for r in rule_table["rules"]}
    feature_digest = sha256_hex(canonical_json(feat_q))

    body = {
        "window_id": str(window["window_id"]),
        "decision": decision,
        "evidence": evidence,
        "missing_features": sorted(set(missing)),
        "feature_digest": feature_digest,
        "rule_table_sha256": rule_table_sha256(rule_table),
        "code_sha256": code_hash if code_hash is not None else code_sha256(),
        "q": Q,
    }
    receipt = dict(body)
    receipt["receipt_sha256"] = sha256_hex(canonical_json(body))
    return receipt


def assert_no_forbidden_fields(obj, path: str = "") -> None:
    """Reject any time/random/device key anywhere in a receipt (recursive)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden key {path}.{key}")
            assert_no_forbidden_fields(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            assert_no_forbidden_fields(value, f"{path}[{i}]")
    elif isinstance(obj, float):
        raise ValueError(f"float in receipt at {path}: {obj!r}")
