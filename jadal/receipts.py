"""Content-hashed, re-verifiable receipts for per-segment QA verdicts.

A receipt pins the segment hash, L0 signals, L1 decisions, the L2 verdict and
the pipeline config, then records the SHA-256 of its canonical JSON
(`receipt_sha256`) plus a short `receipt_id`. Re-verify re-runs the pipeline
via a caller-supplied function and diffs the field paths.

NOT DONE:
- "Signed" here means content-hashed, not private-key signed; no persistence or
  lookup by `receipt_id` (that belongs to the service layer).
- This module never reads segment files itself; the `recompute_fn` passed to
  `verify_receipt` owns re-reading the original and re-running the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping
from dataclasses import is_dataclass
from datetime import datetime, timezone

HASHED_EXCLUDED = frozenset({"receipt_sha256", "receipt_id", "created_utc"})
DEFAULT_FLOAT_DECIMALS = 4

# Config dataclass fields that enter the receipt's config block (JSON-safe only).
THRESHOLD_KEYS = (
    "batch_size",
    "batch_reject_k",
    "segment_seconds",
    "seed",
    "float_decimals",
    "blur_var_min",
    "brightness_min",
    "over_exposed_frac_max",
    "motion_max",
    "hands_visible_frac_min",
    "target_error",
    "delta",
    "min_accepted",
)


def canonical_json(obj) -> bytes:
    """Canonical JSON bytes: sorted keys, compact separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _check_finite(value, path: str) -> None:
    """Reject NaN/inf before anything is hashed.

    A NaN would serialize as the non-standard JSON token ``NaN`` and poison
    every downstream consumer, so it is a hard error naming the field path.
    Booleans are not floats and pass through untouched.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path}: {value!r}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]")
        return


def _round_floats(value, decimals: int):
    """Recursively round floats to `decimals`; `-0.0` becomes `0.0`."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, decimals) + 0.0
    if isinstance(value, Mapping):
        return {key: _round_floats(item, decimals) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item, decimals) for item in value]
    return value


def _config_block(config) -> tuple[dict, int]:
    """Build the receipt `config` block and return it with the float precision."""
    if is_dataclass(config) and not isinstance(config, type):
        decimals = int(getattr(config, "float_decimals", DEFAULT_FLOAT_DECIMALS))
        thresholds = {
            key: getattr(config, key) for key in THRESHOLD_KEYS if hasattr(config, key)
        }
        versions = {
            "python": platform.python_version(),
            "config_sha256": hashlib.sha256(
                canonical_json(
                    {
                        "thresholds": thresholds,
                        "pipeline_version": getattr(config, "pipeline_version", "?"),
                    }
                )
            ).hexdigest(),
        }
        block = {
            "thresholds": thresholds,
            "pipeline_version": getattr(config, "pipeline_version", "?"),
            "versions": versions,
        }
        return block, decimals
    if isinstance(config, Mapping):
        missing = {"thresholds", "pipeline_version", "versions"} - set(config.keys())
        if missing:
            raise ValueError(f"config mapping misses keys: {sorted(missing)}")
        block = {key: config[key] for key in ("thresholds", "pipeline_version", "versions")}
        decimals = DEFAULT_FLOAT_DECIMALS
        thresholds = block["thresholds"]
        if isinstance(thresholds, Mapping) and "float_decimals" in thresholds:
            try:
                decimals = int(thresholds["float_decimals"])
            except (TypeError, ValueError):
                decimals = DEFAULT_FLOAT_DECIMALS
        return block, decimals
    raise ValueError("config must be a Config dataclass or a config mapping")


def receipt_hash(receipt: Mapping) -> str:
    """SHA-256 hex of the canonical JSON of the receipt minus volatile keys."""
    if not isinstance(receipt, Mapping):
        raise ValueError("receipt must be a mapping")
    body = {key: value for key, value in receipt.items() if key not in HASHED_EXCLUDED}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def make_receipt(
    segment: Mapping,
    signals: Mapping,
    decisions: Mapping,
    verdict: Mapping,
    config,
    created_utc: str | None = None,
) -> dict:
    """Build a receipt dict; inputs are never mutated. Hash excludes the id,
    the hash itself and the creation timestamp."""
    if not isinstance(segment, Mapping) or "sha256" not in segment:
        raise ValueError("segment must be a mapping with a 'sha256' field")
    for name, payload in (("signals", signals), ("decisions", decisions), ("verdict", verdict)):
        if not isinstance(payload, Mapping):
            raise ValueError(f"{name} must be a mapping")
    block, decimals = _config_block(config)
    for name, payload in (
        ("signals", signals),
        ("decisions", decisions),
        ("verdict", verdict),
        ("config", block),
    ):
        _check_finite(payload, name)
    body = {
        "segment_sha256": segment["sha256"],
        "signals": _round_floats(signals, decimals),
        "decisions": _round_floats(decisions, decimals),
        "verdict": _round_floats(verdict, decimals),
        "config": _round_floats(block, decimals),
    }
    receipt = dict(body)
    receipt["created_utc"] = created_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    receipt["receipt_id"] = receipt["receipt_sha256"][:16]
    return receipt


def _diff_paths(old, new, prefix: str = "") -> list[str]:
    """Field paths where two (already stripped) payloads differ, sorted."""
    paths: list[str] = []
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        for key in sorted(set(old.keys()) | set(new.keys())):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                paths.append(path)
            else:
                paths.extend(_diff_paths(old[key], new[key], path))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (left, right) in enumerate(zip(old, new)):
            paths.extend(_diff_paths(left, right, f"{prefix}[{i}]"))
    elif old != new:
        paths.append(prefix)
    return paths


def verify_receipt(receipt: Mapping, recompute_fn) -> dict:
    """Re-verify a receipt against a fresh pipeline run.

    `recompute_fn` is called with the stored receipt and must return the
    recomputed receipt mapping. Returns `{"match", "expected", "actual",
    "diff"}` where `diff` lists the differing field paths (dotted, lists
    indexed) of the hashed payload.
    """
    if not isinstance(receipt, Mapping):
        raise ValueError("receipt must be a mapping")
    expected = receipt.get("receipt_sha256")
    recomputed = recompute_fn(receipt)
    if not isinstance(recomputed, Mapping):
        raise ValueError("recompute_fn must return a receipt mapping")
    actual = receipt_hash(recomputed)
    # receipt_id is excluded from the hash by design (SPEC), but a stored
    # receipt whose id does not match the recomputed hash of its own body is
    # internally inconsistent. The id is checked against `actual` (not the
    # stored hash) so that tampering with receipt_sha256 alone does not also
    # raise a spurious receipt_id diff.
    id_inconsistent = (
        "receipt_id" in receipt and receipt["receipt_id"] != actual[:16]
    )
    match = expected is not None and expected == actual and not id_inconsistent
    if match:
        return {"match": True, "expected": expected, "actual": actual, "diff": []}
    stripped_stored = {
        key: value for key, value in receipt.items() if key not in HASHED_EXCLUDED
    }
    stripped_fresh = {
        key: value for key, value in recomputed.items() if key not in HASHED_EXCLUDED
    }
    diff = sorted(_diff_paths(stripped_stored, stripped_fresh))
    if not diff:
        # Bodies are identical, so the mismatch must come from the envelope:
        # a wrong/missing stored hash, or an id that does not name the hash.
        if receipt.get("receipt_sha256") != actual:
            diff = ["receipt_sha256"]
        if id_inconsistent and "receipt_id" not in diff:
            diff = sorted(diff + ["receipt_id"])
    return {"match": False, "expected": expected, "actual": actual, "diff": diff}
