"""Batch pipeline for the Jadal QA service (GitHub issue #3).

Order per docs/SPEC.md: segmenter -> L0 signals -> L1 decisions -> L2 verdict
(with a threshold from certify) -> batch rule -> receipt. Every stage is per
segment (batch size 1), so a segment hashes identically alone or inside a batch
of five.

The real modules live on other branches. ``_resolve()`` prefers
``jadal.signals`` / ``jadal.decision`` / ``jadal.segmenter`` / ``jadal.certify`` /
``jadal.receipts`` whenever they exist and falls back to the deterministic stubs in
``jadal/stubs.py``, so this module keeps working (and keeps matching its tests)
before and after those merges.

NOT DONE:
- L0/L1/segmenter are stubs (hash-derived fakes, whole clip = one segment) until
  the real modules merge.
- ``pick_threshold`` has no calibration data, so the answer threshold is a
  placeholder (see stubs.STUB_THRESHOLD) recorded as ``answer_threshold``.
- Verdict reason rules live here (L2 certify API is thresholding + batch rule).
"""
from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jadal import stubs

# Reason codes, in the canonical order they are reported (docs/SPEC.md).
REASON_ORDER = [
    "UNDECODABLE", "HANDS_NOT_VISIBLE", "BLUR", "TOO_DARK", "OVEREXPOSED",
    "SHAKY", "DUPLICATE", "WRONG_STEP", "NO_ACTIVE_WORK", "LOW_CONFIDENCE",
]
# Reasons that on their own reject the segment; the rest are review-level.
HARD_REASONS = {
    "UNDECODABLE", "HANDS_NOT_VISIBLE", "BLUR", "TOO_DARK", "OVEREXPOSED",
    "SHAKY", "DUPLICATE", "NO_ACTIVE_WORK",
}

BATCH_RULE_TEXT = "reject if rejected_count > k"

_MODULES: dict[str, Any] | None = None


def _resolve() -> dict[str, Any]:
    """Load real module functions if importable, else None (stub fallback)."""
    global _MODULES
    if _MODULES is None:
        import importlib

        def load(name: str, attrs: list[str]) -> dict[str, Any]:
            try:
                mod = importlib.import_module(f"jadal.{name}")
            except Exception:  # module not on this branch yet
                return {a: None for a in attrs}
            return {a: getattr(mod, a, None) for a in attrs}

        _MODULES = {
            "signals": load("signals", ["compute_signals"]),
            "segmenter": load("segmenter", ["split_clip"]),
            "decision": load("decision", ["Decider"]),
            "certify": load("certify", ["pick_threshold", "batch_rule", "risk_coverage"]),
            "receipts": load("receipts", ["make_receipt", "receipt_hash",
                                          "verify_receipt", "canonical_json"]),
        }
    return _MODULES


def impl_versions() -> dict[str, str]:
    """Which implementation each stage is running (goes into every receipt)."""
    m = _resolve()
    return {
        "python": platform.python_version(),
        "signals": "jadal.signals" if m["signals"]["compute_signals"] else "stubs.signals",
        "decision": "jadal.decision" if m["decision"]["Decider"] else "stubs.decision",
        "certify": "jadal.certify" if m["certify"]["batch_rule"] else "stubs.certify",
        "receipts": "jadal.receipts" if m["receipts"]["make_receipt"] else "stubs.receipts",
        "segmenter": "jadal.segmenter" if m["segmenter"]["split_clip"] else "stubs.segmenter",
    }


def _round_floats(obj: Any, decimals: int) -> Any:
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------
# stage wrappers (real module if present, stub otherwise)
# --------------------------------------------------------------------------
def split_clip(path: str, seg_seconds: float, out_dir: str | Path) -> list[dict]:
    fn = _resolve()["segmenter"]["split_clip"] or stubs.stub_split_clip
    return fn(path, seg_seconds, out_dir)


def compute_signals(path: str, config: Any) -> dict:
    fn = _resolve()["signals"]["compute_signals"] or stubs.stub_compute_signals
    return fn(path, config)


def decide(path: str, signals: dict, config: Any) -> dict:
    cls = _resolve()["decision"]["Decider"] or stubs.StubDecider
    try:
        decider = cls(config)
    except TypeError:
        decider = cls()
    return decider.decide(path, signals)


def pick_threshold(config: Any) -> dict:
    """Certify-side threshold selection; placeholder when calibrating is impossible."""
    fn = _resolve()["certify"]["pick_threshold"] or stubs.stub_pick_threshold
    try:
        result = fn(
            conf=[],
            correct=[],
            target_err=float(getattr(config, "target_error", 0.05)),
            delta=float(getattr(config, "delta", 0.1)),
            min_accepted=int(getattr(config, "min_accepted", 30)),
        )
    except TypeError:
        result = stubs.stub_pick_threshold()
    return result if isinstance(result, dict) else {}


def batch_rule(verdicts: list[dict], k: int) -> str:
    fn = _resolve()["certify"]["batch_rule"] or stubs.stub_batch_rule
    return fn(verdicts, k)


def make_receipt(segment: dict, signals: dict, decisions: dict, verdict: dict,
                 config: dict) -> dict:
    fn = _resolve()["receipts"]["make_receipt"] or stubs.stub_make_receipt
    return fn(segment, signals, decisions, verdict, config)


def verify_receipt(receipt: dict, recompute_fn) -> dict:
    fn = _resolve()["receipts"]["verify_receipt"] or stubs.stub_verify_receipt
    return fn(receipt, recompute_fn)


def risk_coverage(points: list[dict]) -> list[dict]:
    """(conf, correct) pairs -> risk/coverage points for /stats/risk_coverage."""
    fn = _resolve()["certify"]["risk_coverage"]
    confs = [p["conf"] for p in points]
    corrects = [int(p["correct"]) for p in points]
    if fn is not None:
        try:
            return fn(confs, corrects)
        except TypeError:
            return fn(points)
    return []


# --------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------
def thresholds_snapshot(config: Any) -> dict:
    """The thresholds a receipt records (also the ones re-verify replays with)."""
    picked = pick_threshold(config)
    answer_threshold = picked.get("threshold")
    if answer_threshold is None:
        answer_threshold = stubs.STUB_THRESHOLD  # nobody calibrated yet
    return {
        "blur_var_min": float(config.blur_var_min),
        "brightness_min": float(config.brightness_min),
        "over_exposed_frac_max": float(config.over_exposed_frac_max),
        "motion_max": float(config.motion_max),
        "hands_visible_frac_min": float(config.hands_visible_frac_min),
        "answer_threshold": float(answer_threshold),
        "target_error": float(config.target_error),
        "delta": float(config.delta),
        "min_accepted": int(config.min_accepted),
        "float_decimals": int(config.float_decimals),
    }


def config_block(config: Any) -> dict:
    return {
        "thresholds": thresholds_snapshot(config),
        "pipeline_version": str(config.pipeline_version),
        "versions": impl_versions(),
    }


def process_clip(path: str | Path, worker_id: str, site_id: str,
                 config: Any) -> list[dict]:
    """Turn one uploaded clip into Segment dicts (stub: a single segment)."""
    out_dir = Path(config.segments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = split_clip(str(path), float(config.segment_seconds), out_dir)
    for seg in segments:
        seg.setdefault("source_clip", str(path))
        seg["worker_id"] = worker_id
        seg["site_id"] = site_id
    return segments


def make_verdict(segment: dict, signals: dict, decisions: dict,
                 thresholds: dict, decimals: int = 4) -> dict:
    """L2 per-segment verdict: hard reasons reject, soft reasons review."""
    t = thresholds
    threshold = float(t["answer_threshold"])
    reasons: set[str] = set()

    if not bool(signals.get("decodable", False)):
        reasons.add("UNDECODABLE")
    if float(signals.get("blur_var", 0.0)) < float(t["blur_var_min"]):
        reasons.add("BLUR")
    if float(signals.get("brightness_mean", 0.0)) < float(t["brightness_min"]):
        reasons.add("TOO_DARK")
    if float(signals.get("over_exposed_frac", 0.0)) > float(t["over_exposed_frac_max"]):
        reasons.add("OVEREXPOSED")
    if float(signals.get("motion_mean", 0.0)) > float(t["motion_max"]):
        reasons.add("SHAKY")
    hands_frac = float(signals.get("hands_visible_frac", 0.0))
    hands_p = float((decisions.get("hands_ok") or {}).get("p_yes", 0.0))
    if hands_frac < float(t["hands_visible_frac_min"]) or hands_p < threshold:
        reasons.add("HANDS_NOT_VISIBLE")
    if float((decisions.get("active_work") or {}).get("p_yes", 0.0)) < threshold:
        reasons.add("NO_ACTIVE_WORK")
    if signals.get("dup_of"):
        reasons.add("DUPLICATE")
    step = decisions.get("step") or {}
    if not step.get("choice"):
        reasons.add("WRONG_STEP")
    if float(step.get("answer_conf", 0.0)) < threshold:
        reasons.add("LOW_CONFIDENCE")

    if reasons & HARD_REASONS:
        verdict = "reject"
    elif reasons:
        verdict = "review"
    else:
        verdict = "accept"

    return {
        "segment_id": segment["segment_id"],
        "verdict": verdict,
        "reasons": [c for c in REASON_ORDER if c in reasons],
        "answer_conf": round(float(step.get("answer_conf", 0.0)), decimals),
        "threshold": round(threshold, decimals),
    }


def run_batch(segments: list[dict], config: Any) -> tuple[dict, list[dict]]:
    """Process a batch of segments per SPEC and return (batch, receipts).

    Stages are strictly per segment (batch size 1) so receipts do not depend on
    how the segments were grouped; phash duplicates are detected in input order.
    """
    if not segments:
        raise ValueError("run_batch needs at least one segment")
    decimals = int(config.float_decimals)
    block = config_block(config)
    phash_index: dict[str, str] = {}
    receipts: list[dict] = []

    for seg in segments:
        signals = compute_signals(seg["path"], config)
        signals = _round_floats(signals, decimals)
        phash = str(signals.get("phash") or "")
        if phash and phash in phash_index:
            signals["dup_of"] = phash_index[phash]  # first occurrence wins
        elif phash:
            phash_index[phash] = seg["segment_id"]
        decisions = _round_floats(decide(seg["path"], signals, config), decimals)
        verdict = make_verdict(seg, signals, decisions, block["thresholds"], decimals)
        receipts.append(make_receipt(seg, signals, decisions, verdict, block))

    verdicts = [r["verdict"] for r in receipts]
    segment_ids = [s["segment_id"] for s in segments]
    site_id = segments[0].get("site_id", "")
    batch_id = "b" + hashlib.sha256(
        "|".join([site_id, *segment_ids, str(config.batch_reject_k)])
        .encode("utf-8")).hexdigest()[:16]
    batch = {
        "batch_id": batch_id,
        "site_id": site_id,
        "segment_ids": segment_ids,
        "verdicts": verdicts,
        "batch_decision": batch_rule(verdicts, int(config.batch_reject_k)),
        "rule": BATCH_RULE_TEXT,
        "k": int(config.batch_reject_k),
        "n": len(verdicts),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return batch, receipts


def reverify_receipt(receipt: dict, segment: dict, config: Any) -> dict:
    """Re-run one stored segment with the receipt's stored config; compare hashes."""
    block = receipt.get("config") or config_block(config)
    thresholds = block.get("thresholds") or thresholds_snapshot(config)
    decimals = int(thresholds.get("float_decimals", config.float_decimals))
    segment = dict(segment)
    path = Path(segment["path"])
    if not path.is_file():
        return {"match": False, "expected": receipt.get("receipt_sha256"),
                "actual": None, "diff": ["<segment file missing>"]}
    # hash the bytes we actually re-read, per the SPEC "which bytes are hashed" rule
    segment["sha256"] = stubs.file_sha256(path)
    signals = _round_floats(compute_signals(segment["path"], config), decimals)
    decisions = _round_floats(decide(segment["path"], signals, config), decimals)
    verdict = make_verdict(segment, signals, decisions, thresholds, decimals)
    actual = make_receipt(segment, signals, decisions, verdict, block)
    return verify_receipt(receipt, lambda _stored: actual)
