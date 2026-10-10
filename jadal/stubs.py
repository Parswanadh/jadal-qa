"""Deterministic stub implementations for the modules owned by other branches.

These stand in for ``jadal/signals.py`` (L0), ``jadal/decision.py`` (L1),
``jadal/segmenter.py``, ``jadal/certify.py`` (L2) and ``jadal/receipts.py`` until
those branches merge. ``jadal/pipeline.py`` prefers the real modules whenever they
exist and falls back to these.

Determinism: every value is a pure function of the file bytes. For test clips the
bytes may be ``JADALFAKE1\\n<json>`` -- a header plus a JSON blob that overrides any
signal or decision field, so tests can force exact values without decoding video.

NOT DONE (stubs only -- real models arrive with their modules):
- signals: no OpenCV/mediapipe frame reading; values are hash-derived fakes.
- decision: no EmbeddingGemma/Laya model; probs are fixed fakes.
- certify: no calibration; ``pick_threshold`` returns a fixed placeholder threshold
  (0.9); ``batch_rule`` implements the SPEC rule but not risk_coverage/ece.
- segmenter: no ffmpeg cutting; the whole clip is one segment.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

FAKE_MAGIC = "JADALFAKE1\n"

# Passing L0 values (matching jadal/config.py placeholder thresholds).
_DEFAULT_SIGNALS: dict[str, Any] = {
    "decodable": True,
    "blur_var": 150.0,
    "brightness_mean": 90.0,
    "under_exposed_frac": 0.0,
    "over_exposed_frac": 0.02,
    "motion_mean": 2.0,
    "hands_visible_frac": 0.8,
    "hand_count_mode": 2,
    "faces_found": 0,
    "faces_blurred": 0,
    "phash": "",
    "dup_of": None,
}

_DEFAULT_DECISIONS: dict[str, Any] = {
    "hands_ok": {"p_yes": 0.97},
    "active_work": {"p_yes": 0.93},
    "step": {"choice": "wash", "probs": {"wash": 0.93}, "answer_conf": 0.93},
    "model_versions": {"stub": "0.1.0"},
}

STUB_THRESHOLD = 0.9  # placeholder until pick_threshold has calibration data


def _read_payload(path: str | Path) -> dict[str, Any]:
    """Parse a ``JADALFAKE1`` test payload; return {} for any other file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    if not text.startswith(FAKE_MAGIC):
        return {}
    try:
        payload = json.loads(text[len(FAKE_MAGIC):])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _round(value: Any, decimals: int) -> Any:
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, list):
        return [_round(v, decimals) for v in value]
    if isinstance(value, dict):
        return {k: _round(v, decimals) for k, v in value.items()}
    return value


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _round_floats(obj: dict, decimals: int) -> dict:
    return _round(obj, decimals)  # type: ignore[return-value]


# --------------------------------------------------------------------------
# segmenter stub
# --------------------------------------------------------------------------
def stub_split_clip(path: str, seg_seconds: float, out_dir: str | Path | None = None) -> list[dict]:
    """One segment spanning the whole clip (no real cutting in the stub)."""
    sha = file_sha256(path)
    segment_id = f"seg-{sha[:12]}-0000"
    return [{
        "segment_id": segment_id,
        "source_clip": str(path),
        "worker_id": "",
        "site_id": "",
        "start_s": 0.0,
        "end_s": float(seg_seconds),
        "path": str(path),  # the stub reads the ORIGINAL file
        "sha256": sha,
    }]


# --------------------------------------------------------------------------
# signals stub (L0)
# --------------------------------------------------------------------------
def stub_compute_signals(path: str | Path, config: Any = None) -> dict:
    """Deterministic fake L0 signals derived from the file bytes."""
    decimals = int(getattr(config, "float_decimals", 4))
    payload = _read_payload(path)
    try:
        sha = file_sha256(path)
    except OSError:
        sha = "0" * 64
    h = int.from_bytes(bytes.fromhex(sha[:16]), "big") if sha != "0" * 64 else 0
    signals = dict(_DEFAULT_SIGNALS)
    signals["phash"] = hashlib.blake2b(sha.encode("utf-8"), digest_size=8).hexdigest()
    signals["blur_var"] = float(60 + (h % 140))               # 60 .. 199
    signals["motion_mean"] = float(h % 15)                    # 0 .. 14
    signals["hands_visible_frac"] = 0.3 + (h % 70) / 100.0    # 0.30 .. 0.99
    signals["decodable"] = bool(h % 31 != 0)                  # ~97% decodable
    signals["faces_found"] = h % 2
    signals["faces_blurred"] = signals["faces_found"]
    signals.update(payload.get("signals", {}))
    return _round_floats(signals, decimals)


# --------------------------------------------------------------------------
# decision stub (L1)
# --------------------------------------------------------------------------
class StubDecider:
    """Deterministic fake L1 decisions (no model weights involved)."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def decide(self, path: str, signals: dict) -> dict:
        decimals = int(getattr(self.config, "float_decimals", 4))
        payload = _read_payload(path)
        decisions = dict(_DEFAULT_DECISIONS)
        decisions.update(payload.get("decisions", {}))
        return _round_floats(decisions, decimals)


# --------------------------------------------------------------------------
# certify stubs (L2)
# --------------------------------------------------------------------------
def stub_pick_threshold(
    conf: list[float] | None = None,
    correct: list[int] | None = None,
    target_err: float = 0.05,
    delta: float = 0.1,
    min_accepted: int = 30,
) -> dict:
    """No calibration data in the stub: fixed placeholder threshold."""
    return {
        "threshold": STUB_THRESHOLD,
        "accepted": 0,
        "errors": 0,
        "upper_bound": None,
        "coverage": 0.0,
    }


def stub_batch_rule(verdicts: list[dict], k: int) -> str:
    """SPEC rule: reject if rejected_count > k (fixed rule, no statistical guarantee)."""
    rejected = sum(1 for v in verdicts if v.get("verdict") == "reject")
    return "reject" if rejected > k else "accept"


# --------------------------------------------------------------------------
# receipts stubs
# --------------------------------------------------------------------------
def stub_canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def stub_receipt_hash(receipt: dict) -> str:
    payload = {k: v for k, v in receipt.items()
               if k not in ("receipt_sha256", "receipt_id", "created_utc")}
    return hashlib.sha256(stub_canonical_json(payload)).hexdigest()


def stub_make_receipt(segment: dict, signals: dict, decisions: dict, verdict: dict,
                      config: dict) -> dict:
    receipt = {
        "segment_sha256": segment["sha256"],
        "signals": signals,
        "decisions": decisions,
        "verdict": verdict,
        "config": config,
    }
    receipt["receipt_sha256"] = stub_receipt_hash(receipt)
    receipt["receipt_id"] = receipt["receipt_sha256"][:16]
    receipt["created_utc"] = datetime.now(timezone.utc).isoformat()
    return receipt


def stub_verify_receipt(receipt: dict, recompute_fn: Callable[[dict], dict]) -> dict:
    actual = recompute_fn(receipt)
    return {
        "match": actual.get("receipt_sha256") == receipt.get("receipt_sha256"),
        "expected": receipt.get("receipt_sha256"),
        "actual": actual.get("receipt_sha256"),
        "diff": receipt_diff(receipt, actual),
    }


def receipt_diff(stored: dict, recomputed: dict) -> list[str]:
    """Dotted paths where the two receipts differ (excluded fields aside)."""
    skip = ("receipt_sha256", "receipt_id", "created_utc")

    def walk(a: Any, b: Any, prefix: str) -> list[str]:
        if isinstance(a, dict) and isinstance(b, dict):
            out: list[str] = []
            for key in sorted(set(a) | set(b)):
                if prefix == "" and key in skip:
                    continue
                out += walk(a.get(key, "<MISSING>"), b.get(key, "<MISSING>"),
                            f"{prefix}.{key}" if prefix else str(key))
            return out
        if isinstance(a, list) and isinstance(b, list):
            out = []
            for i in range(max(len(a), len(b))):
                av = a[i] if i < len(a) else "<MISSING>"
                bv = b[i] if i < len(b) else "<MISSING>"
                out += walk(av, bv, f"{prefix}[{i}]")
            return out
        return [] if a == b else [prefix]

    return walk(stored, recomputed, "")
