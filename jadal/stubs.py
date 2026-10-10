"""Deterministic stub implementations for L0/L1/segmenter.

Selected ONLY when the real modules are absent AND the explicit env flag
``JADAL_USE_STUBS=1`` is set (see ``jadal/pipeline.py``). Values are derived
from the file's SHA-256, never from file contents -- there is no file-content
backdoor: any bytes hash to some signals, and no bytes can choose them.

Tests that need exact signal/decision values use the programmatic override
registry below (``set_test_override``), which also requires the flag.

NOT DONE (stubs only -- real models arrive with their modules):
- signals: no OpenCV/mediapipe frame reading; values are hash-derived fakes.
- decision: no EmbeddingGemma/Laya model; probs are fixed fakes.
- segmenter: no ffmpeg cutting; the whole clip is one segment.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

STUB_THRESHOLD = 0.9  # placeholder until pick_threshold has calibration data


def stubs_enabled() -> bool:
    return os.environ.get("JADAL_USE_STUBS") == "1"


# --------------------------------------------------------------------------
# test override registry (programmatic only; requires the flag)
# --------------------------------------------------------------------------
_TEST_OVERRIDES: dict[str, dict[str, Any]] = {}


def set_test_override(path: str | Path, *, signals: dict | None = None,
                      decisions: dict | None = None) -> None:
    """Register exact stub outputs for one path. Tests only."""
    if not stubs_enabled():
        raise RuntimeError("test overrides require JADAL_USE_STUBS=1")
    _TEST_OVERRIDES[str(Path(path).resolve())] = {
        "signals": dict(signals or {}),
        "decisions": dict(decisions or {}),
    }


def clear_test_overrides() -> None:
    _TEST_OVERRIDES.clear()


def _override(path: str | Path) -> dict[str, Any]:
    if stubs_enabled():
        return _TEST_OVERRIDES.get(str(Path(path).resolve()), {})
    return {}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _round_floats(obj: Any, decimals: int) -> Any:
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------
# segmenter stub
# --------------------------------------------------------------------------
def stub_split_clip(path: str, seg_seconds: float, out_dir: str | Path | None = None) -> list[dict]:
    """One segment spanning the whole clip (no real cutting in the stub)."""
    sha = file_sha256(path)
    return [{
        "segment_id": f"seg-{sha[:12]}-0000",
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
    """Deterministic fake L0 signals derived from the file's SHA-256."""
    decimals = int(getattr(config, "float_decimals", 4))
    try:
        sha = file_sha256(path)
    except OSError:
        sha = "0" * 64
    h = int(sha[:16], 16)
    signals = {
        "decodable": bool(h % 31 != 0),                  # ~97% decodable
        "blur_var": float(60 + (h % 140)),               # 60 .. 199
        "brightness_mean": 90.0,
        "under_exposed_frac": 0.0,
        "over_exposed_frac": 0.02,
        "motion_mean": float(h % 15),                    # 0 .. 14
        "hands_visible_frac": 0.3 + (h % 70) / 100.0,    # 0.30 .. 0.99
        "hand_count_mode": 2,
        "faces_found": h % 2,
        "faces_blurred": h % 2,
        "phash": hashlib.blake2b(sha.encode("utf-8"), digest_size=8).hexdigest(),
        "dup_of": None,
    }
    signals.update(_override(path).get("signals", {}))
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
        decisions = {
            "hands_ok": {"p_yes": 0.97},
            "active_work": {"p_yes": 0.93},
            "step": {"choice": "wash", "probs": {"wash": 0.93}, "answer_conf": 0.93},
            "model_versions": {"stub": "0.1.0"},
        }
        decisions.update(_override(path).get("decisions", {}))
        return _round_floats(decisions, decimals)
