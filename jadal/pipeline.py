"""Batch pipeline for the Jadal QA service (GitHub issue #3).

Order per docs/SPEC.md: segmenter -> L0 signals -> L1 decisions -> L2 verdict
(with a threshold from certify) -> batch rule -> receipt. Every stage is per
segment (batch size 1), so a segment hashes identically alone or inside a batch
of five.

Module resolution: the real ``jadal.signals`` / ``jadal.decision`` /
``jadal.segmenter`` are preferred whenever importable. The deterministic stubs
in ``jadal/stubs.py`` are used ONLY when ``JADAL_USE_STUBS=1`` is set (tests);
otherwise a clear ``RuntimeError`` is raised. ``jadal.certify`` and
``jadal.receipts`` are always the real modules.

NOT DONE:
- L0/L1/segmenter are stubs (hash-derived fakes, whole clip = one segment)
  until the real modules merge; set JADAL_USE_STUBS=1 to run on stubs. When
  jadal.signals / jadal.decision / jadal.segmenter are importable they are
  used automatically (no flag needed).
- ``pick_threshold`` has no calibration data, so the answer threshold falls back
  to a recorded placeholder (0.9) until eval data exists.
- Verdict reason rules live here (L2 certify API is thresholding + batch rule).
- The real jadal.decision module does not exist yet (feat/decider has no
  Decider); L1 stays on stubs until it lands.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jadal import certify as certify_mod
from jadal import receipts as receipts_mod
from jadal import stubs
from jadal.config import Config

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

# Config dataclass fields that are plain scalars (safe to restore from a receipt).
_SCALAR_CONFIG_FIELDS = frozenset(
    f.name for f in dataclass_fields(Config)
    if f.name not in ("bucket_dir", "segments_dir", "blurred_dir", "db_path",
                      "voicebank_dir", "sop_dir", "web_dir", "pico_port")
)


def _load_stage(module_name: str, attr: str, stub: Any, kind: str) -> Any:
    """Prefer the real module; fall back to the stub only under the flag."""
    try:
        mod = __import__(f"jadal.{module_name}", fromlist=[attr])
    except ImportError:
        mod = None
    fn = getattr(mod, attr, None) if mod is not None else None
    if fn is not None:
        return fn
    if stubs.stubs_enabled():
        return stub
    raise RuntimeError(
        f"no {kind} implementation: jadal.{module_name}.{attr} is not importable "
        f"and JADAL_USE_STUBS is not '1'"
    )


def _compute_signals_fn():
    return _load_stage("signals", "compute_signals",
                       stubs.stub_compute_signals, "L0 signals")


def _decider_class():
    return _load_stage("decision", "Decider", stubs.StubDecider, "L1 decision")


def _split_clip_fn():
    return _load_stage("segmenter", "split_clip",
                       stubs.stub_split_clip, "segmenter")


def impl_versions(config: Any = None) -> dict[str, str]:
    """Which implementation each stage is running (goes into every receipt).

    Beyond module names this records the L0 backend details reported by the
    real signals module (``backend_versions()``, else ``backend_info()``) and
    the sha256 of the face/hand model weight files when they exist on disk.
    Everything is best-effort: a missing module or file only means a missing
    key, never an exception. Weight hashes are cached per process.
    """
    versions = {"python": platform.python_version()}
    for name, attr, stub_tag in (
            ("signals", "compute_signals", "stubs.signals"),
            ("decision", "Decider", "stubs.decision"),
            ("segmenter", "split_clip", "stubs.segmenter")):
        try:
            mod = __import__(f"jadal.{name}", fromlist=[attr])
            real = getattr(mod, attr, None) is not None
        except ImportError:
            real = False
        versions[name] = f"jadal.{name}" if real else stub_tag
    versions["certify"] = "jadal.certify"
    versions["receipts"] = "jadal.receipts"
    try:
        from jadal import signals as _sig_mod
    except ImportError:
        _sig_mod = None  # type: ignore[assignment]
    if _sig_mod is not None:
        fn = getattr(_sig_mod, "backend_versions", None)
        if not callable(fn):
            fn = getattr(_sig_mod, "backend_info", None)
        if callable(fn):
            try:
                try:
                    info = fn(config) if config is not None else fn()
                except TypeError:
                    info = fn()
            except Exception:
                info = None
            if isinstance(info, dict):
                versions["signals_backend"] = json.dumps(
                    info, sort_keys=True, separators=(",", ":"),
                    default=str)
            elif info is not None:
                versions["signals_backend"] = str(info)
    if config is not None:
        for key in ("face_detector_task", "hand_landmarker_task"):
            token = _weight_token(getattr(config, key, None))
            if token is not None:
                versions[key] = token
    return versions


# sha256 of model weight files, cached per process (files are small).
_WEIGHT_HASH_CACHE: dict[str, str] = {}


def _weight_hash(path: Path) -> str | None:
    try:
        resolved = str(path.resolve())
    except OSError:
        return None
    if resolved in _WEIGHT_HASH_CACHE:
        return _WEIGHT_HASH_CACHE[resolved]
    try:
        h = hashlib.sha256()
        with open(resolved, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    digest = h.hexdigest()
    _WEIGHT_HASH_CACHE[resolved] = digest
    return digest


def _weight_token(path: Any) -> str | None:
    """`name:sha256:<hex>` for a weight file, else `name:size:<n>`, else None."""
    if path is None:
        return None
    p = Path(path)
    try:
        if not p.is_file():
            return None
        size = p.stat().st_size
    except OSError:
        return None
    digest = _weight_hash(p)
    if digest is not None:
        return f"{p.name}:sha256:{digest}"
    return f"{p.name}:size:{size}"


def ensure_determinism(config: Any) -> None:
    """Pin down RNGs and deterministic kernels (SPEC determinism rules).

    Best-effort and side-effect free on failure: every backend is guarded, so
    a missing torch/numpy never breaks the pipeline. Real model modules own
    their own fixed sampling; this only covers the process-global knobs.
    """
    seed = int(getattr(config, "seed", 1234))
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as _np
        _np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _round_floats(obj: Any, decimals: int) -> Any:
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------
def pick_threshold(config: Any) -> dict:
    """Certify-side threshold selection over (empty) calibration data."""
    return certify_mod.pick_threshold(
        conf=[],
        correct=[],
        target_err=float(getattr(config, "target_error", 0.05)),
        delta=float(getattr(config, "delta", 0.1)),
        min_accepted=int(getattr(config, "min_accepted", 30)),
    )


def thresholds_snapshot(config: Any) -> dict:
    """The thresholds a receipt records (also the ones re-verify replays with)."""
    picked = pick_threshold(config)
    answer_threshold = picked.get("threshold")
    if answer_threshold is None:
        answer_threshold = stubs.STUB_THRESHOLD  # nobody calibrated yet
    snapshot = {
        "batch_size": int(config.batch_size),
        "batch_reject_k": int(config.batch_reject_k),
        "segment_seconds": float(config.segment_seconds),
        "seed": int(config.seed),
        "float_decimals": int(config.float_decimals),
        "blur_var_min": float(config.blur_var_min),
        "brightness_min": float(config.brightness_min),
        "over_exposed_frac_max": float(config.over_exposed_frac_max),
        "motion_max": float(config.motion_max),
        "hands_visible_frac_min": float(config.hands_visible_frac_min),
        "target_error": float(config.target_error),
        "delta": float(config.delta),
        "min_accepted": int(config.min_accepted),
        "answer_threshold": float(answer_threshold),
    }
    return snapshot


def config_block(config: Any) -> dict:
    return {
        "thresholds": thresholds_snapshot(config),
        "pipeline_version": str(config.pipeline_version),
        "versions": impl_versions(config),
    }


def stored_config(block: Any) -> Any:
    """Rebuild a Config from the scalar fields of a receipt's stored block."""
    if not isinstance(block, dict):
        return block
    thresholds = block.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        return block
    values = {k: v for k, v in thresholds.items() if k in _SCALAR_CONFIG_FIELDS}
    try:
        import dataclasses
        return dataclasses.replace(Config(), **values)
    except (TypeError, ValueError):
        return block


def process_clip(path: str | Path, worker_id: str, site_id: str,
                 config: Any) -> list[dict]:
    """Turn one uploaded clip into Segment dicts (stub: a single segment)."""
    out_dir = Path(config.segments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = _split_clip_fn()(str(path), float(config.segment_seconds), out_dir)
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
    ensure_determinism(config)
    decimals = int(config.float_decimals)
    block = config_block(config)
    compute_signals = _compute_signals_fn()
    decider_cls = _decider_class()
    try:
        decider = decider_cls(config)
    except TypeError:
        decider = decider_cls()
    phash_index: dict[str, str] = {}
    receipts: list[dict] = []

    for seg in segments:
        signals = _round_floats(compute_signals(seg["path"], config), decimals)
        phash = str(signals.get("phash") or "")
        if phash and phash in phash_index:
            signals["dup_of"] = phash_index[phash]  # first occurrence wins
        elif phash:
            phash_index[phash] = seg["segment_id"]
        decisions = _round_floats(decider.decide(seg["path"], signals), decimals)
        verdict = make_verdict(seg, signals, decisions, block["thresholds"], decimals)
        receipts.append(receipts_mod.make_receipt(
            seg, signals, decisions, verdict, block))

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
        "batch_decision": certify_mod.batch_rule(verdicts, int(config.batch_reject_k)),
        "rule": BATCH_RULE_TEXT,
        "k": int(config.batch_reject_k),
        "n": len(verdicts),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return batch, receipts


def reverify_receipt(receipt: dict, segment: dict, config: Any) -> dict:
    """Re-run one stored segment with the receipt's STORED config; compare hashes.

    The L0/L1 stages get a Config rebuilt from the stored thresholds (never the
    live object), the verdict threshold comes from the stored verdict, and the
    config block passed to make_receipt is the stored block verbatim.
    """
    block = receipt.get("config")
    if not isinstance(block, dict):
        block = config_block(config)
    thresholds = block.get("thresholds") if isinstance(block, dict) else None
    if not isinstance(thresholds, dict):
        thresholds = thresholds_snapshot(config)
    replay_config = stored_config(block if isinstance(block, dict) else None) or config
    if not isinstance(replay_config, Config):
        replay_config = config
    answer_threshold = (receipt.get("verdict") or {}).get("threshold")
    if answer_threshold is None:
        answer_threshold = thresholds.get("answer_threshold",
                                          stubs.STUB_THRESHOLD)
    replay_thresholds = dict(thresholds)
    replay_thresholds["answer_threshold"] = float(answer_threshold)
    decimals = int(replay_thresholds.get("float_decimals",
                                         getattr(config, "float_decimals", 4)))

    segment = dict(segment)
    path = Path(segment["path"])
    if not path.is_file():
        return {"match": False, "expected": receipt.get("receipt_sha256"),
                "actual": None, "diff": ["<segment file missing>"]}
    # hash the bytes we actually re-read, per the SPEC "which bytes" rule
    segment["sha256"] = stubs.file_sha256(path)
    compute_signals = _compute_signals_fn()
    decider_cls = _decider_class()
    try:
        decider = decider_cls(replay_config)
    except TypeError:
        decider = decider_cls()
    signals = _round_floats(compute_signals(segment["path"], replay_config), decimals)
    decisions = _round_floats(decider.decide(segment["path"], signals), decimals)
    verdict = make_verdict(segment, signals, decisions, replay_thresholds, decimals)
    actual = receipts_mod.make_receipt(segment, signals, decisions, verdict, block)
    result = receipts_mod.verify_receipt(receipt, lambda _stored: actual)
    result["version_match"], result["version_diff"] = compare_versions(
        block, impl_versions(config))
    return result


def compare_versions(stored_block: Any,
                     current_versions: dict) -> tuple[bool, list[str]]:
    """Stored receipt versions vs the running code: (match, differing keys).

    A re-verify mismatch after a module upgrade (stubs -> real signals, new
    weights) is expected; the diff names the cause instead of leaving a bare
    hash inequality.
    """
    stored_versions = (stored_block or {}).get("versions") if isinstance(
        stored_block, dict) else None
    if not isinstance(stored_versions, dict):
        return False, ["<no stored versions>"]
    keys = sorted(set(stored_versions) | set(current_versions))
    diff = [k for k in keys
            if stored_versions.get(k) != current_versions.get(k)]
    return (not diff), diff


def risk_coverage(points: list[dict]) -> list[dict]:
    """(conf, correct) pairs -> risk/coverage points for /stats/risk_coverage."""
    confs = [p["conf"] for p in points]
    corrects = [int(p["correct"]) for p in points]
    return certify_mod.risk_coverage(confs, corrects)


def risk_coverage_from_eval(results_dir: str | Path | None = None) -> list[dict]:
    """Risk/coverage points from the real eval results JSON, else [].

    Reads ``eval/results/calibration.json`` (written by eval/calibration_report.py)
    and returns its validated ``risk_coverage`` list. Returns [] when the file
    is absent or malformed -- that is the honest "no measured data" state, and
    callers must surface it as such instead of inventing numbers.
    """
    from jadal.config import ROOT
    path = Path(results_dir) if results_dir else ROOT / "eval" / "results" / "calibration.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = payload.get("risk_coverage") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    clean = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            clean.append({
                "threshold": float(row["threshold"]),
                "coverage": float(row["coverage"]),
                "error": float(row["error"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return clean


def sop_gaps_from_verdicts(batches: list[dict]) -> list[dict]:
    """Reason-code counts over stored batch verdicts (real /stats/sop_gaps data).

    Returns ``[{reason_code, count, share}]`` sorted by count desc, then code.
    ``share`` is count / total verdicts (0.0 when there are no verdicts).
    """
    counts: dict[str, int] = {}
    total = 0
    for batch in batches:
        for verdict in batch.get("verdicts") or []:
            total += 1
            for code in verdict.get("reasons") or []:
                counts[str(code)] = counts.get(str(code), 0) + 1
    rows = [{"reason_code": code, "count": n,
             "share": round(n / total, 4) if total else 0.0}
            for code, n in counts.items()]
    rows.sort(key=lambda r: (-r["count"], r["reason_code"]))
    return rows


__all__ = [
    "REASON_ORDER", "HARD_REASONS", "BATCH_RULE_TEXT",
    "impl_versions", "ensure_determinism", "pick_threshold", "thresholds_snapshot", "config_block",
    "stored_config", "process_clip", "make_verdict", "run_batch",
    "reverify_receipt", "compare_versions", "risk_coverage",
    "risk_coverage_from_eval", "sop_gaps_from_verdicts",
]
