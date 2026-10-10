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
- ``pick_threshold`` has no calibration data, so the answer threshold falls back
  to a recorded placeholder (0.9) until eval data exists.
- Verdict reason rules live here (L2 certify API is thresholding + batch rule).
- Funnel routing (``jadal.funnel.route``) is a runtime seam only: when that
  module is importable it decides the verdict (with evidence); the funnel
  thresholds/logic themselves are owned by another worker.
- ``receipt_problems`` treats the OpenCV colour proxy as a failure: the edge
  path may legitimately run on the proxy (BASELINE amendment B9), but this
  integration demo proves the full mediapipe path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jadal import certify as certify_mod
from jadal import receipts as receipts_mod
from jadal import stubs
from jadal.config import Config

log = logging.getLogger("jadal.pipeline")

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


# One Decider instance per (class, float_decimals) per process. The EG2
# backbone load is expensive (multi-GB weights onto CUDA), so batches and
# re-verifies share the instance instead of reloading per call. Keyed by the
# class OBJECT so tests with injected fakes never share instances with
# production (or each other); deciders are stateless across segments.
_DECIDER_CACHE: dict[tuple, Any] = {}


def _get_decider(decider_cls: Any, config: Any) -> Any:
    key = (decider_cls, int(getattr(config, "float_decimals", 4)))
    decider = _DECIDER_CACHE.get(key)
    if decider is None:
        try:
            decider = decider_cls(config)
        except TypeError:
            decider = decider_cls()
        _DECIDER_CACHE[key] = decider
    return decider


def _split_clip_fn():
    return _load_stage("segmenter", "split_clip",
                       stubs.stub_split_clip, "segmenter")


def _git_file_sha(rel_path: str) -> str | None:
    """Last-commit sha touching ``rel_path`` under the repo root (best-effort)."""
    try:
        from jadal.config import ROOT
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", rel_path],
            capture_output=True, text=True, timeout=15)
        sha = out.stdout.strip().splitlines()
        return sha[0] if sha and len(sha[0]) == 40 else None
    except Exception:
        return None


def _repo_head_sha() -> str | None:
    try:
        from jadal.config import ROOT
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15)
        sha = out.stdout.strip().splitlines()
        return sha[0] if sha and len(sha[0]) == 40 else None
    except Exception:
        return None


def _lib_version(module_name: str) -> str | None:
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return None


def _device_info() -> tuple[str, str | None]:
    """(device, gpu_name): device is 'cuda' or 'cpu' (best-effort, never raises)."""
    try:
        import torch
        try:
            if torch.cuda.is_available():
                try:
                    return "cuda", str(torch.cuda.get_device_name(0))
                except Exception:
                    return "cuda", "unknown"
        except Exception:
            pass
        return "cpu", None
    except Exception:
        return "cpu", None


def impl_versions(config: Any = None) -> dict[str, str]:
    """Which implementation each stage is running (goes into every receipt).

    Beyond module names this records, per baseline B2/B3: the last-commit git
    sha of each implementation file (signals, decision, segmenter, funnel,
    certify, receipts) plus the repo HEAD; the compute device (cpu/cuda +
    GPU name); torch/opencv/mediapipe/transformers/numpy versions; the L0
    backend details reported by the real signals module (``backend_versions()``,
    else ``backend_info()``); the sha256 of the face/hand task files; the
    sha256 of the EmbeddingGemma-2 backbone weights and the decider head
    files when present; python, platform and machine architecture (so
    cross-architecture determinism is explicitly NOT claimed).
    Everything is best-effort: a missing module or file only means a missing
    key, never an exception. Weight hashes are cached per process (the 1.4 G
    backbone hashes once, then reuses the digest).
    """
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "arch": platform.machine(),
    }
    head_sha = _repo_head_sha()
    if head_sha is not None:
        versions["repo_git_sha"] = head_sha
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
    versions["funnel"] = "jadal.funnel"
    try:
        from jadal import funnel as _funnel_mod
        if getattr(_funnel_mod, "route", None) is None:
            versions["funnel"] = "jadal.funnel(no-route)"
    except ImportError:
        versions["funnel"] = "missing"
    for name in ("signals", "decision", "segmenter", "funnel",
                 "certify", "receipts"):
        sha = _git_file_sha(f"jadal/{name}.py")
        if sha is not None:
            versions[f"{name}_git_sha"] = sha
    device, gpu_name = _device_info()
    versions["device"] = device
    if gpu_name is not None:
        versions["gpu_name"] = gpu_name
    versions["torch_version"] = _lib_version("torch") or "missing"
    versions["opencv_version"] = _lib_version("cv2") or "missing"
    versions["mediapipe_version"] = _lib_version("mediapipe") or "missing"
    versions["transformers_version"] = _lib_version("transformers") or "missing"
    versions["numpy_version"] = _lib_version("numpy") or "missing"
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
    try:
        from jadal import decision as _dec_mod
        from jadal.config import ROOT as _ROOT
        eg2 = Path(getattr(_dec_mod, "EG2_DIR", "")) / "model.safetensors"
        token = _weight_token(eg2) if str(eg2) else None
        if token is not None:
            versions["eg2_weights"] = token
        for fname in (getattr(_dec_mod, "HANDS_HEAD", "decider_hands.npz"),
                      getattr(_dec_mod, "WORK_HEAD", "decider_work.npz"),
                      getattr(_dec_mod, "STEP_PROTO",
                              "decider_step_prototypes.npz")):
            token = _weight_token(Path(_ROOT) / "models" / str(fname))
            if token is not None:
                versions[f"decider_{Path(str(fname)).stem}"] = token
    except Exception:
        pass
    return versions


def receipt_problems(receipt: dict) -> list[str]:
    """Stub/silent-fallback problems in one receipt ([] = real, clean).

    Fails a receipt when ``config.versions`` names a stub implementation for
    any stage, when the L0 backend ran without mediapipe (OpenCV colour
    proxy is a recorded fallback, not the real path this demo proves), when
    the decider ran without its head files (0.5 silent defaults), or when
    no frames decoded while L0 claims the segment is decodable. Used by
    ``scripts/demo.sh`` (non-zero exit on any problem) and unit-tested with
    crafted receipts (no models needed).
    """
    problems: list[str] = []
    versions = (receipt.get("config") or {}).get("versions") or {}
    if not isinstance(versions, dict):
        return ["receipt has no config.versions block"]
    for name in ("signals", "decision", "segmenter"):
        value = versions.get(name)
        if not isinstance(value, str) or value.startswith("stubs."):
            problems.append(f"{name}={value!r} names a stub implementation")
    if versions.get("funnel", "jadal.funnel") != "jadal.funnel":
        problems.append(f"funnel={versions.get('funnel')!r} is not the real router")
    backend_raw = versions.get("signals_backend", "")
    try:
        backend = json.loads(backend_raw) if isinstance(backend_raw, str) else {}
    except ValueError:
        backend = {}
    if not isinstance(backend, dict) or backend.get("faces") != "mediapipe" \
            or backend.get("hands") != "mediapipe":
        problems.append(
            "signals backend is not mediapipe "
            f"(faces={backend.get('faces')!r}, hands={backend.get('hands')!r}); "
            "provide models/blaze_face_short_range.tflite and "
            "models/hand_landmarker.task, do not stub silently")
    decisions = receipt.get("decisions") or {}
    model_versions = decisions.get("model_versions") or {}
    heads = model_versions.get("heads", "")
    if "missing" in str(heads):
        problems.append(
            f"decider heads missing ({heads!r}): provide "
            "models/decider_hands.npz and models/decider_work.npz, "
            "do not fall back to 0.5 silently")
    signals = receipt.get("signals") or {}
    if signals.get("decodable") and str(
            model_versions.get("frames_decoded", "1")) == "0":
        problems.append("decider decoded 0 frames on a decodable segment")
    return problems


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


class ThresholdsError(RuntimeError):
    """The funnel thresholds file failed its hash check (tampered/corrupt).

    Raised by :func:`thresholds_snapshot` instead of falling back to other
    thresholds: a hash mismatch must fail closed, never silently degrade the
    router. Callers turn this into review verdicts with mismatch evidence
    (never accepts) and record it in the errors store.
    """


def thresholds_snapshot(config: Any) -> dict:
    """The thresholds a receipt records (also the ones re-verify replays with).

    Starts from the funnel router's own thresholds file (via
    ``jadal.funnel.load_thresholds``, hash-verified by that module) so the
    router can enforce every rule it calibrates (motion_min, min_duration_s,
    accept_conf, calibration gate), with the service Config values overlaid
    on the shared L0 keys. A hash mismatch (tampered or corrupt file) raises
    :class:`ThresholdsError` -- fail closed, never a silent fallback. Only a
    missing file (nothing configured to verify) or a missing funnel module
    falls back to the Config-only snapshot, which the inline verdict path
    uses alone. The file content is stored verbatim, so re-verify replays the
    exact thresholds behind the receipt even if the file later changes.
    """
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
    }
    try:
        from jadal import funnel as _funnel_mod
        load_thresholds = getattr(_funnel_mod, "load_thresholds", None)
        thresholds_hash = getattr(_funnel_mod, "thresholds_hash", None)
        file_thresholds = load_thresholds() if callable(load_thresholds) else None
    except ValueError as exc:
        # Hash mismatch (or otherwise invalid file content): fail closed.
        raise ThresholdsError(
            f"funnel thresholds file failed verification: {exc}") from exc
    except (ImportError, OSError):
        # No funnel module, or no file configured to verify: the documented
        # Config-only fallback.
        file_thresholds = None
        thresholds_hash = None
        log.warning("funnel thresholds unavailable; Config-only snapshot")
    if isinstance(file_thresholds, dict):
        merged = {k: v for k, v in file_thresholds.items() if k != "sha256"}
        merged.update(snapshot)  # service Config wins the shared L0 keys
        snapshot = merged
        if callable(thresholds_hash):
            try:
                snapshot["funnel_thresholds_sha256"] = thresholds_hash(file_thresholds)
            except Exception:
                pass
    picked = pick_threshold(config)
    answer_threshold = picked.get("threshold")
    if answer_threshold is None:
        answer_threshold = stubs.STUB_THRESHOLD  # nobody calibrated yet
    snapshot["answer_threshold"] = float(answer_threshold)
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
    """Turn one uploaded clip into Segment dicts (stub: a single segment).

    An undecodable clip never raises here: when the segmenter fails on the
    bytes (or yields zero segments), one deterministic placeholder segment
    over the whole clip is returned so the batch still closes with an
    UNDECODABLE reject verdict downstream. Only transient I/O (OSError)
    propagates to the caller's retry path.
    """
    out_dir = Path(config.segments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        segments = _split_clip_fn()(str(path), float(config.segment_seconds), out_dir)
    except OSError:
        raise
    except Exception as exc:
        log.warning("segmenter failed for %s (%s); using UNDECODABLE placeholder",
                    path, exc)
        segments = [_undecodable_placeholder(path, config)]
    if not segments:
        log.warning("segmenter yielded zero segments for %s; "
                    "using UNDECODABLE placeholder", path)
        segments = [_undecodable_placeholder(path, config)]
    for seg in segments:
        seg.setdefault("source_clip", str(path))
        seg["worker_id"] = worker_id
        seg["site_id"] = site_id
    return segments


def _undecodable_placeholder(path: str | Path, config: Any) -> dict:
    """One deterministic whole-clip segment for bytes no splitter accepts."""
    sha = stubs.file_sha256(path)  # OSError here propagates (transient I/O)
    return {
        "segment_id": f"seg-{sha[:12]}-0000",
        "source_clip": str(path),
        "worker_id": "",
        "site_id": "",
        "start_s": 0.0,
        "end_s": float(config.segment_seconds),
        "path": str(path),
        "sha256": sha,
    }


# Signals synthesised when the real decoder raises on a segment: zero
# measurements with decodable=False. make_verdict / funnel route both map
# this to a reject with reason UNDECODABLE (dependent hard checks may also
# fire on the zeros; the verdict is reject either way).
_UNDECODABLE_SIGNALS = {
    "decodable": False,
    "blur_var": 0.0,
    "brightness_mean": 0.0,
    "under_exposed_frac": 0.0,
    "over_exposed_frac": 0.0,
    "motion_mean": 0.0,
    "hands_visible_frac": 0.0,
    "hand_count_mode": 0,
    "faces_found": 0,
    "faces_blurred": 0,
    "phash": "",
    "dup_of": None,
}

# Transient decoder I/O is retried this many times with doubling backoff,
# then the error propagates (the service requeues the batch and records the
# error); anything else the decoder raises becomes an UNDECODABLE verdict,
# never a retry.
_SIGNAL_IO_RETRIES = 3
_SIGNAL_IO_BASE_SLEEP_S = 0.02
_SIGNAL_IO_MAX_SLEEP_S = 0.5


def _compute_signals_safe(compute_signals: Any, seg_path: str,
                          config: Any, decimals: int) -> dict:
    """Decode-or-UNDECODABLE: capped retries for transient I/O, synthesis else."""
    attempts = 0
    while True:
        try:
            return _round_floats(compute_signals(seg_path, config), decimals)
        except OSError:
            attempts += 1
            if attempts >= _SIGNAL_IO_RETRIES:
                raise
            time.sleep(min(_SIGNAL_IO_BASE_SLEEP_S * (2.0 ** (attempts - 1)),
                           _SIGNAL_IO_MAX_SLEEP_S))
        except Exception as exc:
            log.warning("decoder failed for %s (%s: %s); verdict UNDECODABLE",
                        seg_path, type(exc).__name__, exc)
            return _round_floats(dict(_UNDECODABLE_SIGNALS), decimals)


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


def _funnel_route_fn() -> Any | None:
    """``jadal.funnel.route`` when that module is importable, else None.

    Resolved at runtime on every call (never cached) so tests can inject a
    fake ``jadal.funnel`` module and production picks the real one up the
    moment it merges -- with zero behaviour change while it is absent.
    """
    try:
        mod = __import__("jadal.funnel", fromlist=["route"])
    except ImportError:
        return None
    fn = getattr(mod, "route", None)
    return fn if callable(fn) else None


def route_verdict(segment: dict, signals: dict, decisions: dict,
                  thresholds: dict, decimals: int = 4) -> dict:
    """L2 verdict through the funnel seam.

    When ``jadal.funnel`` is importable its SPEC ``route(signals, decisions,
    thresholds)`` decides (pure ``{verdict, reasons, evidence, answer_conf,
    queue_priority}``); otherwise the inline :func:`make_verdict` logic runs.
    The funnel path carries the optional ``evidence`` list on the verdict
    (included in the receipt hash when present); the inline path never adds
    the key, so existing receipt hashes are unchanged. A ``route`` exception
    is never swallowed into a verdict: it propagates to the caller's
    failure path (batch requeue + recorded error), because a crashing router
    is a bug, not a measurement.
    """
    fn = _funnel_route_fn()
    if fn is None:
        return make_verdict(segment, signals, decisions, thresholds, decimals)
    routed = fn(signals, decisions, thresholds)  # raises propagate, by design
    verdict = str(routed.get("verdict", "review"))
    if verdict not in ("accept", "reject", "review"):
        verdict = "review"
    reasons = [c for c in REASON_ORDER if c in (routed.get("reasons") or [])]
    try:
        answer_conf = (round(float(routed["answer_conf"]), decimals)
                       if routed.get("answer_conf") is not None else 0.0)
    except (TypeError, ValueError):
        answer_conf = 0.0
    accept_conf = thresholds.get("accept_conf",
                                 thresholds.get("answer_threshold",
                                                stubs.STUB_THRESHOLD))
    try:
        threshold = round(float(accept_conf), decimals)
    except (TypeError, ValueError):
        threshold = round(float(stubs.STUB_THRESHOLD), decimals)
    out: dict[str, Any] = {
        "segment_id": segment["segment_id"],
        "verdict": verdict,
        "reasons": reasons,
        "answer_conf": answer_conf,
        "threshold": threshold,
    }
    if routed.get("evidence") is not None:
        out["evidence"] = routed["evidence"]
    return out


def _thresholds_mismatch_verdict(segment: dict, decimals: int) -> dict:
    """Fail-closed verdict when the thresholds file hash does not match.

    Review (never accept), with evidence naming the cause so a worker can
    dispute it and the batch stays visibly non-accepting.
    """
    return {
        "segment_id": segment["segment_id"],
        "verdict": "review",
        "reasons": ["LOW_CONFIDENCE"],
        "answer_conf": 0.0,
        "threshold": round(float(stubs.STUB_THRESHOLD), decimals),
        "evidence": [{
            "check": "thresholds",
            "value": "hash mismatch",
            "op": "==",
            "threshold": "valid-hash",
            "code": "LOW_CONFIDENCE",
        }],
    }


def run_batch(segments: list[dict], config: Any) -> tuple[dict, list[dict]]:
    """Process a batch of segments per SPEC and return (batch, receipts).

    Stages are strictly per segment (batch size 1) so receipts do not depend on
    how the segments were grouped; phash duplicates are detected in input order.

    A tampered thresholds file fails closed: every segment routes to review
    with 'thresholds hash mismatch' evidence (never an accept, never a
    silent fallback), and the batch still closes visibly.
    """
    if not segments:
        raise ValueError("run_batch needs at least one segment")
    ensure_determinism(config)
    decimals = int(config.float_decimals)
    try:
        block = config_block(config)
    except ThresholdsError as exc:
        log.warning("thresholds verification failed (%s); batch fails closed "
                    "to review", exc)
        return _closed_mismatch_batch(segments, config, decimals, str(exc))
    compute_signals = _compute_signals_fn()
    decider = _get_decider(_decider_class(), config)
    phash_index: dict[str, str] = {}
    receipts: list[dict] = []

    for seg in segments:
        signals = _compute_signals_safe(compute_signals, seg["path"],
                                        config, decimals)
        phash = str(signals.get("phash") or "")
        if phash and phash in phash_index:
            signals["dup_of"] = phash_index[phash]  # first occurrence wins
        elif phash:
            phash_index[phash] = seg["segment_id"]
        decisions = _round_floats(decider.decide(seg["path"], signals), decimals)
        verdict = route_verdict(seg, signals, decisions, block["thresholds"], decimals)
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


def _closed_mismatch_batch(segments: list[dict], config: Any, decimals: int,
                           error: str) -> tuple[dict, list[dict]]:
    """Fail-closed batch: real L0/L1 stages, every verdict review-with-evidence.

    The receipt block records ``{"thresholds_error": ...}`` instead of
    thresholds (there are none to trust). The batch carries
    ``"thresholds_error"`` too so the service records it in the errors store;
    the store layer ignores the extra key. These receipts never re-verify
    MATCH (a later replay runs the live router on the stored duress block and
    gets a different verdict) -- honest: trust was unverifiable when made.
    """
    block = {
        "thresholds": {"thresholds_error": error},
        "pipeline_version": str(config.pipeline_version),
        "versions": impl_versions(config),
    }
    compute_signals = _compute_signals_fn()
    decider = _get_decider(_decider_class(), config)
    phash_index: dict[str, str] = {}
    receipts: list[dict] = []
    for seg in segments:
        signals = _compute_signals_safe(compute_signals, seg["path"],
                                        config, decimals)
        phash = str(signals.get("phash") or "")
        if phash and phash in phash_index:
            signals["dup_of"] = phash_index[phash]
        elif phash:
            phash_index[phash] = seg["segment_id"]
        decisions = _round_floats(decider.decide(seg["path"], signals), decimals)
        verdict = _thresholds_mismatch_verdict(seg, decimals)
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
        "thresholds_error": error,
    }
    return batch, receipts


def reverify_receipt(receipt: dict, segment: dict, config: Any) -> dict:
    """Re-run one stored segment with the receipt's STORED config; compare hashes.

    The L0/L1 stages get a Config rebuilt from the stored thresholds (never the
    live object), the verdict threshold comes from the stored verdict, and the
    config block passed to make_receipt is the stored block verbatim.
    """
    ensure_determinism(config)
    block = receipt.get("config")
    if not isinstance(block, dict):
        block = config_block(config)
    thresholds = block.get("thresholds") if isinstance(block, dict) else None
    if not isinstance(thresholds, dict):
        thresholds = thresholds_snapshot(config)
    replay_config = stored_config(block if isinstance(block, dict) else None) or config
    if not isinstance(replay_config, Config):
        replay_config = config
    # the replayed seed governs from here on, so a torch-backed L0/L1 replays
    # with the same RNG state that produced the receipt (no spurious MISMATCH)
    ensure_determinism(replay_config)
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
    decider = _get_decider(_decider_class(), replay_config)
    signals = _compute_signals_safe(compute_signals, segment["path"],
                                      replay_config, decimals)
    decisions = _round_floats(decider.decide(segment["path"], signals), decimals)
    verdict = route_verdict(segment, signals, decisions, replay_thresholds, decimals)
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


def _read_eval_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _find_eval_file(results_dir: Path, stem: str) -> Path | None:
    """``<stem>.json`` preferred, else the first ``<stem>*.json``; None if none.

    The suffixed fallback (e.g. ``decider_baseline_DEV_ONLY.json``) only ever
    surfaces real measurement files -- nothing is fabricated when absent.
    """
    exact = results_dir / f"{stem}.json"
    if exact.is_file():
        return exact
    cands = sorted(results_dir.glob(f"{stem}*.json"))
    return cands[0] if cands else None


def _source_of(text: Any) -> str | None:
    """One provenance text -> 'gemini' | 'human_gold' | 'synthetic' (None unknown).

    An explicit auto-labeller mention wins inside a single text:
    "labels by gemini-2.5-flash, not human-checked" is gemini labels.
    """
    if not isinstance(text, str):
        return None
    blob = text.lower()
    if "gemini" in blob:
        return "gemini"
    if "human" in blob:
        return "human_gold"
    if "synthetic" in blob:
        return "synthetic"
    return None


def _label_source(*texts: Any) -> str | None:
    """First named source wins (most specific text first); conflicts reported.

    Callers pass provenance texts from most specific (head level) to least
    (artifact file level). The first text that names a source decides; when a
    less specific text names a DIFFERENT source the disagreement is reported
    as ``'conflict:<specific>|<general>'`` instead of being silently
    overridden. Returns None when no text names a source.
    """
    decided: str | None = None
    for text in texts:
        source = _source_of(text)
        if source is None:
            continue
        if decided is None:
            decided = source
        elif source != decided:
            return f"conflict:{decided}|{source}"
    return decided


def _decider_head_entries(payload: dict, config: Any) -> list[dict]:
    """One entry per L1 head in a decider_baseline.json payload."""
    heads = payload.get("heads")
    if not isinstance(heads, dict):
        return []
    target_err = float(getattr(config, "target_error", 0.05))
    delta = float(getattr(config, "delta", 0.1))
    entries = []
    for head in sorted(heads):
        body = heads[head]
        if not isinstance(body, dict):
            continue
        entry: dict[str, Any] = {"head": str(head)}
        calib = body.get("calibration") if isinstance(body.get("calibration"), dict) else {}
        test = body.get("test") if isinstance(body.get("test"), dict) else {}
        pick = calib.get("threshold_pick") if isinstance(calib.get("threshold_pick"), dict) else {}
        points: list[dict] = []
        conf, correct = calib.get("conf"), calib.get("correct")
        if (isinstance(conf, list) and isinstance(correct, list) and conf
                and len(conf) == len(correct)):
            try:
                points = risk_coverage(
                    [{"conf": c, "correct": bool(v)} for c, v in zip(conf, correct)])
            except (ValueError, TypeError, KeyError):
                points = []
        entry["points"] = points
        if pick or calib:
            # Something was measured: the key is present even when the head
            # was explicitly NOT certified (threshold None).
            entry["chosen_threshold"] = pick.get("threshold", None)
            entry["upper_bound"] = pick.get("upper_bound", None)
        entry["target_err"] = target_err
        entry["delta"] = delta
        if isinstance(calib.get("n"), int):
            entry["n_calibration"] = calib["n"]
        if isinstance(test.get("n"), int):
            entry["n_test"] = test["n"]
        source = _label_source(body.get("provenance"), body.get("label_caveat"),
                               body.get("label_caveat_long"), payload.get("provenance"),
                               payload.get("label_caveat"))
        if source is not None:
            entry["label_source"] = source
        entries.append(entry)
    return entries


def _funnel_entries(payload: dict) -> list[dict]:
    """Accept entry plus one entry per reject rule in funnel_calibration.json.

    Understands both the original schema (top-level ``accept`` /
    ``reject_rules`` with ``calib_certified`` / ``calib`` blocks) and the
    post-fix-round-1 schema (``per_source.<origin>.accept`` /
    ``reject_rules`` with ``select_certified`` / ``select`` blocks and a
    ``certify`` held-out block). Numbers are only ever read, never made up:
    ``n_calibration`` is the SELECT count, ``n_test`` the CERTIFY count, and
    either key is absent when the file does not measure it.
    """
    if not isinstance(payload, dict):
        return []
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    targets = meta.get("targets") if isinstance(meta.get("targets"), dict) else {}
    target_alpha = targets.get("target_alpha", targets.get("accept_step", 0.05))
    target_beta = targets.get("target_beta", target_alpha)
    try:
        target_alpha_f, target_beta_f = float(target_alpha), float(target_beta)
    except (TypeError, ValueError):
        target_alpha_f, target_beta_f = 0.05, 0.05
    try:
        delta_f = float(targets.get("delta", 0.1))
    except (TypeError, ValueError):
        delta_f = 0.1
    per_source = payload.get("per_source")
    groups: dict[str, dict] = {}
    if isinstance(per_source, dict):
        for origin, group in per_source.items():
            if isinstance(group, dict):
                groups[str(origin)] = group
    else:
        groups["file"] = payload
    file_source = _label_source(meta.get("provenance"), payload.get("provenance"))
    entries = []
    for origin in sorted(groups):
        group = groups[origin]
        source = _label_source(origin if origin != "file" else None,
                               meta.get("provenance"),
                               payload.get("provenance"))
        if source is None:
            source = file_source
        accept = group.get("accept")
        if isinstance(accept, dict):
            certified = accept.get("select_certified")
            if not isinstance(certified, dict):
                certified = accept.get("calib_certified") if isinstance(
                    accept.get("calib_certified"), dict) else {}
            certify = accept.get("certify") if isinstance(
                accept.get("certify"), dict) else {}
            entry: dict[str, Any] = {"head": "accept", "points": []}
            if "threshold" in accept:
                entry["chosen_threshold"] = accept["threshold"]
                entry["upper_bound"] = certified.get("upper_bound", None)
            entry["target_err"] = target_alpha_f
            entry["delta"] = delta_f
            n_c = accept.get("n_select", accept.get("n_calib",
                             meta.get("n_calib")))
            if isinstance(n_c, int):
                entry["n_calibration"] = n_c
            n_t = certify.get("n_certify", meta.get("n_heldout"))
            if isinstance(n_t, int):
                entry["n_test"] = n_t
            if source is not None:
                entry["label_source"] = source
            entries.append(entry)
        rules = group.get("reject_rules")
        if isinstance(rules, dict):
            for key in sorted(rules):
                rule = rules[key]
                if not isinstance(rule, dict) or rule.get("disabled", False):
                    continue
                select = rule.get("select")
                if not isinstance(select, dict):
                    select = rule.get("calib") if isinstance(
                        rule.get("calib"), dict) else {}
                certify = rule.get("certify") if isinstance(
                    rule.get("certify"), dict) else {}
                entry = {"head": f"reject:{key}", "points": []}
                if "threshold" in rule:
                    entry["chosen_threshold"] = rule["threshold"]
                    entry["upper_bound"] = select.get("upper_bound", None)
                entry["target_err"] = target_beta_f
                entry["delta"] = delta_f
                n_c = rule.get("n_select", meta.get("n_calib"))
                if isinstance(n_c, int):
                    entry["n_calibration"] = n_c
                n_t = certify.get("n_certify", meta.get("n_heldout"))
                if isinstance(n_t, int):
                    entry["n_test"] = n_t
                if source is not None:
                    entry["label_source"] = source
                entries.append(entry)
    return entries


def risk_coverage_report(config: Any = None,
                         results_dir: str | Path | None = None,
                         store_points: list[dict] | None = None) -> list[dict]:
    """Per-head risk/coverage report for GET /stats/risk_coverage.

    Returns ``[{head, points, chosen_threshold, upper_bound, target_err,
    delta, n_calibration, n_test, label_source}]`` (one entry per measured
    head), reading ``eval/results/decider_baseline.json`` and
    ``eval/results/funnel_calibration.json`` when present, then SQLite
    calibration rows (head ``"answer"``), then legacy
    ``eval/results/calibration.json``. ``[]`` means no measured data --
    numbers are never invented. ``chosen_threshold`` is ``None`` when the
    head was measured but explicitly NOT certified; the key is absent only
    when nothing about a threshold was measured. The same rule governs every
    measurement field: ``upper_bound`` is ``None`` only when a threshold
    measurement exists but carries no bound (legacy fallback), otherwise the
    key is absent; ``n_calibration`` / ``n_test`` are present only with real
    counts (the store-backed head has no test set, so it omits ``n_test``).
    ``label_source`` is one of ``'gemini' | 'human_gold' | 'synthetic'``,
    head-level provenance first, and is omitted when no provenance is stated;
    a head/file disagreement is reported as ``'conflict:<a>|<b>'``.
    ``target_err`` / ``delta`` are the funnel file's targets on the funnel
    path and the service config everywhere else.
    """
    from jadal.config import ROOT
    base = Path(results_dir) if results_dir else ROOT / "eval" / "results"
    entries: list[dict] = []
    decider_path = _find_eval_file(base, "decider_baseline")
    if decider_path is not None:
        payload = _read_eval_json(decider_path)
        if payload is not None:
            entries.extend(_decider_head_entries(payload, config))
    funnel_path = _find_eval_file(base, "funnel_calibration")
    if funnel_path is not None:
        payload = _read_eval_json(funnel_path)
        if payload is not None:
            entries.extend(_funnel_entries(payload))
    if store_points:
        try:
            points = risk_coverage(store_points)
            pick = certify_mod.pick_threshold(
                conf=[p["conf"] for p in store_points],
                correct=[p["correct"] for p in store_points],
                target_err=float(getattr(config, "target_error", 0.05)),
                delta=float(getattr(config, "delta", 0.1)),
                min_accepted=int(getattr(config, "min_accepted", 30)),
            )
            entries.append({
                "head": "answer",
                "points": points,
                "chosen_threshold": pick.get("threshold", None),
                "upper_bound": pick.get("upper_bound", None),
                "target_err": float(getattr(config, "target_error", 0.05)),
                "delta": float(getattr(config, "delta", 0.1)),
                "n_calibration": len(store_points),
                # no "n_test": the store holds no test set, and an absent
                # measurement must stay absent (never 0).
            })
        except (ValueError, TypeError, KeyError):
            pass
    if not entries:
        legacy = risk_coverage_from_eval(base / "calibration.json")
        if legacy:
            entries.append({
                "head": "answer",
                "points": legacy,
                "chosen_threshold": None,
                "upper_bound": None,
                "target_err": float(getattr(config, "target_error", 0.05)) if config is not None else 0.05,
                "delta": float(getattr(config, "delta", 0.1)) if config is not None else 0.1,
            })
    return entries


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
    "REASON_ORDER", "HARD_REASONS", "BATCH_RULE_TEXT", "ThresholdsError",
    "impl_versions", "ensure_determinism", "pick_threshold", "thresholds_snapshot", "config_block",
    "stored_config", "process_clip", "make_verdict", "route_verdict", "run_batch",
    "reverify_receipt", "compare_versions", "receipt_problems", "risk_coverage",
    "risk_coverage_from_eval", "risk_coverage_report", "sop_gaps_from_verdicts",
]
