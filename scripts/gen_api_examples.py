#!/usr/bin/env python
"""Generate docs/api_examples/*.json from the running stub-mode service.

Spins up the service on temp dirs (stubs forced, fixed clip bytes so every
id and hash is deterministic), exercises every documented endpoint, drops
the volatile timestamp keys (created_utc/started_utc/failed_utc -- shown
separately in docs/API.md) and writes one JSON file per endpoint. The test
``test_all_committed_examples_match_live_responses`` rebuilds the same dict
via :func:`build_examples` and fails on ANY drift.

Run: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/gen_api_examples.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXAMPLES_DIR = ROOT / "docs" / "api_examples"
_MISSING = object()
VOLATILE_KEYS = frozenset({"created_utc", "started_utc", "failed_utc"})
VOLATILE_VALUES = {
    "receipt_sha256": "<sha256>",
    "receipt_id": "<receipt-id>",
    "expected": "<sha256>",
    "actual": "<sha256>",
}

PASS_SIGNALS = {
    "decodable": True, "blur_var": 120.0, "brightness_mean": 90.0,
    "under_exposed_frac": 0.0, "over_exposed_frac": 0.02, "motion_mean": 2.0,
    "hands_visible_frac": 0.8, "hand_count_mode": 2, "faces_found": 0,
    "faces_blurred": 0, "dup_of": None,
}
PASS_DECISIONS = {
    "hands_ok": {"p_yes": 0.97},
    "active_work": {"p_yes": 0.93},
    "step": {"choice": "wash", "probs": {"wash": 0.93}, "answer_conf": 0.93},
}


def _normalized(obj):
    # Canonicalize values that legitimately vary run to run (timestamps,
    # content hashes, impl version strings) to fixed tokens. Key presence
    # and list lengths stay pinned, so shape drift still fails.
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in VOLATILE_KEYS:
                continue
            if k == "versions" and isinstance(v, dict):
                out[k] = {vk: "<version>" for vk in sorted(v)}
            elif k == "receipt_ids" and isinstance(v, list):
                out[k] = ["<receipt-id>" for _ in v]
            elif k in VOLATILE_VALUES and isinstance(v, str):
                out[k] = VOLATILE_VALUES[k]
            else:
                out[k] = _normalized(v)
        return out
    if isinstance(obj, list):
        return [_normalized(v) for v in obj]
    return obj


def build_examples() -> dict:
    """Run the stub-mode service and return {file_stem: payload}.

    Stub callables are injected (not just the env flag) AND the real
    L0/L1/segmenter modules are blocked, so the output -- including the
    receipt ``versions`` block -- is identical whether or not those modules
    are importable. The funnel module stays live (its route + evidence are
    part of the documented behavior).
    """
    from fastapi.testclient import TestClient
    from jadal import pipeline
    from jadal import stubs
    from jadal.config import Config
    from jadal.service import create_app
    from jadal import store as store_mod

    stage_patches = [
        mock.patch.object(pipeline, "_compute_signals_fn",
                          new=lambda: stubs.stub_compute_signals),
        mock.patch.object(pipeline, "_decider_class",
                          new=lambda: stubs.StubDecider),
        mock.patch.object(pipeline, "_split_clip_fn",
                          new=lambda: stubs.stub_split_clip),
    ]
    blocked = ("jadal.signals", "jadal.decision", "jadal.segmenter")
    old_modules = {name: sys.modules.get(name,
                                         _MISSING) for name in blocked}
    with mock.patch.dict(os.environ, {"JADAL_USE_STUBS": "1",
                                      "JADAL_EVAL_RESULTS_DIR":
                                          str(EXAMPLES_DIR)}):
        for patcher in stage_patches:
            patcher.start()
        for name in blocked:
            sys.modules[name] = None  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = Config(
                    batch_size=5, batch_reject_k=1, segment_seconds=20.0, seed=1234,
                    bucket_dir=root / "bucket", segments_dir=root / "segments",
                    blurred_dir=root / "blurred", db_path=root / "jadal.db",
                    web_dir=root / "web", voicebank_dir=root / "voicebank")
                cfg.bucket_dir.mkdir(parents=True, exist_ok=True)
                store_mod.init_db(cfg.db_path)
                app = create_app(cfg, batch_max_age_s=30.0, watch_bucket=False)
                with TestClient(app) as client:
                    clips = []
                    for i in range(5):
                        path = cfg.bucket_dir / f"example_{i}.mp4"
                        path.write_bytes((f"api-example-clip:{i}\n".encode()
                                          * 64)[:4096])
                        signals = dict(PASS_SIGNALS)
                        if i == 0:
                            signals = dict(signals, blur_var=2.0)  # one BLUR reject
                        stubs.set_test_override(path, signals=signals,
                                                decisions=dict(PASS_DECISIONS))
                        clips.append(path)
                    last = None
                    for path in clips:
                        response = client.post("/ingest/event", json={
                            "path": str(path), "worker_id": "w-ex",
                            "site_id": "site-ex"})
                        assert response.status_code == 200, response.text
                        last = response.json()
                    assert last["batch"] is not None
                    batch_id = last["batch"]["batch_id"]
                    detail = client.get(f"/batches/{batch_id}").json()
                    receipt_id = detail["receipt_ids"][0]
                    out = {
                        "health": client.get("/health").json(),
                        "batches": client.get("/batches").json(),
                        "batch_detail": detail,
                        "receipt": client.get(f"/receipts/{receipt_id}").json(),
                        "reverify": client.post(
                            f"/receipts/{receipt_id}/reverify").json(),
                        "risk_coverage": client.get(
                            "/stats/risk_coverage").json(),
                        "sop_gaps": client.get("/stats/sop_gaps").json(),
                        "pico_status": client.get("/pico/status").json(),
                        "ingest_errors": client.get("/ingest/errors").json(),
                        "batches_failed": client.get("/batches/failed").json(),
                    }
                stubs.clear_test_overrides()
        finally:
            for name in blocked:
                if old_modules[name] is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_modules[name]
            for patcher in reversed(stage_patches):
                patcher.stop()
    return {name: _normalized(payload) for name, payload in out.items()}


def main() -> int:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in sorted(build_examples().items()):
        path = EXAMPLES_DIR / f"{name}.json"
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
