"""Determinism tests for the L1 Decider (issue #7, fix round 1).

CPU-hermetic: a fake ``embed_fn`` (pure function of the preprocessed frame
bytes) plus a synthetic ``frame_source`` are injected, so no GPU weights are
needed. GPU/backbone tests run only with ``JADAL_TEST_GPU=1`` and CUDA
present, otherwise they skip gracefully.

On batching and determinism: the backbone gives ~1e-3-different embeddings
for the same frame encoded alone vs inside a batch of 5 (measured in
``eval/results/bench_eg2.json``), so the Decider NEVER batches — every frame
is encoded with batch size 1. A segment is therefore identical whether the
pipeline processes it alone or inside a batch of 5, which test C1 checks on
the real backbone with the embedding cache DISABLED.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from jadal import decision as decision_mod
from jadal.decision import Decider, head_predict_proba, load_head, save_head
from jadal.decision import save_step_prototypes
from jadal.receipts import canonical_json


def fake_embed_pil(img) -> np.ndarray:
    """Deterministic 768-d pseudo-embedding from preprocessed frame bytes (test only)."""
    raw = np.asarray(img.convert("RGB")).tobytes()
    h = hashlib.sha256(raw).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(decision_mod.EMBED_DIM)
    return v / np.linalg.norm(v)


def synthetic_frame_source(path: str, n: int = decision_mod.N_FRAMES) -> list:
    """Deterministic BGR-ish frames per path (test only; no video decoding)."""
    seed = int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(64, 64, 3)).astype(np.uint8) for _ in range(n)]


def make_decider(**kw):
    tmp = tempfile.mkdtemp(prefix="decider-test-")
    kw.setdefault("heads_dir", tmp)
    kw.setdefault("cache_dir", os.path.join(tmp, "cache"))
    kw.setdefault("embed_fn", fake_embed_pil)
    kw.setdefault("frame_source", synthetic_frame_source)
    return Decider(config=None, **kw)


def write_clip(path: str, seed: int, n_frames: int = 10, size: int = 64) -> None:
    import cv2

    rng = np.random.default_rng(seed)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (size, size))
    try:
        for _ in range(n_frames):
            w.write(rng.integers(0, 256, size=(size, size, 3)).astype(np.uint8))
    finally:
        w.release()


class TestDeciderDeterminism(unittest.TestCase):
    def test_same_input_twice_identical(self):
        d = make_decider()
        a = d.decide("/tmp/segA.mp4", {})
        b = d.decide("/tmp/segA.mp4", {})
        self.assertEqual(a, b)

    def test_alone_vs_in_batch_of_5_identical(self):
        paths = [f"/tmp/batch-seg{i}.mp4" for i in range(5)]
        d_alone = make_decider()
        solo = d_alone.decide(paths[0], {})
        d_batch = make_decider()
        batch = [d_batch.decide(p, {}) for p in paths]
        self.assertEqual(solo, batch[0])

    def test_spec_shape_and_rounding(self):
        d = make_decider()
        out = d.decide("/tmp/shape.mp4", {})
        self.assertTrue({"hands_ok", "active_work", "step", "model_versions"} <= set(out))
        self.assertEqual(out["model_versions"]["granularity"], "per single frame")
        for head in ("hands_ok", "active_work"):
            p = out[head]["p_yes"]
            self.assertIsInstance(p, float)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            self.assertEqual(p, round(p, 4))
        self.assertIsNone(out["step"]["choice"])
        self.assertEqual(out["step"]["answer_conf"], 0.0)

    def test_missing_heads_fallback_half(self):
        d = make_decider()
        out = d.decide("/tmp/noheads.mp4", {})
        self.assertEqual(out["hands_ok"]["p_yes"], 0.5)
        self.assertEqual(out["active_work"]["p_yes"], 0.5)
        self.assertIn("missing", out["model_versions"]["heads"])

    def test_head_roundtrip_and_temperature(self):
        tmp = tempfile.mkdtemp(prefix="decider-head-")
        rng = np.random.default_rng(0)
        coef = rng.standard_normal(decision_mod.EMBED_DIM)
        sha = save_head(os.path.join(tmp, "h.npz"), coef=coef, intercept=0.5,
                        temperature=2.0)
        self.assertEqual(len(sha), 64)
        h = load_head(os.path.join(tmp, "h.npz"))
        self.assertEqual(h["sha256"], sha)
        x = rng.standard_normal(decision_mod.EMBED_DIM)
        x = x / np.linalg.norm(x)
        z = float(x @ coef + 0.5)
        want = 1.0 / (1.0 + np.exp(-z / 2.0))
        self.assertAlmostEqual(float(head_predict_proba(h, x)[0]), want, places=9)

    def test_trained_heads_change_probs(self):
        d = make_decider()
        before = d.decide("/tmp/withheads.mp4", {})
        rng = np.random.default_rng(1)
        meta = {"provenance": "full_splits:test"}
        save_head(os.path.join(d.heads_dir, decision_mod.HANDS_HEAD),
                  coef=rng.standard_normal(decision_mod.EMBED_DIM),
                  intercept=2.0, temperature=1.0, meta=meta)
        save_head(os.path.join(d.heads_dir, decision_mod.WORK_HEAD),
                  coef=rng.standard_normal(decision_mod.EMBED_DIM),
                  intercept=-2.0, temperature=1.0, meta=meta)
        d2 = Decider(config=None, heads_dir=d.heads_dir,
                     cache_dir=os.path.join(d.heads_dir, "cache2"),
                     embed_fn=fake_embed_pil, frame_source=synthetic_frame_source)
        after = d2.decide("/tmp/withheads.mp4", {})
        self.assertNotEqual(before, after)
        self.assertGreater(after["hands_ok"]["p_yes"], 0.5)
        self.assertLess(after["active_work"]["p_yes"], 0.5)

    def test_step_prototypes_wire_into_decide(self):
        d = make_decider()
        self.assertIsNone(d.decide("/tmp/nostep.mp4", {})["step"]["choice"])
        rng = np.random.default_rng(3)
        A = rng.standard_normal(decision_mod.EMBED_DIM)
        A = A / np.linalg.norm(A)
        B = -A
        save_step_prototypes(os.path.join(d.heads_dir, decision_mod.STEP_PROTO),
                             steps=["step_a", "step_b"],
                             prototypes=np.stack([A, B]), temperature=0.05)
        d2 = Decider(config=None, heads_dir=d.heads_dir,
                     cache_dir=os.path.join(d.heads_dir, "cache3"),
                     embed_fn=fake_embed_pil, frame_source=synthetic_frame_source)
        out = d2.decide("/tmp/withstep.mp4", {})
        step = out["step"]
        self.assertIn(step["choice"], ("step_a", "step_b"))
        self.assertAlmostEqual(sum(step["probs"].values()), 1.0, places=3)
        self.assertEqual(step["answer_conf"], round(step["probs"][step["choice"]], 4))
        self.assertNotEqual(step["choice"], None)
        # deterministic across calls
        self.assertEqual(out, d2.decide("/tmp/withstep.mp4", {}))

    def test_cache_key_includes_budget(self):
        """Two budgets sharing a cache dir must never share an entry (I2)."""
        tmp = tempfile.mkdtemp(prefix="decider-budget-")
        calls = []

        def counting_embed(img):
            calls.append(1)
            return fake_embed_pil(img)

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        d70 = Decider(config=None, heads_dir=tmp, cache_dir=os.path.join(tmp, "c"),
                      image_token_budget=70, embed_fn=counting_embed,
                      frame_source=lambda p: [frame])
        d140 = Decider(config=None, heads_dir=tmp, cache_dir=os.path.join(tmp, "c"),
                       image_token_budget=140, embed_fn=counting_embed,
                       frame_source=lambda p: [frame])
        d70.embed_frame(frame)
        d140.embed_frame(frame)
        self.assertEqual(len(calls), 2)  # second budget missed: keys differ
        files = os.listdir(os.path.join(tmp, "c"))
        self.assertEqual(len(files), 2)
        # and a repeat hit uses the cache
        d70.embed_frame(frame)
        self.assertEqual(len(calls), 2)

    def test_dev_heads_refused_without_env(self):
        tmp = tempfile.mkdtemp(prefix="decider-prov-")
        rng = np.random.default_rng(5)
        save_head(os.path.join(tmp, decision_mod.HANDS_HEAD),
                  coef=rng.standard_normal(decision_mod.EMBED_DIM),
                  intercept=0.0, temperature=1.0,
                  meta={"provenance": "DEV_ONLY"})
        save_head(os.path.join(tmp, decision_mod.WORK_HEAD),
                  coef=rng.standard_normal(decision_mod.EMBED_DIM),
                  intercept=0.0, temperature=1.0,
                  meta={"provenance": "DEV_ONLY"})
        env = {k: v for k, v in os.environ.items() if k != decision_mod.ALLOW_DEV_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            d = Decider(config=None, heads_dir=tmp,
                        cache_dir=os.path.join(tmp, "c"),
                        embed_fn=fake_embed_pil, frame_source=synthetic_frame_source)
            with self.assertRaises(RuntimeError):
                d.decide("/tmp/devheads.mp4", {})
        with mock.patch.dict(os.environ, {decision_mod.ALLOW_DEV_ENV: "1"}):
            d2 = Decider(config=None, heads_dir=tmp,
                         cache_dir=os.path.join(tmp, "c2"),
                         embed_fn=fake_embed_pil, frame_source=synthetic_frame_source)
            out = d2.decide("/tmp/devheads.mp4", {})
            self.assertIn("DEV_ONLY", out["model_versions"]["heads_provenance"])

    def test_full_heads_allowed_without_env(self):
        tmp = tempfile.mkdtemp(prefix="decider-prov-full-")
        rng = np.random.default_rng(6)
        meta = {"provenance": "full_splits:abc123"}
        save_head(os.path.join(tmp, decision_mod.HANDS_HEAD),
                  coef=rng.standard_normal(decision_mod.EMBED_DIM),
                  intercept=0.0, temperature=1.0, meta=meta)
        save_head(os.path.join(tmp, decision_mod.WORK_HEAD),
                  coef=rng.standard_normal(decision_mod.EMBED_DIM),
                  intercept=0.0, temperature=1.0, meta=meta)
        env = {k: v for k, v in os.environ.items() if k != decision_mod.ALLOW_DEV_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            d = Decider(config=None, heads_dir=tmp,
                        cache_dir=os.path.join(tmp, "c"),
                        embed_fn=fake_embed_pil, frame_source=synthetic_frame_source)
            out = d.decide("/tmp/fullheads.mp4", {})
            self.assertIn("full_splits", out["model_versions"]["heads_provenance"])


class TestRenderDeciderReport(unittest.TestCase):
    """The report tables are generated from the JSON (fix round 3)."""

    def test_renderer_contains_every_head_metric_and_bound(self):
        from eval.render_decider_report import (
            render_baseline_tables, render_bench_line, render_codec_shift_line)

        baseline = {
            "provenance": "full_splits:fixture",
            "heads": {
                "hands_ok": {
                    "calibration": {"n": 10, "accuracy": 0.9, "ece": 0.05, "temperature": 0.7},
                    "test": {"n": 11, "accuracy": 0.8182, "ece": 0.0601, "temperature": 0.7},
                    "held_out_bound_at_cal_threshold": {
                        "threshold": 0.8, "accepted": 5, "errors": 1, "upper_bound": 0.63},
                },
                "active_work": {
                    "calibration": {"n": 12, "accuracy": 0.75, "ece": 0.11, "temperature": 1.2},
                    "test": {"n": 13, "accuracy": 0.6923, "ece": 0.2, "temperature": 1.2},
                    "held_out_bound_at_cal_threshold": {
                        "threshold": None, "accepted": 0, "errors": 0, "upper_bound": None},
                },
            },
            "codec_shift": {"n": 4, "mean": 0.99, "min": 0.97, "p5": 0.975, "p50": 0.991},
        }
        tables = render_baseline_tables(baseline)
        for head in ("hands_ok", "active_work"):
            for split in ("calibration", "test"):
                m = baseline["heads"][head][split]
                for value in (m["accuracy"], m["ece"], m["temperature"]):
                    self.assertIn(f"{value:.4f}", tables)
            b = baseline["heads"][head]["held_out_bound_at_cal_threshold"]
            for value in (b["accepted"], b["errors"]):
                self.assertIn(f"| {value} |", tables)
            self.assertIn(str(b["upper_bound"]), tables)
        self.assertIn("0.9900", render_codec_shift_line(baseline))
        self.assertIn("not present", render_bench_line(None))

    def test_renderer_on_real_baseline_json(self):
        from eval.render_decider_report import render_baseline_tables

        real = Path(__file__).resolve().parent.parent / "eval" / "results" / "decider_baseline.json"
        if not real.exists():
            self.skipTest("decider_baseline.json not present")
        import json
        tables = render_baseline_tables(json.loads(real.read_text()))
        for head in ("hands_ok", "active_work"):
            self.assertIn(head, tables)


@unittest.skipUnless(os.environ.get("JADAL_TEST_GPU") == "1", "needs JADAL_TEST_GPU=1")
class TestDeciderGPU(unittest.TestCase):
    """C1: real-backbone determinism with the embedding cache DISABLED.

    (a) same segment twice -> identical Decisions; (b) segments s0..s4 in
    sequence (as the pipeline processes a batch of 5) then s0 again -> s0
    identical to its solo result, compared after 4-decimal rounding AND as a
    hash of the canonical JSON; (c) 10 repeated runs identical. Each Decider
    uses use_cache=False so every equality exercises the backbone, never RAM.
    """

    def test_backbone_determinism_cache_disabled(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA absent")
        if not os.path.exists(os.path.join(decision_mod.EG2_DIR, "model.safetensors")):
            self.skipTest("EG2 weights absent")
        tmp = tempfile.mkdtemp(prefix="decider-gpu-")
        paths = [os.path.join(tmp, f"s{i}.mp4") for i in range(5)]
        for i, p in enumerate(paths):
            write_clip(p, seed=100 + i)
        # full-provenance dummy heads so the head path is exercised too
        rng = np.random.default_rng(9)
        meta = {"provenance": "full_splits:test"}
        for fname in (decision_mod.HANDS_HEAD, decision_mod.WORK_HEAD):
            save_head(os.path.join(tmp, fname),
                      coef=rng.standard_normal(decision_mod.EMBED_DIM),
                      intercept=0.0, temperature=1.0, meta=meta)

        def fresh_decider():
            return Decider(config=None, heads_dir=tmp, use_cache=False)

        # (a) same segment twice
        d = fresh_decider()
        a1 = d.decide(paths[0], {})
        a2 = d.decide(paths[0], {})
        self.assertEqual(a1, a2)

        # (b) s0..s4 in sequence (batch of 5 as the pipeline runs them), then s0
        d_batch = fresh_decider()
        seq = [d_batch.decide(p, {}) for p in paths]
        s0_again = d_batch.decide(paths[0], {})
        self.assertEqual(seq[0], s0_again)
        self.assertEqual(a1, seq[0])
        h1 = hashlib.sha256(canonical_json(a1)).hexdigest()
        h2 = hashlib.sha256(canonical_json(s0_again)).hexdigest()
        self.assertEqual(h1, h2)

        # (c) 10 repeated runs identical
        reps = [fresh_decider().decide(paths[0], {}) for _ in range(10)]
        for r in reps[1:]:
            self.assertEqual(reps[0], r)
        self.assertEqual(a1["model_versions"]["frames_decoded"], "4")


if __name__ == "__main__":
    unittest.main()
