"""Determinism tests for the L1 Decider (issue #7).

CPU-hermetic: a fake ``embed_fn`` (pure function of the frame bytes) is
injected, so no GPU weights are needed. GPU/backbone tests run only with
``JADAL_TEST_GPU=1`` and CUDA present, otherwise they skip gracefully.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

import numpy as np

from jadal import decision as decision_mod
from jadal.decision import Decider, head_predict_proba, load_head, save_head
from jadal.decision import save_step_prototypes, step_predict_proba


def fake_embed(bgr: np.ndarray) -> np.ndarray:
    """Deterministic 768-d pseudo-embedding from frame bytes (test only)."""
    h = hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(decision_mod.EMBED_DIM)
    return v / np.linalg.norm(v)


def make_decider(**kw):
    tmp = tempfile.mkdtemp(prefix="decider-test-")
    return Decider(config=None, heads_dir=tmp, cache_dir=os.path.join(tmp, "cache"),
                   embed_fn=fake_embed, **kw)


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
        self.assertEqual(set(out), {"hands_ok", "active_work", "step", "model_versions"})
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
                     cache_dir=os.path.join(d.heads_dir, "cache3"), embed_fn=fake_embed)
        out = d2.decide("/tmp/withstep.mp4", {})
        step = out["step"]
        self.assertIn(step["choice"], ("step_a", "step_b"))
        self.assertAlmostEqual(sum(step["probs"].values()), 1.0, places=3)
        self.assertEqual(step["answer_conf"], round(step["probs"][step["choice"]], 4))
        self.assertNotEqual(step["choice"], None)
        # deterministic across calls
        self.assertEqual(out, d2.decide("/tmp/withstep.mp4", {}))

    def test_trained_heads_change_probs(self):
        d = make_decider()
        before = d.decide("/tmp/withheads.mp4", {})
        rng = np.random.default_rng(1)
        save_head(os.path.join(d.heads_dir, decision_mod.HANDS_HEAD),
                  coef=rng.standard_normal(decision_mod.EMBED_DIM),
                  intercept=2.0, temperature=1.0)
        save_head(os.path.join(d.heads_dir, decision_mod.WORK_HEAD),
                  coef=rng.standard_normal(decision_mod.EMBED_DIM),
                  intercept=-2.0, temperature=1.0)
        d2 = Decider(config=None, heads_dir=d.heads_dir,
                     cache_dir=os.path.join(d.heads_dir, "cache2"), embed_fn=fake_embed)
        after = d2.decide("/tmp/withheads.mp4", {})
        self.assertNotEqual(before, after)
        self.assertGreater(after["hands_ok"]["p_yes"], 0.5)
        self.assertLess(after["active_work"]["p_yes"], 0.5)


@unittest.skipUnless(os.environ.get("JADAL_TEST_GPU") == "1", "needs JADAL_TEST_GPU=1")
class TestDeciderGPU(unittest.TestCase):
    def test_backbone_loads_and_decides(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA absent")
        if not os.path.exists(os.path.join(decision_mod.EG2_DIR, "model.safetensors")):
            self.skipTest("EG2 weights absent")
        import cv2

        tmp = tempfile.mkdtemp(prefix="decider-gpu-")
        clip = os.path.join(tmp, "clip.mp4")
        w = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
        rng = np.random.default_rng(7)
        for _ in range(10):
            w.write(rng.integers(0, 256, size=(64, 64, 3)).astype(np.uint8))
        w.release()
        d = Decider(config=None, heads_dir=tmp, cache_dir=os.path.join(tmp, "c"))
        a = d.decide(clip, {})
        b = d.decide(clip, {})
        self.assertEqual(a, b)
        self.assertEqual(a["model_versions"]["frames_decoded"], "4")


if __name__ == "__main__":
    unittest.main()
