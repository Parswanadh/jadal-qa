"""Tests for the E6 motion-cue experiment (< 30 s, no GPU/network/real video).

Covers the cue extraction on synthetic frames, the label-free per-clip sign
calibration, the statistics helpers and that the experiment runners emit the
required results-JSON fields.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval" / "exp_motion_cue"))

import motion_cue  # noqa: E402
import run_e6  # noqa: E402


def _textured_base(seed: int = 3, sigma: float = 3.0):
    """Smooth, non-periodic, fully textured frame (phase correlation is well posed)."""
    import cv2
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.random((motion_cue.H, motion_cue.W)).astype(np.float32),
                            (0, 0), sigma)


def _stack(dx: float, dy: float, t: int = 6, noise: float = 0.0, seed: int = 5):
    import cv2
    base = _textured_base()
    rng = np.random.default_rng(seed)
    out = []
    for k in range(t):
        f = cv2.warpAffine(base, np.float32([[1, 0, dx * k], [0, 1, dy * k]]),
                           (motion_cue.W, motion_cue.H), borderMode=cv2.BORDER_REPLICATE)
        if noise > 0:
            f = f + noise * rng.standard_normal((motion_cue.H, motion_cue.W)).astype(np.float32)
        out.append(f)
    return np.stack(out).astype(np.float32)


def test_cue_series_zero_motion():
    rng = np.random.default_rng(0)
    base = rng.random((motion_cue.H, motion_cue.W)).astype(np.float32)
    stack = np.stack([base] * 6)
    c = motion_cue.cue_series(stack)
    assert c["resid_mean"] < 1e-6
    assert c["diff_mean"] < 1e-6
    assert c["ego_mean"] < 1e-3
    assert c["n_pairs"] == 5
    assert set(c) >= {"resid_mean", "resid_med", "resid_p90", "resid_std", "ego_mean", "diff_mean"}


def test_global_translation_is_detected_and_leaves_residual_motion():
    c = motion_cue.cue_series(_stack(4.0, 2.0))
    assert c["ego_mean"] > 2.0                       # translation-only motion detected
    assert c["resid_mean"] > 0.0                     # interpolation/edge residual, not zero
    assert c["diff_mean"] > 0.0


def test_independent_noise_is_residual_not_global_motion():
    c = motion_cue.cue_series(_stack(0.0, 0.0, noise=0.05))
    assert c["resid_mean"] > 0.02                    # residual motion present
    assert c["ego_mean"] < 1.5                       # but no consistent global translation


def test_calibrate_abs_z_and_majority_use_no_labels():
    z = np.array([-2.0, -0.5, 0.6, 1.2])            # one inverted clip
    out = run_e6.calibrate_abs_z(z)
    assert np.all(out <= 0)                          # extremes -> low usable score
    assert out[0] < out[1]                           # bigger |z| scores worse
    s, flipped = run_e6.calibrate_majority(z)
    assert np.mean(s > 0) >= 0.5
    assert flipped is False
    # a skewed clip: flipping the input must flip the chosen sign
    skew = np.array([3.0, 3.2, 3.1, 0.2])
    _, f1 = run_e6.calibrate_majority(skew)
    _, f2 = run_e6.calibrate_majority(-skew)
    assert f1 is False and f2 is True


def test_zscore_within_is_per_clip():
    v = np.array([1.0, 2.0, 10.0, 20.0])
    clips = np.array(["a", "a", "b", "b"])
    z = run_e6.zscore_within(v, clips)
    assert np.allclose(z[clips == "a"], [-1.0, 1.0], atol=1e-3)
    assert np.allclose(z[clips == "b"], [-1.0, 1.0], atol=1e-3)


def test_auroc_and_ci_helpers():
    y = np.array([0, 0, 1, 1])
    clips = np.array(["a", "a", "b", "b"])
    assert run_e6.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    lo, hi = run_e6.clip_bootstrap_ci(y, np.array([0.1, 0.2, 0.8, 0.9]), clips, B=50)
    assert lo <= 1.0 <= hi + 1e-9
    lo2, hi2 = run_e6.window_bootstrap_ci(y, np.array([0.1, 0.2, 0.8, 0.9]), B=50)
    assert lo2 <= 1.0 <= hi2 + 1e-9


def test_partial_spearman_removes_a_confound():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    cue = z + 0.01 * rng.normal(size=200)            # cue explained by z only
    y = (z > 0).astype(int)                          # label explained by z only
    r, lo, hi = run_e6.spearman_partial_ci(cue, y, z, np.array(["c"] * 200), B=40)
    assert abs(r) < 0.3                              # confound removed
    assert lo <= r <= hi


def test_results_json_has_required_fields(monkeypatch):
    """Run E6.01/E6.02/E6.04 on tiny synthetic data and check the required keys."""
    # keep the suite fast: shrink the bootstrap inside this test only
    import functools
    for fn in ("clip_bootstrap_ci", "window_bootstrap_ci"):
        monkeypatch.setattr(run_e6, fn, functools.partial(getattr(run_e6, fn), B=20))
    for fn in ("spearman_ci", "spearman_partial_ci", "mean_per_clip_auroc_ci"):
        monkeypatch.setattr(run_e6, fn, functools.partial(getattr(run_e6, fn), B=20))
    n, cl = 24, np.array(["a"] * 8 + ["b"] * 8 + ["c"] * 8)
    rng = np.random.default_rng(1)
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0] * 3)
    truth = -y + 0.1
    cue = {k: (truth + 0.3 * rng.normal(size=n)) for k in run_e6.CUE_VARIANTS}
    cached = {"shift_px_mean": np.abs(rng.normal(size=n)) * 10,
              "shift_px_med": np.abs(rng.normal(size=n)),
              "diff_e_mean": truth + rng.normal(size=n),
              "motion_energy": truth + rng.normal(size=n),
              "global_shift": rng.normal(size=n)}
    meta = {"device": "test", "command": "pytest", "git_sha": "x", "cost": {}}
    for fn, args in ((run_e6.run_e601, (cue, y, cl, cached, meta)),
                     (run_e6.run_e602, (cue, y, cl, meta)),
                     (run_e6.run_e604, (cue, y, cl, cached, meta))):
        res = fn(*args)
        for key in ("n", "label_source", "protocol", "ci95", "device", "command", "git_sha", "cost"):
            assert key in res, f"{fn.__name__} missing {key}"
        assert res["n"] == n
        json.dumps(res)                              # serialisable
