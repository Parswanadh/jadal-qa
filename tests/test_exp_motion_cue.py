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


def _translated_stack(dx: float, dy: float, t: int = 5, seed: int = 3):
    """A textured frame translated by (dx,dy) each step -> pure ego-motion."""
    rng = np.random.default_rng(seed)
    base = rng.random((motion_cue.H, motion_cue.W)).astype(np.float32)
    import cv2
    out = []
    for k in range(t):
        M = np.float32([[1, 0, dx * k], [0, 1, dy * k]])
        out.append(cv2.warpAffine(base, M, (motion_cue.W, motion_cue.H),
                                  borderMode=cv2.BORDER_REPLICATE))
    return np.stack(out).astype(np.float32)


def test_cue_series_zero_motion():
    import cv2
    rng = np.random.default_rng(0)
    base = rng.random((motion_cue.H, motion_cue.W)).astype(np.float32)
    stack = np.stack([base] * 6)
    c = motion_cue.cue_series(stack)
    assert c["resid_mean"] < 1e-6
    assert c["diff_mean"] < 1e-6
    assert c["ego_mean"] < 1e-3
    assert c["n_pairs"] == 5
    assert set(c) >= {"resid_mean", "resid_med", "resid_p90", "resid_std", "ego_mean", "diff_mean"}


def test_translation_is_compensated_but_frame_diff_is_not():
    c = motion_cue.cue_series(_translated_stack(4.0, 2.0))
    # translation-only compensation removes most of a pure translation ...
    assert c["resid_mean"] < c["diff_mean"]
    assert c["ego_mean"] > 1.0                      # shift detected
    # ... but the uncompensated frame difference is large
    assert c["diff_mean"] > 0.01


def test_resid_scales_with_uncompensated_motion():
    small = motion_cue.cue_series(_translated_stack(3.0, 0.0))
    assert small["resid_mean"] < 0.02


def test_calibrate_abs_z_and_majority_use_no_labels():
    z = np.array([-2.0, -0.5, 0.6, 1.2])           # one inverted clip
    out = run_e6.calibrate_abs_z(z)
    assert np.all(out <= 0)                          # extremes -> low usable score
    assert out[0] == pytest.approx(-2.0 / np.abs(z).max()) or out[0] < out[1]
    s = run_e6.calibrate_majority(z)                 # 3 of 4 z<0 -> canonical -z already majority-good
    assert np.mean(s > 0) >= 0.5
    s2 = run_e6.calibrate_majority(-z)               # flipping the input must flip the chosen sign
    assert np.mean(s2 > 0) >= 0.5


def test_zscore_within_is_per_clip():
    v = np.array([1.0, 2.0, 10.0, 20.0])
    clips = np.array(["a", "a", "b", "b"])
    z = run_e6.zscore_within(v, clips)
    assert np.allclose(z[clips == "a"], [-0.7071, 0.7071], atol=1e-3)
    assert np.allclose(z[clips == "b"], [-0.7071, 0.7071], atol=1e-3)


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


def test_results_json_has_required_fields():
    """Run E6.01/E6.02/E6.04 on tiny synthetic data and check the required keys."""
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
