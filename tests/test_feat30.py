"""Tests for the 30-feature per-pixel vector (F30). Fast, no network, no decode."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jadal.gpu.features import FEATURE_NAMES, cpu_window_features  # noqa: E402
from jadal.gpu.features30 import (FEATURE_NAMES30, NEW_FEATURES, invariants_step,  # noqa: E402
                                  window_invariants, cpu_window_features30)


def _synthetic(T=6, H=90, W=160, seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.random((T, H, W, 3)) * 0.4 + 0.3
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return luma, rgb


def test_names_are_26_plus_4():
    assert len(FEATURE_NAMES) == 26
    assert len(FEATURE_NAMES30) == 30
    assert FEATURE_NAMES30[:26] == list(FEATURE_NAMES)
    assert FEATURE_NAMES30[26:] == NEW_FEATURES
    assert len(set(FEATURE_NAMES30)) == 30


def test_invariants_step_clip_frac_rails():
    H, W = 90, 160
    black = np.zeros((H, W, 3))
    white = np.ones((H, W, 3))
    mid = np.full((H, W, 3), 0.5)
    assert invariants_step(black)["inv_clip_frac"] == 1.0
    assert invariants_step(white)["inv_clip_frac"] == 1.0
    assert invariants_step(mid)["inv_clip_frac"] == 0.0


def test_window_invariants_shapes_and_finite():
    _, rgb = _synthetic()
    inv = window_invariants(rgb)
    assert set(inv) == set(NEW_FEATURES)
    for k, v in inv.items():
        assert np.isfinite(v), k


def test_window_features30_has_all_keys():
    luma, rgb = _synthetic()
    out = cpu_window_features30(luma, rgb, fps=10.0)
    assert list(out.keys()) == FEATURE_NAMES30
    base = cpu_window_features(luma, rgb, fps=10.0)
    for k in FEATURE_NAMES:
        assert out[k] == base[k]  # 26 features unchanged
    for k in NEW_FEATURES:
        assert np.isfinite(out[k])


def test_window_invariants_average_matches_steps():
    _, rgb = _synthetic(T=4)
    inv = window_invariants(rgb)
    steps = [invariants_step(rgb[t]) for t in range(rgb.shape[0] - 1)]
    for k in NEW_FEATURES:
        assert abs(inv[k] - np.mean([s[k] for s in steps])) < 1e-12
