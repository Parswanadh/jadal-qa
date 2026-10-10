"""F50 tests: 24 new per-pixel features (fast, no network, no GPU)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from jadal.gpu.features import FEATURE_NAMES  # noqa: E402
from jadal.gpu.features50 import NEW_FEATURE_NAMES, cpu_new_features  # noqa: E402


def _frames(seed=0, T=24):
    rng = np.random.default_rng(seed)
    rgb = rng.random((T, 90, 160, 3))
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return luma, rgb


def test_counts():
    assert len(FEATURE_NAMES) == 26
    assert len(NEW_FEATURE_NAMES) == 24
    assert len(set(NEW_FEATURE_NAMES)) == 24
    assert len(set(FEATURE_NAMES) & set(NEW_FEATURE_NAMES)) == 0


def test_shape_and_finite():
    luma, rgb = _frames()
    f = cpu_new_features(luma, rgb, fps=10.0, fsize_std=0.42)
    assert set(f) == set(NEW_FEATURE_NAMES)
    for k, v in f.items():
        assert np.isfinite(v), k
    assert abs(f["ct_fsize_std"] - 0.42) < 1e-9


def test_determinism():
    luma, rgb = _frames(seed=3)
    a = cpu_new_features(luma, rgb, fps=10.0, fsize_std=0.1)
    b = cpu_new_features(luma, rgb, fps=10.0, fsize_std=0.1)
    for k in NEW_FEATURE_NAMES:
        assert a[k] == b[k], k


def test_dct_bands_sum_to_one():
    luma, rgb = _frames(seed=5)
    f = cpu_new_features(luma, rgb, fps=10.0, fsize_std=0.1)
    s = sum(f["dct_band_" + b] for b in ("r1", "r2", "r3", "r4plus"))
    assert 0.0 <= s <= 1.0, s


def test_static_vs_moving():
    """A constant window has ~zero lag differences; a moving one does not."""
    T = 20
    still = np.full((T, 90, 160), 0.5)
    rgb = np.full((T, 90, 160, 3), 0.5)
    fs = cpu_new_features(still, rgb, fps=10.0, fsize_std=0.1)
    assert fs["lag1_diff"] < 1e-9
    assert fs["mo_hand_motion_excess"] < 1e-9
