"""Fast smoke tests for eval/exp_adapt (no network; < 30 s).

Run: OMP_NUM_THREADS=2 python3 -m pytest tests/test_exp_adapt.py -q
"""
import numpy as np


def test_loaders_shapes():
    from eval.exp_adapt import adaptlib as A
    g = A.load_gold()
    assert g["X26"].shape == (150, 26)
    assert set(np.unique(g["y"])) == {0, 1}
    assert len(set(g["clips"].tolist())) == 7
    o = A.load_ood_pixel()
    assert o["X_native"].shape == (60, 26) and o["X_10fps"].shape == (60, 26)
    assert int((o["y"] == 0).sum()) == 9


def test_auroc_and_features():
    from eval.exp_adapt import adaptlib as A
    g = A.load_gold()
    j = A.FEATURE_NAMES.index("radial_slope")
    a = A.auroc(g["y"], g["X26"][:, j])
    assert 0.5 < a < 0.75, a  # E5.03: 0.633


def test_coral_maps_source_onto_target_cov():
    from eval.exp_adapt import adaptlib as A
    rng = np.random.RandomState(0)
    Xs = rng.randn(200, 5) @ rng.randn(5, 5)
    Xt = rng.randn(200, 5) @ rng.randn(5, 5) * 2 + 1.0
    t = A.Coral().fit(Xs, Xt, direction="s2t")
    Y = t.transform(Xs, "s")
    assert np.allclose(Y.mean(0), Xt.mean(0), atol=1e-6)
    assert Y.shape == Xs.shape


def test_group_norm_removes_group_shift():
    from eval.exp_adapt import adaptlib as A
    X = np.vstack([np.ones((4, 2)), np.ones((4, 2)) * 10.0])
    X[:, 1] = np.arange(8)
    g = np.array([0] * 4 + [1] * 4)
    Xn = A.group_norm_apply(X, g, A.group_norm_fit(X, g))
    assert np.allclose(Xn[g == 0, 0], 0.0, atol=1e-9)
    assert np.allclose(Xn[g == 1, 0], 0.0, atol=1e-9)


def test_results_json_present():
    import json
    import os
    from eval.exp_adapt import adaptlib as A
    for name in ("Z1.01.json", "Z1.02.json", "Z1.03.json"):
        p = os.path.join(A.OUT, name)
        assert os.path.exists(p), p
        assert json.load(open(p))["experiment"] == name[:-5]
