"""E7.01 cost-model tests: pure arithmetic + end-to-end model assembly.

< 30 s, no network, no GPU, no decode. Synthetic timings only — the tests check
the MODEL, not the measured numbers.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.exp_cost import run_e7 as E  # noqa: E402


def synth_measure():
    dec = {}
    for fam, rate, native in (("indego_480p10_h264", 100.0, 10.0),
                              ("ego10k_1080p30_hevc", 60.0, 30.0)):
        for hw in ("cpu", "nvdec"):
            r = rate if hw == "cpu" else rate * 4
            by = {}
            for dur in (4.0, 10.0):
                frames = dur * native
                by[str(dur)] = {"fps": r, "n_windows": 3,
                                "frames_mean": frames,
                                "wall_s_mean": frames / r,
                                "wall_s_ci95": [0.1, 0.2]}
            dec[f"{fam}__{hw}"] = {"by_duration_s": by}
    return {
        "decode": dec,
        "features_pixel26": {"n_windows": 12,
                             "features_ms_per_window": {
                                 "10@10": {"mean": 100.0, "ci95": [90, 110], "n": 12},
                                 "5@10": {"mean": 60.0, "ci95": [50, 70], "n": 12},
                                 "2@10": {"mean": 30.0, "ci95": [25, 35], "n": 12},
                                 "10@4": {"mean": 45.0, "ci95": [40, 50], "n": 12},
                                 "5@4": {"mean": 30.0, "ci95": [25, 35], "n": 12},
                                 "2@4": {"mean": 15.0, "ci95": [12, 18], "n": 12}}},
        "mosaic_build": {"status": "measured", "layout_ms_mean": 30.0,
                         "png_encode_ms_mean": 20.0, "n_windows": 6},
        "embedding_eg2": {"inputs": {
            "tile_224x224": {"ms_per_image_mean": 5.0,
                             "ms_per_image_ci95": [4.0, 6.0], "n_timed": 8},
            "mosaic_3x3_680x680_real": {"ms_per_image_mean": 200.0,
                                        "ms_per_image_ci95": [180, 220],
                                        "n_timed": 8}}},
        "head_logistic": {"ms_per_predict": 0.1},
    }


def test_video_hours_per_wall_hour_formula():
    # 1000 ms/window -> 3600 windows/h; k=2 covers 20 s each -> 20 h/h
    assert E.video_hours_per_wall_hour(1000.0, 2) == pytest.approx(20.0)
    assert E.video_hours_per_wall_hour(1000.0, 1) == pytest.approx(10.0)
    assert E.video_hours_per_wall_hour(0, 1) is None


def test_decode_ms_for_scales_with_native_frames():
    m = synth_measure()
    dec = m["decode"]
    # IndEgo: 10 s @10fps = 100 frames at 100 fps -> 1000 ms
    assert E.decode_ms_for(dec, "indego_480p10_h264", "CPU", 10.0) == pytest.approx(1000.0)
    # 4 s early exit -> 40 frames -> 400 ms
    assert E.decode_ms_for(dec, "indego_480p10_h264", "CPU", 4.0) == pytest.approx(400.0)
    # Ego10K: 10 s @30fps = 300 frames at 60 fps -> 5000 ms
    assert E.decode_ms_for(dec, "ego10k_1080p30_hevc", "CPU", 10.0) == pytest.approx(5000.0)
    # NVDEC 4x faster
    assert E.decode_ms_for(dec, "ego10k_1080p30_hevc", "NVDEC", 10.0) == pytest.approx(1250.0)
    # unknown family / missing measurement fails loudly / returns None
    with pytest.raises(KeyError):
        E.decode_ms_for(dec, "nope", "CPU", 10.0)
    assert E.decode_ms_for({}, "indego_480p10_h264", "CPU", 10.0) is None


def test_decode_fit_negative_slope_falls_back():
    # real pod shape: NVDEC on the small 480p clip is slower than CPU and the
    # 4 s run is dominated by session startup -> negative slope -> fallback
    dec = {"indego_480p10_h264__nvdec": {"by_duration_s": {
        "4.0": {"wall_s_mean": 1.3667, "frames_mean": 40.0},
        "10.0": {"wall_s_mean": 1.1128, "frames_mean": 100.0}}}}
    fit = E.decode_fit(dec, "indego_480p10_h264", "NVDEC")
    assert fit["usable_marginal_fit"] is False
    assert fit["fps_marginal"] is None
    assert fit["ms_per_frame"] == pytest.approx(1.1128 / 100.0 * 1000.0, rel=1e-6)
    assert E.decode_fit({}, "indego_480p10_h264", "CPU") is None


def test_decode_fit_excludes_constant_startup():
    # wall = 2 s startup + 10 ms/frame for both durations
    dec = {"ego10k_1080p30_hevc__cpu": {"by_duration_s": {
        "4.0": {"wall_s_mean": 2.0 + 120 * 0.010, "frames_mean": 120.0},
        "10.0": {"wall_s_mean": 2.0 + 300 * 0.010, "frames_mean": 300.0}}}}
    fit = E.decode_fit(dec, "ego10k_1080p30_hevc", "CPU")
    assert fit["ms_per_frame"] == pytest.approx(10.0)
    assert fit["startup_ms"] == pytest.approx(2000.0)
    assert fit["fps_marginal"] == pytest.approx(100.0)
    # 4 s span -> 120 native frames * 10 ms (startup NOT charged per window)
    assert E.decode_ms_for(dec, "ego10k_1080p30_hevc", "CPU", 4.0) == pytest.approx(1200.0)


def test_boot_ci_contains_mean():
    ci = E.boot_ci([1.0, 2.0, 3.0, 4.0, 5.0], B=200)
    assert ci[0] < 3.0 < ci[1]
    assert E.boot_ci([1.0]) == [None, None]


def test_stage_model_end_to_end(tmp_path, monkeypatch):
    mj = tmp_path / "measure.json"
    oj = tmp_path / "E7.01.json"
    json.dump(synth_measure(), open(mj, "w"))
    monkeypatch.setattr(E, "MEASURE_JSON", str(mj))
    monkeypatch.setattr(E, "OUT_JSON", str(oj))
    E.stage_model(None)
    out = json.load(open(oj))
    names = sorted({r["profile"] for r in out["rows"]})
    assert names == ["blurry_lowlight", "clear_stationary", "default",
                     "factory_1080p30", "walking_logistics"]
    assert len(out["rows"]) == 2 * len(names)          # CPU + NVDEC per profile
    assert out["not_measured"] == []
    assert out["label_source"].startswith("n/a")
    assert out["device"] == "A40"
    assert out["command"] and out["git_sha"]
    for r in out["rows"]:
        assert r["video_hours_per_wall_hour"] > 0
        assert r["ms_per_window"]["total"] > 0
        assert 0.0 <= r["decode_share_of_total"] <= 1.0

    # hand-checked row: default = fps10, k=1, no exit, both features, mosaic
    d = [r for r in out["rows"] if r["profile"] == "default"
         and r["decode_path"] == "CPU"][0]
    assert d["ms_per_window"]["decode"] == pytest.approx(1000.0)
    assert d["ms_per_window"]["features_pixel26"] == pytest.approx(100.0)
    assert d["ms_per_window"]["embedding_eg2"] == pytest.approx(500.0)   # 100 frames
    assert d["ms_per_window"]["mosaic_evidence"] == pytest.approx(250.0)  # 30+20+200
    assert d["ms_per_window"]["total"] == pytest.approx(1850.1)
    assert d["video_hours_per_wall_hour"] == pytest.approx(
        E.video_hours_per_wall_hour(1850.1, 1), abs=0.01)

    # factory_1080p30 decodes the 1080p HEVC family, NVDEC is cheaper there
    f_cpu = [r for r in out["rows"] if r["profile"] == "factory_1080p30"
             and r["decode_path"] == "CPU"][0]
    f_nv = [r for r in out["rows"] if r["profile"] == "factory_1080p30"
            and r["decode_path"] == "NVDEC"][0]
    assert f_cpu["source_family"] == "ego10k_1080p30_hevc"
    assert f_nv["video_hours_per_wall_hour"] > f_cpu["video_hours_per_wall_hour"]

    # the early-exit profile observes 6 s, not 10
    cs = [r for r in out["rows"] if r["profile"] == "clear_stationary"][0]
    assert cs["observed_s"] == 6.0


def test_reference_rows_use_the_same_formula(tmp_path, monkeypatch):
    mj = tmp_path / "m.json"
    json.dump(synth_measure(), open(mj, "w"))
    monkeypatch.setattr(E, "MEASURE_JSON", str(mj))
    monkeypatch.setattr(E, "OUT_JSON", str(tmp_path / "o.json"))
    E.stage_model(None)
    out = json.load(open(tmp_path / "o.json"))
    adopted = [r for r in out["reference_rows"]
               if r["profile"] == "E3_adopted_every2_exit4_pixel26"][0]
    baseline = [r for r in out["reference_rows"]
                if r["profile"] == "E3_baseline_full_pixel26"][0]
    assert adopted["video_hours_per_wall_hour"] > baseline["video_hours_per_wall_hour"]
    assert out["headline"]["speedup_x"] > 1.0


# --- pod: make bare module-level `test_*` functions visible to `unittest discover` ---
# `unittest` collects only TestCase subclasses; the functions above were written pytest-style and were
# therefore silently skipped by `python -m unittest discover -s tests`. This hook adapts them, including a
# minimal `monkeypatch` shim (only `setattr` is used) so no test is silently dropped.
class _PodMonkeyPatch:
    _MISSING = object()

    def __init__(self):
        self._undo = []

    def setattr(self, target, name=None, value=_MISSING, raising=True):
        if isinstance(target, str):  # pytest-style "module.attr" form
            import importlib

            mod, _, attr = target.rpartition(".")
            target, name, value = importlib.import_module(mod), attr, name
        old = getattr(target, name, self._MISSING)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def undo(self):
        while self._undo:
            target, name, old = self._undo.pop()
            if old is self._MISSING:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
            else:
                setattr(target, name, old)


def _pod_wrap(fn):
    def _test(self):
        import inspect
        import tempfile

        kwargs = {}
        params = inspect.signature(fn).parameters
        mp = _PodMonkeyPatch()
        _tmp = None
        if "monkeypatch" in params:
            kwargs["monkeypatch"] = mp
        if "tmp_path" in params:
            _tmp = tempfile.TemporaryDirectory()
            import pathlib

            kwargs["tmp_path"] = pathlib.Path(_tmp.name)
        try:
            fn(**kwargs)
        finally:
            mp.undo()
            if _tmp is not None:
                _tmp.cleanup()

    return _test


def load_tests(loader, tests, pattern):  # noqa: D103
    import inspect
    import unittest as _u

    class _ModuleTests(_u.TestCase):
        pass

    for _name, _fn in sorted(list(globals().items())):
        if not (_name.startswith("test_") and callable(_fn)):
            continue
        if {"monkeypatch", "tmp_path"} & set(inspect.signature(_fn).parameters):
            setattr(_ModuleTests, _name, _pod_wrap(_fn))
        else:
            setattr(_ModuleTests, _name, staticmethod(_fn))
    return _u.TestSuite([loader.loadTestsFromTestCase(_ModuleTests)])
