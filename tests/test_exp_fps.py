"""Tests for the E1 fps / frame-selection code (eval/exp_fps).

No network, no GPU, <30 s. Run: python3 -m pytest tests/test_exp_fps.py -q
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_fps"))

import fps_lib  # noqa: E402
import run_fps  # noqa: E402

SRC_DIR = os.path.join(ROOT, "eval", "exp_fps")


# ---------- paths / port hygiene ----------

def test_no_laptop_paths_left():
    """No laptop home dir and no hardcoded bundle path outside the env default.

    Lines that document a remap (main's `run_e1_02b.py` `paths_note`) are allowed;
    only actual path constants must not appear.
    """
    for fn in sorted(os.listdir(SRC_DIR)):
        if not fn.endswith(".py"):
            continue
        for i, line in enumerate(open(os.path.join(SRC_DIR, fn)), 1):
            if "remapping" in line:
                continue
            assert "/home/parshu" not in line, (fn, i, line)
            assert "projects/jadal" not in line, (fn, i, line)


def test_data_paths_resolve_from_bundle():
    b = fps_lib.BUNDLE
    assert b == os.environ.get("JADAL_BUNDLE", b)
    for p in (fps_lib.MANIFEST, fps_lib.QUEUE1, fps_lib.QUEUE2,
              fps_lib.OOD_LABELS, fps_lib.OOD_LABELS2, fps_lib.SEGMENTS_NPZ):
        assert p.startswith(b), p


def test_results_dir_is_repo_local():
    assert fps_lib.RESULTS.endswith(os.path.join("eval", "results", "exp_fps"))
    assert os.path.isdir(fps_lib.RESULTS)


def test_git_sha():
    s = fps_lib.git_sha()
    assert s and s != "unknown"
    assert len(s) >= 7


# ---------- index rule / selection arms ----------

def test_subsample_idx_is_arange():
    for T, step, n in [(100, 5, 20), (100, 1, 100), (300, 30, 10), (34, 3, 12)]:
        idx = fps_lib.subsample_idx(T, step)
        assert np.array_equal(idx, np.arange(0, T, step))
        assert len(idx) == n


def _synthetic_luma(T=30, H=16, W=16):
    luma = np.full((T, H, W), 0.3)
    for t in range(T):  # a bright block that jumps -> large inter-frame diff
        luma[t, :, (t * 3) % W:(t * 3) % W + 3] = 0.9
    return luma


def test_motion_keep_anchor_budget_and_range():
    luma = _synthetic_luma(30)
    keep = run_fps.motion_keep(luma, budget=20)
    assert 0 in keep, "frame 0 anchor missing"
    assert len(keep) <= 20
    assert keep.min() >= 0 and keep.max() < 30
    assert len(set(keep.tolist())) == len(keep)
    # the highest-diff frame must be selected (block moves every frame -> idx 1..)
    d = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))
    assert int(np.argmax(d)) + 1 in keep


def test_random_keep_deterministic_and_bounded():
    a = run_fps.random_keep(100, 0)
    b = run_fps.random_keep(100, 0)
    c = run_fps.random_keep(100, 1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(a) == 20 and a.min() >= 0 and a.max() < 100


def test_keyframe_positions_parses_ffprobe_csv(monkeypatch):
    csv = "0.000000,I,\n0.100000,B,\n4.600000,I,\n5.000000,P,\n"
    seen = {}

    def fake_run(cmd, **k):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, csv, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ts = run_fps.keyframe_positions("fake.mp4", 4.0, 10.0)
    assert ts == [0.0, 4.6]
    assert "-skip_frame" in seen["cmd"] and "nokey" in seen["cmd"]
    assert "-read_intervals" in seen["cmd"]


def test_keyframe_idx_pads_with_uniform_when_few_iframes(monkeypatch):
    monkeypatch.setattr(run_fps, "keyframe_positions",
                        lambda *a, **k: [0.0, 4.6])
    idx, nif, used = run_fps.keyframe_idx("fake.mp4", 0.0, 100)
    assert nif == 2 and used == 2
    assert len(idx) == 20
    # padding is uniform frames, so the arm still covers the window
    assert idx.max() >= 90


def test_keyframe_idx_uses_iframes_when_enough(monkeypatch):
    ts = [i * 0.4 for i in range(25)]  # 25 I-frames in 10 s
    monkeypatch.setattr(run_fps, "keyframe_positions", lambda *a, **k: ts)
    idx, nif, used = run_fps.keyframe_idx("fake.mp4", 0.0, 100)
    assert nif == 25 and used == 20 and len(idx) == 20
    assert idx.min() >= 0 and idx.max() < 100


def test_keyframe_idx_ignores_iframes_outside_window(monkeypatch):
    monkeypatch.setattr(run_fps, "keyframe_positions",
                        lambda *a, **k: [500.0, 501.0, 1000.0])
    idx, nif, used = run_fps.keyframe_idx("fake.mp4", 0.0, 100)
    assert nif == 0 and used == 0 and len(idx) == 20


# ---------- eval helpers ----------

def test_jaccard():
    assert fps_lib.jaccard([0, 1, 2], [0, 1, 2]) == 1.0
    assert fps_lib.jaccard([0, 1], [2, 3]) == 0.0
    assert fps_lib.jaccard([0, 1, 2, 3], [2, 3, 4, 5]) == pytest.approx(2 / 6)


def test_pareto_front():
    pts = [(0.80, 100), (0.78, 50), (0.70, 200), (0.80, 200)]
    # (0.70,200) dominated by (0.78,50) and (0.80,100); (0.80,200) dominated by (0.80,100)
    assert fps_lib.pareto_front(pts) == [0, 1]


def test_clip_bootstrap_ci_shape():
    rng = np.random.default_rng(0)
    y = np.array([1] * 60 + [0] * 60)
    p = np.concatenate([rng.uniform(0.5, 1, 60), rng.uniform(0, 0.5, 60)])
    clips = np.array([f"c{i % 6}" for i in range(120)])
    lo, hi = fps_lib.clip_bootstrap_ci(y, p, clips, B=200)
    assert 0.5 <= lo <= hi <= 1.0


# ---------- decode determinism (needs the bundle; skipped otherwise) ----------

@pytest.mark.skipif(not os.path.exists(fps_lib.MANIFEST),
                    reason="IndEgo bundle not present")
def test_decode_deterministic_and_shaped():
    it = fps_lib.load_manifest()[0]
    if not os.path.exists(it["path"]):
        pytest.skip("clip missing")
    rgb1, luma1, _ = fps_lib.decode_window(it["path"], it["start_s"], n_frames=12)
    rgb2, luma2, _ = fps_lib.decode_window(it["path"], it["start_s"], n_frames=12)
    assert rgb1.shape[1:] == (fps_lib.H, fps_lib.W, 3)
    assert luma1.shape == rgb1.shape[:3]
    assert np.array_equal(rgb1, rgb2)
    assert np.array_equal(luma1, luma2)


@pytest.mark.skipif(not os.path.exists(fps_lib.QUEUE1),
                    reason="OOD queue not present")
def test_ood_queue_has_original_1080p_sources():
    segs = json.load(open(fps_lib.QUEUE1))["segments"]
    assert len(segs) == 60
    assert all(s.get("source_path") for s in segs)
    assert all(os.path.exists(s["source_path"]) for s in segs)


@pytest.mark.skipif(not os.path.exists(fps_lib.OOD_LABELS),
                    reason="OOD labels not present")
def test_ood_labels_are_human_51_9():
    lab = json.load(open(fps_lib.OOD_LABELS))["latest"]
    assert len(lab) == 60
    bad = sum(1 for v in lab.values() if v["verdict"] == "bad")
    assert bad == 9


# ---------- docs must trace to the results JSONs (PR #88 review F1-F6) ----------

RDIR = fps_lib.RESULTS
DOC = os.path.join(ROOT, "docs", "reports", "exp_fps_pod.md")
REGISTRY = os.path.join(ROOT, "experiments", "registry.jsonl")


def _res(name):
    return json.load(open(os.path.join(RDIR, name)))


def _latest_registry():
    """registry.jsonl is append-only; the latest line per id wins (as in exp.py)."""
    d = {}
    for line in open(REGISTRY):
        if line.strip():
            e = json.loads(line)
            d.setdefault(e["id"], {}).update(e)
    return d


def test_registry_latest_lines_cite_results_json():
    """F1/F2: the live registry cost numbers must exist in the results JSONs."""
    reg = _latest_registry()
    pareto, e103, e104 = _res("pareto.json"), _res("E1.03.json"), _res("E1.04.json")
    c_ood = pareto["clean_bench_ood"]["30"]
    c_ind = pareto["clean_bench_indego"]["2"]
    cost103, cost104 = reg["E1.03"]["cost"], reg["E1.04"]["cost"]
    assert f'{c_ood["decode_ms"]:.1f}' in cost103            # clean OOD decode ms/win
    assert f'{c_ood["feat_ms"]:.1f}' in cost103              # clean OOD feat ms/win @30fps
    assert f'{e103["decode_ms_mean_300f"]:.1f}' in cost103   # loaded shard run
    assert f'{c_ind["decode_ms"]:.1f}' in cost104            # clean in-domain decode
    assert f'{c_ind["feat_ms"]:.1f}' in cost104              # clean 20-frame feature cost
    assert f'{e104["decode_ms_mean"]:.1f}' in cost104        # loaded shard run
    assert f'{pareto["clean_vs_shard_ratio"]["range"][0]:.2f}x' in cost103


def _retracted_only(txt, needle):
    """`needle` may only appear on lines that mark it as retracted/superseded."""
    ctx = re.compile(r"F\d|retract|unsupported|supersed|earlier revision|removed|no results JSON|not in this repo")
    bad = [l.strip() for l in txt.splitlines() if needle in l and not ctx.search(l)]
    assert not bad, (needle, bad)


def test_registry_summaries_quote_results_json():
    """F3/F7: the numbers inside the live summaries come from the result files."""
    reg = _latest_registry()
    e103, e104 = _res("E1.03.json"), _res("E1.04.json")
    s103, s104 = reg["E1.03"]["summary"], reg["E1.04"]["summary"]
    pf = e103["per_feature_signed_auroc"]
    assert f'{pf["1"]["rppg_pulse"]:.4f}' in s103            # 0.5000 (not 0.2854)
    assert f'{pf["2"]["rppg_pulse"]:.4f}' in s103            # 0.2854 stated as the 2fps value
    for r in e103["frozen_rows"]:
        if r["fps"] in (30, 1):
            assert f'{r["frozen_auroc"]:.4f}' in s103
    rows = {r["strategy"]: r for r in e104["rows"]}
    for arm in ("uniform20", "motion20"):
        for b in rows[arm]["ci95"]:
            assert f"{b:.3f}" in s104, (arm, b)
    # F7 sanity note: both CI widths stated, to bootstrap rounding
    m = re.search(r"\(width ([\d.]+) vs ([\d.]+)\)", s104)
    assert m, "CI-width sanity note missing from the E1.04 verdict"
    w = {a: r["ci95"][1] - r["ci95"][0] for a, r in rows.items() if "ci95" in r}
    assert float(m.group(1)) == pytest.approx(w["motion20"], abs=0.001)
    assert float(m.group(2)) == pytest.approx(w["uniform20"], abs=0.001)


def test_keyframe_arm_is_invalid_and_withheld():
    """C1: a keyframe arm that pads every window back to uniform must not report AUROC."""
    d = _res("E1.04.json")
    ki = d["keyframe_info"]
    row = {r["strategy"]: r for r in d["rows"]}["keyframe"]
    if ki["n_windows_iframes_lt_budget"] > 0:
        assert row.get("invalid") is True
        assert "auroc" not in row and "ci95" not in row and "bal_acc" not in row
        assert row["withheld"]["auroc"] == pytest.approx(0.7798, abs=1e-4)
        assert d["invalid_arms"]["keyframe"]["reason"]
        doc = open(DOC).read()
        assert "keyframe-only 0.7798" not in doc
        assert "INVALID" in doc


def test_report_per_feature_table_matches_e1_03():
    """F3: every cell of the report's per-feature table equals E1.03.json."""
    pf = _res("E1.03.json")["per_feature_signed_auroc"]
    n = 0
    for line in open(DOC):
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0].strip("`") not in pf["30"]:
            continue
        name = cells[0].strip("`")
        for col, fps in ((1, "30"), (2, "10"), (3, "1")):
            assert float(cells[col].strip("*")) == pf[fps][name], (name, fps)
        v = [pf[f][name] for f in ("30", "15", "10", "5", "3", "2", "1")]
        assert float(cells[4].strip("*")) == pytest.approx(round(max(v) - min(v), 4)), name
        n += 1
    assert n >= 12, n


def test_report_cost_tables_match_pareto():
    """F1/F2/F5: the clean cost numbers quoted in the report are in pareto.json."""
    txt = open(DOC).read()
    pareto = _res("pareto.json")
    c_ind = pareto["clean_bench_indego"]["2"]
    c_ood30 = pareto["clean_bench_ood"]["30"]
    assert f'{c_ind["decode_ms"]:.0f}' in txt          # 112 ms/win in-domain
    assert f'{c_ind["feat_ms"]:.0f}' in txt            # 54 (53.7) ms/win at 20 frames
    assert f'{c_ood30["decode_ms"]:.0f}' in txt        # 2900 ms/win OOD
    assert f'{c_ood30["feat_ms"]:.0f}' in txt          # 843 ms/win OOD at 30 fps
    _retracted_only(txt, "decode 129 ms")              # F2 stale value, retracted
    _retracted_only(txt, "same to 4 dp")               # unsupported laptop claim, retracted
    for k, v in pareto["feat_fit"].items():
        assert f'{v["slope_ms_per_frame"]:.2f}' in txt, k


def test_report_has_no_unsupported_shard_multiplier():
    """F5: the shard/clean multiplier is read from the JSON, not typed by hand."""
    txt = open(DOC).read()
    pareto = _res("pareto.json")
    lo, hi = pareto["clean_vs_shard_ratio"]["range"]
    assert f"{lo:.1f}\u2013{hi:.1f}\u00d7" in txt      # e.g. 2.2-8.0x
    assert "2-4x" not in txt
    assert "2-4x" not in pareto["cost_note"]
    # the old, unsupported wording may only survive where it is retracted (F5)
    _retracted_only(txt, "2\u20134\u00d7")
    assert "clean_vs_shard_ratio" in txt              # points at the JSON block
    for k in ("ood_decode", "e104_decode", "indego_decode"):
        v = pareto["clean_vs_shard_ratio"][k]
        assert f"{v:.2f}\u00d7" in txt, k


def test_report_states_motion_energy_is_not_comparable_across_fps():
    """F4: the mean |frame diff| caveat is stated next to the fps table."""
    txt = open(DOC).read()
    fc = _res("e1_01.json")["fps_comparability"]
    assert any("motion_energy" in x for x in fc["NOT_comparable_across_fps"])
    assert "motion_energy" in txt
    assert "NOT comparable across fps" in txt
    assert "mean |frame diff|" in txt
    assert "fps_comparability" in txt


def test_run_all_regenerates_the_committed_pareto_command():
    """F6: one command (run_all.sh) reproduces the committed pareto.json bench."""
    pareto = _res("pareto.json")
    sh = open(os.path.join(SRC_DIR, "run_all.sh")).read()
    spread = pareto["bench_spread"]
    assert f'--benches {spread["n_bench_runs"]}' in sh
    assert f'--benches {spread["n_bench_runs"]}' in pareto["command"]
    assert f'--reps {spread["n_reps_each"]}' in pareto["command"]
    assert "pareto.py --reps 3 --benches 3" in sh


def test_clean_vs_shard_ratio_is_recomputable_from_the_results():
    """F5: pareto.json's ratio block is arithmetic on the other results files."""
    pareto = _res("pareto.json")
    e101, e103, e104 = _res("e1_01.json"), _res("E1.03.json"), _res("E1.04.json")
    r = pareto["clean_vs_shard_ratio"]
    ci, co = pareto["clean_bench_indego"], pareto["clean_bench_ood"]
    arm = e104["cost"]["feat_ms_per_window_by_arm"]
    want = {
        "indego_decode": e101["rows"][0]["decode_ms_mean"] / ci["10"]["decode_ms"],
        "indego_feat_10fps": e101["rows"][0]["feat_ms_mean"] / ci["10"]["feat_ms"],
        "indego_feat_1fps": e101["rows"][-1]["feat_ms_mean"] / ci["1"]["feat_ms"],
        "ood_decode": e103["decode_ms_mean_300f"] / co["30"]["decode_ms"],
        "ood_feat_30fps": e103["frozen_rows"][0]["feat_ms_mean"] / co["30"]["feat_ms"],
        "ood_feat_1fps": e103["frozen_rows"][-1]["feat_ms_mean"] / co["1"]["feat_ms"],
        "e104_decode": e104["decode_ms_mean"] / ci["2"]["decode_ms"],
        "e104_feat_uniform20": arm["uniform20"] / ci["2"]["feat_ms"],
        "e104_feat_keyframe": arm["keyframe"] / ci["2"]["feat_ms"],
    }
    for k, v in want.items():
        assert r[k] == pytest.approx(round(v, 2), abs=0.011), k
    vals = sorted(round(v, 2) for v in want.values())
    assert r["range"] == [vals[0], vals[-1]]


def test_results_jsons_carry_provenance_and_experiment_fields():
    """Every committed results JSON: experiment id, device, command, git_sha."""
    for name, exp in (("E1.03.json", "E1.03"), ("E1.04.json", "E1.04"),
                      ("e1_01.json", "E1.01"), ("e1_02.json", "E1.02"),
                      ("pareto.json", "E1.cost")):
        d = _res(name)
        assert d["experiment"] == exp, name
        assert d.get("device") and d.get("command") and d.get("git_sha"), name
        assert d["git_sha"] != "unknown", name


# --- pod: make bare module-level `test_*` functions visible to `unittest discover` ---
# `unittest` collects only TestCase subclasses; the functions above were written pytest-style and were
# therefore silently skipped by `python -m unittest discover -s tests`. This hook adapts them.
def load_tests(loader, tests, pattern):  # noqa: D103
    import inspect as _inspect
    import unittest as _u

    class _Missing:
        pass

    class _MonkeyPatch:
        """Minimal pytest `monkeypatch` shim: setattr() only (all this module needs), undone per test."""

        def __init__(self):
            self._undo = []

        def setattr(self, target, name, value=None, raising=True):
            if isinstance(target, str):  # "module.attr" form
                mod, _, attr = target.rpartition(".")
                target, name, value = __import__(mod, fromlist=[attr]), attr, name
            old = getattr(target, name, _Missing)
            self._undo.append((target, name, old))
            setattr(target, name, value)

        def undo(self):
            while self._undo:
                target, name, old = self._undo.pop()
                if old is _Missing:
                    delattr(target, name)
                else:
                    setattr(target, name, old)

    class _ModuleTests(_u.TestCase):
        pass

    for _name, _fn in sorted(list(globals().items())):
        if not (_name.startswith("test_") and callable(_fn)):
            continue
        if "monkeypatch" in _inspect.signature(_fn).parameters:

            def _wrap(fn):
                def _test(self):
                    mp = _MonkeyPatch()
                    try:
                        fn(monkeypatch=mp)
                    finally:
                        mp.undo()

                return _test

            setattr(_ModuleTests, _name, _wrap(_fn))
        else:
            setattr(_ModuleTests, _name, staticmethod(_fn))
    return _u.TestSuite([loader.loadTestsFromTestCase(_ModuleTests)])

