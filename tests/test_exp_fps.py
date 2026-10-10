"""Tests for the E1 fps / frame-selection code (eval/exp_fps).

No network, no GPU, <30 s. Run: python3 -m pytest tests/test_exp_fps.py -q
"""
import json
import os
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
    """No laptop home dir and no hardcoded bundle path outside the env default."""
    for fn in sorted(os.listdir(SRC_DIR)):
        if not fn.endswith(".py"):
            continue
        txt = open(os.path.join(SRC_DIR, fn)).read()
        assert "/home/parshu" not in txt, fn
        assert "projects/jadal" not in txt, fn


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
