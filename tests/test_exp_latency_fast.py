"""Tests for L3.01 (eval/exp_latency_fast): adopted-optimization flags.

Fast, no network, no GPU: the decode branch is tested by capturing the ffmpeg
command line instead of running it. The integration run is exercised elsewhere.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_latency_fast"))

import run_fast  # noqa: E402


def test_read_opts_defaults_match_frozen_baseline(monkeypatch):
    for k in ("JADAL_PIXEL_FPS", "JADAL_EG2_FPS", "JADAL_NATIVE_FPS",
              "JADAL_EARLY_EXIT_S", "JADAL_SKIP_MOSAIC"):
        monkeypatch.delenv(k, raising=False)
    o = run_fast.read_opts()
    # defaults must reproduce the L2.01 baseline exactly
    assert o == {"pixel_fps": 0.0, "eg2_fps": 2.0, "native_fps": 0.8,
                 "early_exit_s": 0.0, "skip_mosaic": False}


def test_read_opts_env_overrides(monkeypatch):
    monkeypatch.setenv("JADAL_PIXEL_FPS", "3.33")
    monkeypatch.setenv("JADAL_EG2_FPS", "1")
    monkeypatch.setenv("JADAL_EARLY_EXIT_S", "4")
    monkeypatch.setenv("JADAL_SKIP_MOSAIC", "1")
    o = run_fast.read_opts()
    assert o["pixel_fps"] == pytest.approx(3.33)
    assert o["eg2_fps"] == 1.0
    assert o["early_exit_s"] == 4.0
    assert o["skip_mosaic"] is True


def test_read_opts_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("JADAL_PIXEL_FPS", "not-a-number")
    assert run_fast.read_opts()["pixel_fps"] == 0.0


class _FakeProc:
    def __init__(self, cmd, **_kw):
        self.cmd = cmd
        self.returncode = 0
        self.stdout = b""
        self.stderr = _FakeStderr()

    def wait(self, timeout=None):
        return 0


class _FakeStderr:
    def read(self):
        return b""


def _capture_decode_cmd(monkeypatch, **kw):
    import subprocess
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    # _drain threads will read nothing from the real pipe fds; that is fine,
    # we only assert on the command line.
    try:
        run_fast.decode_three_branch_cfg("x.mp4", 0.0, 4.0, **kw)
    except Exception:  # noqa: BLE001  (no real ffmpeg output)
        pass
    return captured.get("cmd", [])


def _filter_complex(cmd):
    i = cmd.index("-filter_complex")
    return cmd[i + 1]


def test_decode_defaults_match_frozen_graph(monkeypatch):
    fc = _filter_complex(_capture_decode_cmd(monkeypatch))
    assert "[a]scale_cuda=160:90" in fc          # pixel: every frame
    assert "[b]fps=2,scale_cuda=512:288" in fc   # EG2: 2 fps
    assert "[c]fps=0.8,hwdownload" in fc         # native: 0.8 fps


def test_decode_adopted_optimizations_insert_fps(monkeypatch):
    fc = _filter_complex(_capture_decode_cmd(
        monkeypatch, pixel_fps=3.33, eg2_fps=1.0, native_fps=0.8))
    assert "[a]fps=3.33,scale_cuda=160:90" in fc
    assert "[b]fps=1,scale_cuda=512:288" in fc
    assert "[c]fps=0.8,hwdownload" in fc


def test_outdir_is_fast():
    assert run_fast.OUTDIR.endswith(os.path.join("results", "exp_latency_fast"))
