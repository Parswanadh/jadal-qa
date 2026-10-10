"""Unit tests for the gold-label server (video tooling, stdlib page).

Run: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m pytest \\
         tools/gold_label/test_label_server.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

SERVER_PATH = Path(__file__).resolve().parent / "label_server.py"


def load_server():
    spec = importlib.util.spec_from_file_location("gold_label_server",
                                                  SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


srv = load_server()


@pytest.fixture()
def manifest(tmp_path):
    segs = [{"segment_id": f"g{i:03d}", "clip": "ecu_full",
             "path": "/clip/ecu.mp4",
             "start_s": float(i * 10), "end_s": float(i * 10 + 10)}
            for i in range(5)]
    man = tmp_path / "segments.json"
    man.write_text(json.dumps({"n": 5, "segments": segs}))
    return man


@pytest.fixture()
def store(manifest, tmp_path):
    return srv.Store(manifest, tmp_path / "labels.json")


# --- Range handling ------------------------------------------------------


@pytest.mark.parametrize("header,size,expected", [
    ("bytes=0-99", 1000, (0, 99)),
    ("bytes=100-", 1000, (100, 999)),
    ("bytes=-50", 1000, (950, 999)),
    ("bytes=0-", 1000, (0, 999)),
    ("bytes=900-9999", 1000, (900, 999)),  # clamped
    (None, 1000, None),
    ("bytes=1000-2000", 1000, None),  # unsatisfiable
    ("bytes=5-3", 1000, (5, 3)),  # degenerate but parseable; server sends 0 bytes
    ("bytes=0-1,3-4", 1000, None),  # multipart unsupported
    ("items=0-10", 1000, None),
    ("bytes=", 1000, None),
    ("bytes=-0", 1000, None),
])
def test_parse_range(header, size, expected):
    assert srv.parse_range(header, size) == expected


# --- Label saving: atomic, resume-safe, undo ------------------------------


def test_add_good_and_bad(store):
    e1 = store.add("g000", "good", None, "")
    assert e1["verdict"] == "good" and e1["reason"] is None
    e2 = store.add("g001", "bad", "BLUR", "note")
    assert e2["reason"] == "BLUR"
    assert set(store.latest) == {"g000", "g001"}
    payload = json.loads(store.labels_path.read_text())
    assert len(payload["history"]) == 2
    assert set(payload["latest"]) == {"g000", "g001"}


def test_bad_needs_reason(store):
    with pytest.raises(ValueError):
        store.add("g000", "bad", None, "")
    with pytest.raises(ValueError):
        store.add("g000", "bad", "NOPE", "")
    assert not store.history  # rejected writes leave no trace


def test_unknown_segment(store):
    with pytest.raises(KeyError):
        store.add("zzz", "good", None, "")


def test_relabels_keep_history(store):
    store.add("g000", "good", None, "")
    store.add("g000", "bad", "TOO_DARK", "")
    assert len(store.history) == 2  # append-only
    assert store.latest["g000"]["verdict"] == "bad"


def test_undo_restores_previous(store):
    store.add("g000", "good", None, "")
    store.add("g000", "bad", "TOO_DARK", "")
    undone = store.undo()
    assert undone["segment_id"] == "g000"
    assert undone["verdict"] == "bad"
    assert store.latest["g000"]["verdict"] == "good"
    undone = store.undo()
    assert undone["verdict"] == "good"
    assert "g000" not in store.latest
    assert store.undo() is None


def test_resume_preserves_existing_entries(manifest, tmp_path):
    labels_path = tmp_path / "labels.json"
    first = srv.Store(manifest, labels_path)
    first.add("g000", "good", None, "keep me")
    first.add("g001", "bad", "SHAKY", "")
    # Reopen: everything must still be there.
    second = srv.Store(manifest, labels_path)
    assert set(second.latest) == {"g000", "g001"}
    assert second.latest["g000"]["note"] == "keep me"
    assert len(second.history) == 2
    # New saves append, never rewrite old entries.
    second.add("g002", "good", None, "")
    payload = json.loads(labels_path.read_text())
    assert [e["segment_id"] for e in payload["history"]] == \
        ["g000", "g001", "g002"]
    assert payload["history"][0]["note"] == "keep me"


def test_atomic_save_no_tmp_leftovers(store, tmp_path):
    store.add("g000", "good", None, "")
    leftovers = list(tmp_path.glob("labels_*.tmp"))
    assert leftovers == []


def test_state_counts(store):
    store.add("g000", "good", None, "")
    store.add("g001", "bad", "BLUR", "")
    st = store.state()
    assert (st["n"], st["labeled"], st["n_good"], st["n_bad"]) == (5, 2, 1, 1)


def test_median_seconds_per_label(store):
    assert store.median_seconds_per_label() is None  # nothing yet
    store.history.append({"ts": "2026-01-01T00:00:00+00:00",
                          "segment_id": "g000", "verdict": "good",
                          "reason": None, "note": ""})
    store.history.append({"ts": "2026-01-01T00:00:20+00:00",
                          "segment_id": "g001", "verdict": "good",
                          "reason": None, "note": ""})
    store.history.append({"ts": "2026-01-01T00:00:50+00:00",
                          "segment_id": "g002", "verdict": "bad",
                          "reason": "BLUR", "note": ""})
    assert store.median_seconds_per_label() == pytest.approx(25.0)


def test_concurrent_adds_stay_consistent(store):
    def work(k):
        store.add(f"g{k:03d}", "good", None, "")

    threads = [threading.Thread(target=work, args=(k,)) for k in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.history) == 5
    assert len(store.latest) == 5


# --- Step-overlap lookup ---------------------------------------------------


def make_via(options, segments):
    return {
        "attribute": {"1": {"aname": "TEMPORAL-SEGMENTS",
                            "options": {str(i): name
                                        for i, name in enumerate(options)}}},
        "metadata": {f"k{i}": {"vid": "2", "flg": 0, "z": [s, e], "xy": [],
                               "av": {"1": str(options.index(name))}}
                     for i, (s, e, name) in enumerate(segments)},
    }


def test_parse_via_steps():
    via = make_via(["01_inspect_and_clean", "02_attach_ram_chip"],
                   [(8.79, 24.12, "01_inspect_and_clean"),
                    (24.12, 36.36, "02_attach_ram_chip")])
    assert srv.parse_via_steps(via) == [
        (8.79, 24.12, "01_inspect_and_clean"),
        (24.12, 36.36, "02_attach_ram_chip")]


def test_steps_in_window_overlap():
    index = {"/clip/ecu.mp4": [(0.0, 8.0, "step_a"),
                               (8.0, 25.0, "step_b"),
                               (30.0, 40.0, "step_c"),
                               (8.0, 25.0, "step_b")]}  # duplicate entry
    # Window straddling a boundary finds both steps, deduped.
    assert srv.steps_in_window(index, "/clip/ecu.mp4", 5.0, 15.0) == \
        ["step_a", "step_b"]
    # Touching the end edge does not count as overlap.
    assert srv.steps_in_window(index, "/clip/ecu.mp4", 25.0, 30.0) == []
    # Unknown clip -> no steps.
    assert srv.steps_in_window(index, "/clip/other.mp4", 0.0, 10.0) == []


def test_task_line_assembly_between_steps():
    index = {"/clip/ecu.mp4": [(100.0, 200.0, "02_attach_ram_chip")]}
    seg = {"segment_id": "g000", "clip": "ecu_full", "path": "/clip/ecu.mp4",
           "start_s": 0.0, "end_s": 10.0}
    line = srv.task_line(seg, index)
    assert "ECU assembly" in line and "between" in line


def test_task_line_assembly_with_step():
    index = {"/clip/ecu.mp4": [(100.0, 200.0, "02_attach_ram_chip")]}
    seg = {"segment_id": "g010", "clip": "ecu_full", "path": "/clip/ecu.mp4",
           "start_s": 105.0, "end_s": 115.0}
    line = srv.task_line(seg, index)
    assert "02_attach_ram_chip" in line


def test_task_line_logistics_and_unknown():
    seg = {"segment_id": "g000", "clip": "pallet",
           "path": "/x/3_Logistics_Organization/mp4_480/p.mp4",
           "start_s": 0.0, "end_s": 10.0}
    assert "pallet transport" in srv.task_line(seg, {})
    seg2 = {"segment_id": "g001", "clip": "mystery",
            "path": "/x/y.mp4", "start_s": 0.0, "end_s": 10.0}
    assert "unknown" in srv.task_line(seg2, {})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
