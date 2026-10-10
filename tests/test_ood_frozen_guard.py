"""LB4 guard tests: the staged frozen OOD run must fail closed.

Missing labels            -> PENDING (exit 3), nothing scored, results JSON written.
Wrong frozen sha256       -> ABORT   (exit 2), nothing scored.
Wrong label count         -> ABORT   (exit 2).
Single-class labels       -> ABORT   (exit 2).
Scores from another config-> ABORT   (exit 2).
Happy path                -> COMPLETE (exit 0) with hand-checkable metrics.

All fixtures live in tmp_path; the real labels file and the pod bundle are never
written to. No model inference and no network: the happy path consumes a fixture
scores file, the guard paths return before scoring.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNNER = os.path.join(REPO_ROOT, "eval", "exp_ood", "run_ood_frozen.py")
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
RECORDED = "4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31"
GIT_BLOB = "0ede3b83d93ce23dc5b68cd20fb7de0af703ad27"

sys.path.insert(0, REPO_ROOT)
from eval.exp_ood.run_ood_frozen import GuardError, _auroc, build_runtime_config  # noqa: E402


# --------------------------------------------------------------------- helpers
def _sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _write(path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def _frozen_cfg(embed_dir="laptop/eg2"):
    return {
        "name": "ensemble_v1",
        "views": {"eg2": {"embed_model_dir": embed_dir}, "pixel": {"C": 0.1}},
        "fusion": {"weights": [0.5, 0.5]},
        "threshold": 0.5,
        "abstain_band": [0.4, 0.6],
    }


def _make_case(tmp_path, n_good=50, n_bad=50, artifact=None, scores_meta=None):
    """Build artifact/base-config/labels/scores fixtures; return paths + shas."""
    art = tmp_path / "frozen_artifact.json"
    _write(art, artifact if artifact is not None else _frozen_cfg("laptop/eg2"))
    base = tmp_path / "base_config.json"
    _write(base, _frozen_cfg("pod/eg2"))
    labels = {"latest": {}}
    rows = []
    for i in range(n_good + n_bad):
        wid = f"w{i:03d}"
        good = i < n_good
        labels["latest"][wid] = {"segment_id": wid,
                                 "verdict": "good" if good else "bad",
                                 "clip": f"clip{i % 5}"}
        rows.append({"id": wid, "usable_prob": 0.9 if good else 0.1})
    lab = tmp_path / "ood_labels2.json"
    _write(lab, labels)
    sc = tmp_path / "scores_fixture.json"
    payload = {"scores": rows}
    if scores_meta:
        payload.update(scores_meta)
    _write(sc, payload)
    out = tmp_path / "ood_run.json"
    return {"artifact": art, "base": base, "labels": lab, "scores": sc, "out": out,
            "artifact_sha": _sha256(art)}


def _run(case, extra=None, recorded=None, expect_labels=True):
    cmd = [sys.executable, RUNNER,
           "--labels", str(case["labels"]) if expect_labels else str(case["labels"]) + ".absent",
           "--artifact", str(case["artifact"]),
           "--recorded-sha256", recorded or case["artifact_sha"],
           "--base-config", str(case["base"]),
           "--scores", str(case["scores"]),
           "--out", str(case["out"]),
           "--runtime-config", str(case["out"].parent / "runtime_config.json"),
           "--boot", "200", "--seed", "0"]
    if extra:
        cmd += extra
    env = dict(os.environ, OMP_NUM_THREADS="2")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    payload = json.loads(case["out"].read_text()) if case["out"].exists() else None
    return proc, payload


# ------------------------------------------------------------ guard behaviour
def test_missing_labels_is_pending_and_scores_nothing(tmp_path):
    case = _make_case(tmp_path)
    proc, payload = _run(case, expect_labels=False)
    assert proc.returncode == 3, proc.stderr
    assert payload is not None, "PENDING must still write the deliverable"
    assert payload["status"] == "PENDING"
    assert payload["guard"] == "labels_absent"
    assert payload["preregistered"] is True
    assert payload["frozen_sha256"] == case["artifact_sha"]
    assert payload["frozen_sha256_verified"] is True
    assert payload["n"] is None and payload["n_bad"] is None
    # nothing was scored / no metrics exist
    assert "auroc" not in payload
    assert not (case["out"].parent / "runtime_config.json").exists()
    assert "PENDING" in proc.stdout
    assert not case["out"].parent.joinpath("ood_scores2.json").exists()


def test_wrong_recorded_hash_aborts(tmp_path):
    case = _make_case(tmp_path)
    proc, payload = _run(case, recorded="0" * 64)
    assert proc.returncode == 2, proc.stderr
    assert payload["status"] == "ABORT"
    assert payload["guard"] == "frozen_sha256"
    assert payload["frozen_sha256_recorded"] == "0" * 64
    assert payload["frozen_sha256_observed"] == case["artifact_sha"]
    assert "auroc" not in payload
    assert payload["n"] == 100  # labels were delivered; the guard still refused
    assert not (case["out"].parent / "runtime_config.json").exists()


def test_tampered_artifact_aborts(tmp_path):
    case = _make_case(tmp_path)
    good_sha = case["artifact_sha"]
    tampered = _frozen_cfg("laptop/eg2")
    tampered["threshold"] = 0.31  # any byte change must trip the guard
    _write(case["artifact"], tampered)
    proc, payload = _run(case, recorded=good_sha)
    assert proc.returncode == 2, proc.stderr
    assert payload["status"] == "ABORT"
    assert payload["guard"] == "frozen_sha256"
    assert payload["frozen_sha256_observed"] == _sha256(case["artifact"])
    assert payload["frozen_sha256_observed"] != good_sha


def test_wrong_label_count_aborts(tmp_path):
    case = _make_case(tmp_path, n_good=2, n_bad=1)
    proc, payload = _run(case)
    assert proc.returncode == 2, proc.stderr
    assert payload["status"] == "ABORT"
    assert payload["guard"] == "labels_count"
    assert payload["n"] == 3 and payload["n_bad"] == 1
    assert "auroc" not in payload


def test_single_class_labels_aborts(tmp_path):
    case = _make_case(tmp_path, n_good=100, n_bad=0)
    proc, payload = _run(case)
    assert proc.returncode == 2, proc.stderr
    assert payload["status"] == "ABORT"
    assert payload["guard"] == "labels_single_class"


def test_scores_from_other_config_aborts(tmp_path):
    case = _make_case(tmp_path)
    rows = json.loads(case["scores"].read_text())
    rows["config_sha256"] = "deadbeef" * 8
    _write(case["scores"], rows)
    proc, payload = _run(case)
    assert proc.returncode == 2, proc.stderr
    assert payload["status"] == "ABORT"
    assert payload["guard"] == "scores_config_sha256"


# -------------------------------------------------------------- happy path
def test_complete_path_metrics_and_runtime_config(tmp_path):
    case = _make_case(tmp_path, n_good=50, n_bad=50)
    proc, payload = _run(case)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert payload["status"] == "COMPLETE"
    assert payload["preregistered"] is True
    assert payload["n"] == 100 and payload["n_bad"] == 50 and payload["n_good"] == 50
    assert payload["auroc"] == 1.0
    assert payload["balanced_accuracy_at_threshold"] == 1.0
    assert payload["accept_at_threshold"] == 50
    assert payload["reject_at_threshold"] == 50
    assert payload["good_rejected_at_threshold"] == 0
    assert payload["bad_rejected_at_threshold"] == 50
    assert payload["threshold"] == 0.5
    assert payload["frozen_sha256"] == case["artifact_sha"]
    assert payload["frozen_sha256_verified"] is True
    assert payload["frozen_semantics_identical"] is True
    assert payload["runtime_overrides"] == {"views.eg2.embed_model_dir": "pod/eg2"}
    assert payload["auroc_ci95_clip_bootstrap"]["unit"] == "source_clip"
    assert payload["auroc_ci95_clip_bootstrap"]["lo"] == 1.0
    assert payload["auroc_ci95_window_bootstrap_optimistic"]["unit"] == "window"
    assert payload["label_source"].startswith("human")
    assert "ood_labels2.json" in payload["label_source"]
    # the runtime config really is frozen content + the allow-listed path only
    rt_path = case["out"].parent / "runtime_config.json"
    runtime = json.loads(rt_path.read_text())
    assert runtime["views"]["eg2"]["embed_model_dir"] == "pod/eg2"
    assert runtime["threshold"] == 0.5
    sidecar = rt_path.with_name(rt_path.name + ".sha256").read_text().split()[0]
    assert sidecar == hashlib.sha256(json.dumps(runtime, indent=1).encode()).hexdigest()
    assert payload["runtime_config_sha256"] == sidecar


def test_runtime_config_override_is_allowlisted_only(tmp_path):
    frozen = _frozen_cfg("laptop/eg2")
    base = _frozen_cfg("pod/eg2")
    runtime, overrides = build_runtime_config(frozen, base)
    assert overrides == {"views.eg2.embed_model_dir": "pod/eg2"}
    assert runtime["views"]["eg2"]["embed_model_dir"] == "pod/eg2"
    assert runtime["threshold"] == frozen["threshold"]
    # nothing except the allow-listed machine path may leak in from the base config
    leaking = _frozen_cfg("pod/eg2")
    leaking["threshold"] = 0.31
    leaking["views"]["pixel"]["C"] = 999.0
    leaking["fusion"]["weights"] = [1.0, 0.0]
    runtime2, overrides2 = build_runtime_config(frozen, leaking)
    assert runtime2["threshold"] == frozen["threshold"] == 0.5
    assert runtime2["views"]["pixel"]["C"] == 0.1
    assert runtime2["fusion"]["weights"] == [0.5, 0.5]
    assert overrides2 == {"views.eg2.embed_model_dir": "pod/eg2"}
    with pytest.raises(GuardError) as excinfo:
        build_runtime_config(frozen, {"views": {"eg2": {}}})
    assert excinfo.value.guard == "runtime_config"


def test_auroc_perfect_and_all_ties():
    y = np.array([1, 1, 0, 0])
    assert _auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert abs(_auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-12


# ---------------------------------------------------- shipped artifact / pod state
def test_shipped_frozen_artifact_matches_recorded_hash():
    artifact = os.path.join(REPO_ROOT, "configs", "frozen", "ensemble_v1.prereg.json")
    sidecar = artifact + ".sha256"
    assert os.path.exists(artifact), "frozen pre-registered artifact must be committed"
    assert _sha256(artifact) == RECORDED
    assert open(sidecar).read().split()[0] == RECORDED
    if shutil.which("git") and os.path.exists(os.path.join(REPO_ROOT, ".git")):
        proc = subprocess.run(["git", "cat-file", "blob", GIT_BLOB],
                              cwd=REPO_ROOT, capture_output=True, timeout=30)
        if proc.returncode == 0:
            assert hashlib.sha256(proc.stdout).hexdigest() == RECORDED


def test_real_pod_bundle_copy_aborts_when_it_mismatches(tmp_path):
    """The pod-remapped bundle copy legitimately fails the pre-registration check."""
    bundle_copy = os.path.join(BUNDLE, "caches", "ensemble", "ensemble_v1.json")
    if not os.path.exists(bundle_copy):
        pytest.skip("pod bundle copy not present")
    observed = _sha256(bundle_copy)
    if observed == RECORDED:
        pytest.skip("bundle copy currently holds the frozen bytes; nothing to catch")
    case = _make_case(tmp_path)
    proc, payload = _run(case, extra=["--artifact", bundle_copy], recorded=RECORDED)
    assert proc.returncode == 2, proc.stderr
    assert payload["status"] == "ABORT"
    assert payload["guard"] == "frozen_sha256"
    assert payload["frozen_sha256_recorded"] == RECORDED
    assert payload["frozen_sha256_observed"] == observed
    assert payload["frozen_sha256_observed"] != RECORDED
