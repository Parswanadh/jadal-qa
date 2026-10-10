"""Pre-registered, sha256-guarded out-of-domain (OOD) run for FROZEN ensemble_v1.

LB4 staging: the frozen pre-registered scorer's OOD run executes as ONE command
the moment the 100-window human label file lands:

    bash tools/pod/run_ood_frozen.sh

Guards (fail closed, non-zero exit, nothing scored):
  1. the label file must exist with exactly 100 labelled windows   -> else PENDING (exit 3)
  2. the frozen artifact's raw sha256 must EQUAL the recorded
     pre-registered hash before any scoring                        -> else ABORT  (exit 2)

The pre-registered authority hash is RECORDED_FROZEN_SHA256.  The byte-exact
frozen bytes are committed at configs/frozen/ensemble_v1.prereg.json, recovered
from git (blob 0ede3b83d93ce23dc5b68cd20fb7de0af703ad27 = configs/ensemble_v1.json
at freeze commit b21e3af).  Post-freeze on-disk copies (the pod bundle cache and
configs/ensemble_v1.json) were path-remapped and therefore FAIL the check; their
observed hashes are recorded, never substituted for the recorded authority.

Scoring uses the frozen config content, with exactly one allow-listed
machine-local override (views.eg2.embed_model_dir) taken from the current repo
config; the runner asserts that this is the ONLY difference before scoring.

Metrics: AUROC (primary) with a 95% CI by source-clip bootstrap, the optimistic
window-bootstrap CI as a secondary, balanced accuracy at the pre-registered
threshold, and accept/reject counts.  Nothing is tuned, refitted or re-thresholded.

Exit codes: 0 COMPLETE, 2 ABORT (guard failed), 3 PENDING (labels not delivered),
1 ERROR.  Every path writes eval/results/ood_frozen/ood_run.json, so the
deliverable exists before the labels land.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))

ID = "LB4"

# Recorded pre-registered authority.  Recorded in:
#   jadal_pod_bundle/SHA256SUMS.txt line 1 (./caches/ensemble/ensemble_v1.json)
#   jadal_pod_bundle/caches/ensemble/ood_scores.json field config_sha256
#   configs/ensemble_v1.json.sha256 (freeze sidecar)
# and reproduced byte-exactly by git blob 0ede3b83d93ce23dc5b68cd20fb7de0af703ad27
# (configs/ensemble_v1.json at freeze commit b21e3af); also cited, truncated, in
# jadal_pod_bundle/docs/FACTS.md.
RECORDED_FROZEN_SHA256 = "4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31"
FROZEN_ARTIFACT_DEFAULT = os.path.join("configs", "frozen", "ensemble_v1.prereg.json")
FROZEN_PROVENANCE = (
    "git blob 0ede3b83d93ce23dc5b68cd20fb7de0af703ad27 = configs/ensemble_v1.json at "
    "freeze commit b21e3af; recorded in jadal_pod_bundle/SHA256SUMS.txt:1, "
    "jadal_pod_bundle/caches/ensemble/ood_scores.json:config_sha256, "
    "configs/ensemble_v1.json.sha256"
)
BASE_CONFIG_DEFAULT = os.path.join("configs", "ensemble_v1.json")
OUT_DEFAULT = os.path.join("eval", "results", "ood_frozen", "ood_run.json")
CACHE_DEFAULT = os.path.join("eval", "results", "ood_frozen", "ood_scores2.json")
RUNTIME_CONFIG_DEFAULT = os.path.join("eval", "results", "ood_frozen", "runtime_config.json")
LABELS_DEFAULT_REL = os.path.join("labels", "ood_labels2.json")
MANIFEST_DEFAULT_REL = os.path.join("queues", "label_queue_random2.json")
BUNDLE_DEFAULT = "/workspace/jadal_pod_bundle"
EXPECTED_N_DEFAULT = 100

# Machine-local keys that may be overridden on top of the frozen bytes.  These
# are paths/commands, not model content; the frozen heads and threshold are used
# exactly as pre-registered.  The runner asserts nothing else differs.
ALLOWED_MACHINE_PATH_KEYS = ("views.eg2.embed_model_dir",)

PROTOCOL = (
    "out-of-domain Egocentric-10K windows, human labels; FROZEN pre-registered "
    "ensemble_v1 (two-view EG2+pixel equal-weight probability average, threshold "
    "0.5, abstain band [0.4,0.6]) applied as-is with no tuning, no refitting and "
    "no re-thresholding; primary metric AUROC with 95% CI by source-clip bootstrap "
    "(2000 resamples, seed 0); secondary optimistic window-bootstrap CI; balanced "
    "accuracy at the pre-registered threshold"
)
CLAIMS_NOTE = (
    "No OOD claim may be updated until this run completes. OOD headline claims "
    "stay n=60 / n_bad=9 (first batch: jadal_pod_bundle/labels/ood_labels.json, "
    "51 good / 9 bad) and are untouched by this staged runner."
)

_LABEL_GOOD = {"good", "1", "1.0", "true", "yes", "usable"}
_LABEL_BAD = {"bad", "0", "0.0", "false", "no", "unusable"}


class GuardError(RuntimeError):
    """A fail-closed precondition failure; nothing may be scored."""

    def __init__(self, guard: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.guard = guard
        self.details = details or {}


# --------------------------------------------------------------------------- utils
def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: str | None) -> str | None:
    if path is None:
        return None
    ap = os.path.abspath(path)
    if ap.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(ap, REPO_ROOT)
    return ap


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _dotted_get(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _dotted_pop(cfg: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def _strip_keys(cfg: dict, keys) -> dict:
    out = copy.deepcopy(cfg)
    for k in keys:
        _dotted_pop(out, k)
    return out


def _resolve_first(candidates) -> tuple[str | None, list[str]]:
    searched = [os.path.abspath(c) for c in candidates]
    for c in searched:
        if os.path.exists(c):
            return c, searched
    return None, searched


def _bundle_dir() -> str:
    return os.environ.get("JADAL_BUNDLE", BUNDLE_DEFAULT)


# ------------------------------------------------------------------- preconditions
def build_runtime_config(frozen_cfg: dict, base_cfg: dict,
                         allowed=ALLOWED_MACHINE_PATH_KEYS) -> tuple[dict, dict]:
    """Frozen content + allow-listed machine-path overrides only (asserted)."""
    runtime = copy.deepcopy(frozen_cfg)
    overrides = {}
    for dotted in allowed:
        value = _dotted_get(base_cfg, dotted)
        if not value:
            raise GuardError("runtime_config",
                             f"base config has no usable value for override {dotted}")
        if _dotted_get(runtime, dotted) != value:
            cur = runtime
            parts = dotted.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
            overrides[dotted] = value
    if _strip_keys(runtime, allowed) != _strip_keys(frozen_cfg, allowed):
        raise GuardError("runtime_config_semantics",
                         "runtime config differs from the frozen artifact outside the "
                         f"allow-listed machine-path keys {list(allowed)}")
    return runtime, overrides


def materialize_runtime_config(runtime_cfg: dict, path: str) -> str:
    """Write the runtime config + its .sha256 sidecar (load_config format)."""
    body = json.dumps(runtime_cfg, indent=1)
    sha = hashlib.sha256(body.encode()).hexdigest()
    _write_text(path, body)
    _write_text(path + ".sha256", f"{sha}  {os.path.basename(path)}\n")
    with open(path) as f:
        reloaded = json.load(f)
    if reloaded != runtime_cfg:
        raise GuardError("runtime_config", "runtime config did not round-trip through JSON")
    return sha


def artifact_observations(recorded: str, artifact: str) -> list[dict]:
    """Every on-disk ensemble_v1 copy we know about, with observed vs recorded."""
    candidates = [
        ("frozen_preregistered_copy", artifact),
        ("repo_config", os.path.join(REPO_ROOT, BASE_CONFIG_DEFAULT)),
        ("bundle_cache", os.path.join(_bundle_dir(), "caches", "ensemble", "ensemble_v1.json")),
    ]
    out = []
    for role, path in candidates:
        entry = {"role": role, "path": _rel(path), "exists": os.path.exists(path)}
        if entry["exists"]:
            entry["observed_sha256"] = _sha256_file(path)
            entry["matches_recorded_preregistered_hash"] = entry["observed_sha256"] == recorded
        out.append(entry)
    return out


def load_labels(path: str) -> dict:
    """{window_id: {"label": 0/1, "entry": {...}}} for the supported schemas."""
    with open(path) as f:
        doc = json.load(f)
    items = None
    if isinstance(doc, dict):
        items = doc.get("latest", doc.get("items", doc.get("segments", doc.get("windows"))))
    if items is None and isinstance(doc, list):
        items = doc
    out = {}
    if isinstance(items, dict):
        for wid, entry in items.items():
            label = _entry_label(entry)
            if label is not None:
                out[str(wid)] = {"label": label,
                                 "entry": entry if isinstance(entry, dict) else {}}
    else:
        for entry in items or []:
            if not isinstance(entry, dict):
                continue
            wid = entry.get("segment_id", entry.get("id", entry.get("window_id")))
            label = _entry_label(entry)
            if wid is not None and label is not None:
                out[str(wid)] = {"label": label, "entry": entry}
    return out


def _entry_label(entry):
    if isinstance(entry, dict):
        raw = entry.get("verdict", entry.get("label", entry.get("value")))
    else:
        raw = entry
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in _LABEL_GOOD:
        return 1
    if value in _LABEL_BAD:
        return 0
    return None


def load_manifest(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        doc = json.load(f)
    items = doc.get("segments", doc.get("items", doc.get("windows"))) if isinstance(doc, dict) else doc
    out = {}
    for entry in items or []:
        if not isinstance(entry, dict):
            continue
        wid = entry.get("segment_id", entry.get("id", entry.get("window_id")))
        if wid is not None:
            out[str(wid)] = entry
    return out


def resolve_windows(labels: dict, manifest: dict) -> tuple[dict, list]:
    resolved, missing = {}, []
    for wid, rec in labels.items():
        entry = rec["entry"]
        man = manifest.get(wid, {})
        path = (entry.get("path") or entry.get("video_path")
                or man.get("path") or man.get("video_path"))
        if not path:
            missing.append(wid)
            continue
        start = entry.get("start_s", man.get("start_s", 0.0))
        end = entry.get("end_s", man.get("end_s"))
        if end is None:
            end = float(start) + 10.0
        clip = (entry.get("clip") or entry.get("source_path")
                or man.get("clip") or man.get("source_path") or wid)
        resolved[wid] = {"path": str(path), "start_s": float(start),
                         "end_s": float(end), "clip": str(clip)}
    return resolved, missing


def load_scores(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        doc = json.load(f)
    scores, meta = {}, {}
    if isinstance(doc, dict):
        if isinstance(doc.get("config_sha256"), str):
            meta["config_sha256"] = doc["config_sha256"]
        rows = doc.get("scores")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict) or "error" in row:
                    continue
                wid = row.get("id", row.get("segment_id"))
                prob = row.get("usable_prob", row.get("prob", row.get("p")))
                if wid is not None and isinstance(prob, (int, float)):
                    scores[str(wid)] = float(prob)
            return scores, meta
        for wid, value in doc.items():
            if isinstance(value, (int, float)):
                scores[str(wid)] = float(value)
            elif isinstance(value, dict) and isinstance(value.get("usable_prob"), (int, float)):
                scores[str(wid)] = float(value["usable_prob"])
    elif isinstance(doc, list):
        for row in doc:
            if not isinstance(row, dict):
                continue
            wid = row.get("id", row.get("segment_id"))
            prob = row.get("usable_prob", row.get("prob", row.get("p")))
            if wid is not None and isinstance(prob, (int, float)):
                scores[str(wid)] = float(prob)
    return scores, meta


# ------------------------------------------------------------------------- scoring
def score_windows(resolved: dict, order: list, runtime_config: str, cache_path: str,
                  rescore: bool = False) -> tuple[dict, dict]:
    """Score the windows with the frozen scorer, caching every window to disk."""
    sys.path.insert(0, REPO_ROOT)
    from jadal.ensemble.scorer import score as frozen_score  # noqa: E402

    cache = {}
    if os.path.exists(cache_path) and not rescore:
        try:
            with open(cache_path) as f:
                cache = json.load(f).get("scores", {})
        except Exception:
            cache = {}
    errors = {}
    for i, wid in enumerate(order, 1):
        if wid in cache and "usable_prob" in cache[wid] and not rescore:
            continue
        rec = resolved[wid]
        try:
            result = frozen_score(rec["path"], rec["start_s"], rec["end_s"],
                                  config_path=runtime_config)
            cache[wid] = {"usable_prob": result["usable_prob"],
                          "verdict": result["verdict"],
                          "p_eg2": result.get("p_eg2"), "p_pixel": result.get("p_pixel"),
                          "features_digest": result.get("features_digest"),
                          "config_sha256": result.get("versions", {}).get("config_sha256")}
            errors.pop(wid, None)
            print(f"[{ID}] scored {i}/{len(order)} {wid} "
                  f"p={result['usable_prob']:.4f} {result['verdict']}", flush=True)
        except Exception as exc:  # keep going; the run aborts at the end
            errors[wid] = f"{type(exc).__name__}: {exc}"
            cache[wid] = {"error": errors[wid]}
            print(f"[{ID}] FAILED {i}/{len(order)} {wid}: {errors[wid]}", flush=True)
        _write_json(cache_path, {"n": len(cache), "scores": cache})
    return cache, errors


# ------------------------------------------------------------------------- metrics
def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC with tie handling (average ranks)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("AUROC undefined: one class is empty")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=float)
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y)
    pred = np.asarray(pred)
    tpr = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    tnr = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    return (tpr + tnr) / 2.0, tpr, tnr


def _bootstrap_auroc(y: np.ndarray, p: np.ndarray, groups: dict | None,
                     n_boot: int, seed: int, unit: str) -> dict | None:
    rng = np.random.default_rng(seed)
    reps = []
    if unit == "source_clip":
        keys = list(groups)
        idx_by_key = [np.asarray(groups[k], dtype=int) for k in keys]
        m = len(keys)
        for _ in range(n_boot):
            pick = rng.integers(0, m, m)
            idx = np.concatenate([idx_by_key[i] for i in pick])
            if len(np.unique(y[idx])) < 2:
                continue
            reps.append(_auroc(y[idx], p[idx]))
    else:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            reps.append(_auroc(y[idx], p[idx]))
    if not reps:
        return None
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "n_boot_requested": int(n_boot), "n_boot_used": len(reps),
            "seed": int(seed), "unit": unit}


# ---------------------------------------------------------------------- reporting
def _base_payload(status: str, reason: str, guard: str | None = None,
                  recorded: str = RECORDED_FROZEN_SHA256) -> dict:
    payload = {
        "id": ID,
        "status": status,
        "preregistered": True,
        "frozen_sha256": recorded,
        "frozen_artifact": FROZEN_ARTIFACT_DEFAULT,
        "frozen_provenance": FROZEN_PROVENANCE,
        "n": None,
        "n_bad": None,
        "label_source": None,
        "protocol": PROTOCOL,
        "threshold": None,
        "expected_n": EXPECTED_N_DEFAULT,
        "claims_note": CLAIMS_NOTE,
        "command": "bash tools/pod/run_ood_frozen.sh",
        "reason": reason,
        "generated_utc": _now(),
    }
    if guard:
        payload["guard"] = guard
    return payload


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Frozen pre-registered OOD run (sha256-guarded).")
    ap.add_argument("--labels", default=None,
                    help=f"human labels file (default: {LABELS_DEFAULT_REL} under the repo, "
                         f"else $JADAL_BUNDLE/{LABELS_DEFAULT_REL})")
    ap.add_argument("--artifact", default=None,
                    help=f"frozen pre-registered artifact (default: {FROZEN_ARTIFACT_DEFAULT})")
    ap.add_argument("--recorded-sha256", default=RECORDED_FROZEN_SHA256,
                    help="the recorded pre-registered hash (authority; do not edit)")
    ap.add_argument("--base-config", default=None,
                    help=f"source of allow-listed machine paths (default: {BASE_CONFIG_DEFAULT})")
    ap.add_argument("--manifest", default=None,
                    help=f"window->video manifest (default: {MANIFEST_DEFAULT_REL} under the "
                         f"repo, else $JADAL_BUNDLE/{MANIFEST_DEFAULT_REL})")
    ap.add_argument("--scores", default=None,
                    help="optional precomputed per-window scores JSON (skips model scoring)")
    ap.add_argument("--out", default=None, help=f"results JSON (default: {OUT_DEFAULT})")
    ap.add_argument("--cache", default=None,
                    help=f"per-window score cache (default: {CACHE_DEFAULT})")
    ap.add_argument("--runtime-config", default=None,
                    help=f"where to write the verified runtime config (default: {RUNTIME_CONFIG_DEFAULT})")
    ap.add_argument("--expect-n", type=int, default=EXPECTED_N_DEFAULT)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescore", action="store_true", help="ignore the score cache")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    out_path = args.out or os.path.join(REPO_ROOT, OUT_DEFAULT)
    try:
        return _run(args, out_path)
    except GuardError as exc:
        payload = _base_payload("ABORT", str(exc), guard=exc.guard,
                                recorded=args.recorded_sha256)
        payload.update(exc.details)
        _write_json(out_path, payload)
        print(f"[{ID}] ABORT ({exc.guard}): {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - unexpected failure path
        payload = _base_payload("ERROR", f"{type(exc).__name__}: {exc}", guard="unexpected",
                                recorded=args.recorded_sha256)
        payload["traceback_tail"] = traceback.format_exc()[-2000:]
        _write_json(out_path, payload)
        print(f"[{ID}] ERROR: {exc}", file=sys.stderr)
        return 1


def _run(args, out_path: str) -> int:
    recorded = args.recorded_sha256
    artifact = args.artifact or os.path.join(REPO_ROOT, FROZEN_ARTIFACT_DEFAULT)

    # ---- guard 1: labels delivered with the expected number of windows? -------
    if args.labels:
        labels_path, searched = _resolve_first([args.labels])
    else:
        labels_path, searched = _resolve_first([
            os.path.join(REPO_ROOT, LABELS_DEFAULT_REL),
            os.path.join(_bundle_dir(), LABELS_DEFAULT_REL),
        ])
    observations = artifact_observations(recorded, artifact)
    artifact_observed = (_sha256_file(artifact) if os.path.exists(artifact) else None)

    if labels_path is None:
        payload = _base_payload(
            "PENDING",
            "human labels not delivered: no 100-window label file found; nothing was scored "
            f"(searched: {', '.join(_rel(p) for p in searched)})",
            guard="labels_absent", recorded=recorded)
        payload.update({
            "labels_path": None,
            "labels_path_searched": [_rel(p) for p in searched],
            "frozen_artifact": _rel(artifact),
            "frozen_sha256_observed": artifact_observed,
            "frozen_sha256_verified": artifact_observed == recorded,
            "artifact_observations": observations,
            "next_command": "bash tools/pod/run_ood_frozen.sh",
        })
        _write_json(out_path, payload)
        print(f"[{ID}] PENDING: {payload['reason']}")
        print(f"[{ID}] frozen artifact sha256 observed {artifact_observed} "
              f"({'matches' if artifact_observed == recorded else 'DOES NOT MATCH'} the recorded hash)")
        return 3

    try:
        labels = load_labels(labels_path)
    except Exception as exc:
        raise GuardError("labels_parse",
                         f"cannot parse {_rel(labels_path)}: {type(exc).__name__}: {exc}",
                         {"labels_path": _rel(labels_path)})
    n = len(labels)
    n_bad = sum(1 for rec in labels.values() if rec["label"] == 0)
    n_good = n - n_bad
    if n != args.expect_n:
        raise GuardError(
            "labels_count",
            f"{_rel(labels_path)} has {n} labelled windows, expected exactly "
            f"{args.expect_n}: refusing to score",
            {"labels_path": _rel(labels_path), "n": n, "n_bad": n_bad})
    if n_good == 0 or n_bad == 0:
        raise GuardError(
            "labels_single_class",
            f"{_rel(labels_path)} has n_good={n_good} n_bad={n_bad}: AUROC is undefined, "
            "refusing to report numbers",
            {"labels_path": _rel(labels_path), "n": n, "n_bad": n_bad})

    # ---- guard 2: frozen artifact must match the pre-registered hash ---------
    if not os.path.exists(artifact):
        raise GuardError("frozen_artifact_missing",
                         f"frozen pre-registered artifact not found: {_rel(artifact)}",
                         {"frozen_artifact": _rel(artifact), "n": n, "n_bad": n_bad,
                          "labels_path": _rel(labels_path)})
    artifact_observed = _sha256_file(artifact)
    if artifact_observed != recorded:
        raise GuardError(
            "frozen_sha256",
            "frozen artifact sha256 does NOT equal the recorded pre-registered hash; "
            "refusing to score (the pre-registration is the authority, the on-disk bytes "
            "must be recovered, never the hash rewritten)",
            {"frozen_artifact": _rel(artifact),
             "frozen_sha256_recorded": recorded,
             "frozen_sha256_observed": artifact_observed,
             "labels_path": _rel(labels_path), "n": n, "n_bad": n_bad,
             "artifact_observations": observations})

    with open(artifact) as f:
        frozen_cfg = json.load(f)
    base_config = args.base_config or os.path.join(REPO_ROOT, BASE_CONFIG_DEFAULT)
    with open(base_config) as f:
        base_cfg = json.load(f)
    runtime_cfg, overrides = build_runtime_config(frozen_cfg, base_cfg)
    runtime_path = args.runtime_config or os.path.join(REPO_ROOT, RUNTIME_CONFIG_DEFAULT)
    runtime_sha = materialize_runtime_config(runtime_cfg, runtime_path)

    # ---- score (or consume supplied scores) ----------------------------------
    order = sorted(labels)
    if args.scores:
        scores, score_meta = load_scores(args.scores)
        if score_meta.get("config_sha256") and score_meta["config_sha256"] != recorded:
            raise GuardError(
                "scores_config_sha256",
                f"{_rel(args.scores)} was produced with config_sha256 "
                f"{score_meta['config_sha256']}, which is not the pre-registered hash",
                {"scores_path": _rel(args.scores),
                 "scores_config_sha256": score_meta["config_sha256"],
                 "frozen_sha256_recorded": recorded})
        missing_scores = [w for w in order if w not in scores]
        if missing_scores:
            raise GuardError("scores_coverage",
                             f"{len(missing_scores)} labelled windows have no score in "
                             f"{_rel(args.scores)}",
                             {"n_missing": len(missing_scores),
                              "missing_window_ids": missing_scores[:20],
                              "scores_path": _rel(args.scores)})
        scored_from = f"scores file {_rel(args.scores)}"
    else:
        manifest_path, manifest_searched = _resolve_first([
            os.path.join(REPO_ROOT, MANIFEST_DEFAULT_REL),
            os.path.join(_bundle_dir(), MANIFEST_DEFAULT_REL),
        ]) if not args.manifest else _resolve_first([args.manifest])
        manifest = load_manifest(manifest_path)
        resolved, missing = resolve_windows(labels, manifest)
        if missing:
            raise GuardError(
                "window_paths",
                f"{len(missing)} labelled windows have no video path (checked the labels "
                "entries and the manifest): refusing to score a partial set",
                {"n_missing": len(missing), "missing_window_ids": missing[:20],
                 "manifest": _rel(manifest_path),
                 "manifest_searched": [_rel(p) for p in manifest_searched]})
        missing_manifest = [w for w in order if w not in manifest]
        cache_path = args.cache or os.path.join(REPO_ROOT, CACHE_DEFAULT)
        cache, errors = score_windows(resolved, order, runtime_path, cache_path,
                                      rescore=args.rescore)
        if errors:
            raise GuardError(
                "scoring_errors",
                f"{len(errors)} of {len(order)} windows failed to score with the frozen "
                "scorer; partial results are not reported",
                {"n_errors": len(errors),
                 "scoring_errors": dict(list(errors.items())[:10]),
                 "cache_path": _rel(cache_path),
                 "n": n, "n_bad": n_bad, "labels_path": _rel(labels_path)})
        scores = {w: cache[w]["usable_prob"] for w in order}
        scored_from = "frozen scorer jadal.ensemble.scorer.score(video, start_s, end_s)"
        score_meta = {"manifest": _rel(manifest_path),
                      "manifest_searched": [_rel(p) for p in manifest_searched],
                      "n_windows_without_manifest_entry": len(missing_manifest),
                      "cache_path": _rel(cache_path)}

    # ---- metrics -------------------------------------------------------------
    index = {w: i for i, w in enumerate(order)}
    y = np.array([labels[w]["label"] for w in order], dtype=int)
    p = np.array([scores[w] for w in order], dtype=float)
    auroc = _auroc(y, p)
    groups: dict[str, list] = {}
    for w in order:
        clip = resolved[w]["clip"] if not args.scores else _group_of(labels, w)
        groups.setdefault(clip, []).append(index[w])
    ci_clip = _bootstrap_auroc(y, p, groups, args.boot, args.seed, "source_clip")
    ci_window = _bootstrap_auroc(y, p, None, args.boot, args.seed, "window")
    threshold = float(runtime_cfg["threshold"])
    lo_band, hi_band = runtime_cfg["abstain_band"]
    pred = (p > threshold).astype(int)
    bal, tpr, tnr = _balanced_accuracy(y, pred)
    verdicts = {"accept": int((p > threshold).sum()),
                "reject": int((p < lo_band).sum()),
                "review": int(((p >= lo_band) & (p <= hi_band)).sum())}
    clip_counts = {k: len(v) for k, v in sorted(groups.items())}

    payload = _base_payload("COMPLETE",
                            "frozen pre-registered OOD run completed on the delivered labels",
                            recorded=recorded)
    payload.update({
        "n": n, "n_bad": n_bad, "n_good": n_good,
        "label_source": f"human (single labeller), {_rel(labels_path)}",
        "labels_path": _rel(labels_path),
        "threshold": threshold,
        "auroc": round(auroc, 4),
        "auroc_ci95_clip_bootstrap": ci_clip,
        "auroc_ci95_window_bootstrap_optimistic": ci_window,
        "balanced_accuracy_at_threshold": round(bal, 4),
        "tpr_good_accept_rate": round(tpr, 4),
        "tnr_bad_reject_rate": round(tnr, 4),
        "accuracy_at_threshold": round(float((pred == y).mean()), 4),
        "accept_at_threshold": int(pred.sum()),
        "reject_at_threshold": int((pred == 0).sum()),
        "frozen_verdict_counts": verdicts,
        "good_rejected_at_threshold": int(((p <= threshold) & (y == 1)).sum()),
        "bad_rejected_at_threshold": int(((p <= threshold) & (y == 0)).sum()),
        "good_n": n_good, "bad_n": n_bad,
        "mean_prob_good": round(float(p[y == 1].mean()), 4),
        "mean_prob_bad": round(float(p[y == 0].mean()), 4),
        "n_source_clips": len(groups),
        "source_clip_counts": clip_counts,
        "frozen_sha256": recorded,
        "frozen_sha256_observed": artifact_observed,
        "frozen_sha256_verified": artifact_observed == recorded,
        "frozen_artifact": _rel(artifact),
        "frozen_provenance": FROZEN_PROVENANCE,
        "frozen_semantics_identical": True,
        "runtime_config": _rel(runtime_path),
        "runtime_config_sha256": runtime_sha,
        "runtime_overrides": overrides,
        "artifact_observations": observations,
        "scored_from": scored_from,
        "scoring_meta": score_meta,
    })
    _write_json(out_path, payload)
    print(f"[{ID}] COMPLETE n={n} n_bad={n_bad} AUROC={auroc:.3f} "
          f"clip-CI=[{ci_clip['lo'] if ci_clip else float('nan'):.3f},"
          f"{ci_clip['hi'] if ci_clip else float('nan'):.3f}] "
          f"balacc@{threshold}={bal:.3f} accept={int(pred.sum())} reject={int((pred == 0).sum())}")
    print(f"[{ID}] results: {_rel(out_path)}")
    return 0


def _group_of(labels: dict, wid: str) -> str:
    """Source-clip id for a window when scores come from a file (no manifest use)."""
    entry = labels[wid]["entry"]
    return str(entry.get("clip") or entry.get("source_path") or wid)


if __name__ == "__main__":
    sys.exit(main())
