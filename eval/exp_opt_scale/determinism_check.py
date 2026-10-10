#!/usr/bin/env python3
"""R1/T4 — determinism check: score 10 windows twice, compare receipts.

Run A = the already-completed rows of the full-corpus run (windows.jsonl.partial).
Run B = a fresh single-worker re-score of the same 10 windows.

Compares, per window:
  * score, verdict, features_digest, reasons  (deterministic content)
  * ``content_sha256`` = sha256(canonical(line without ``latency_ms``))
  * the raw ``receipt_sha256`` (includes ``latency_ms`` -> expected to differ)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))

from contract import canonical_bytes, sha256_hex  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "queues", "manifest_scale_all.json")
PARTIAL = os.path.join(REPO, "eval", "results", "exp_opt_scale", "windows.jsonl.partial")
MINI = "/tmp/opencode/det_mini_manifest.json"
RUNB = "/tmp/opencode/det_run_b.jsonl"
OUT = os.path.join(REPO, "eval", "results", "exp_opt_scale", "determinism.json")


def content_sha(line):
    body = {k: v for k, v in line.items() if k not in ("latency_ms", "receipt_sha256")}
    return sha256_hex(canonical_bytes(body))


def normalized_receipt(line):
    """The contract receipt with the timing field removed (byte-identical)."""
    body = {k: v for k, v in line.items() if k != "latency_ms"}
    body.pop("receipt_sha256", None)
    return sha256_hex(canonical_bytes(body))


def pick(n=10):
    rows = []
    for ln in open(PARTIAL):
        try:
            r = json.loads(ln)["row"]
        except Exception:  # noqa: BLE001
            continue
        rows.append(r)
    # spread across clips: take every len//n-th
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def main():
    if os.path.exists(RUNB):
        b_ids = [json.loads(ln)["window_id"] for ln in open(RUNB)]
        by_id_partial = {}
        for ln in open(PARTIAL):
            try:
                r = json.loads(ln)["row"]
            except Exception:  # noqa: BLE001
                continue
            by_id_partial[r["window_id"]] = r
        picks = [by_id_partial[i] for i in b_ids]
    else:
        picks = pick(10)
    ids = [r["window_id"] for r in picks]
    man = json.load(open(MANIFEST))
    by_id = {it["id"]: it for it in man["items"]}
    items = [by_id[i] for i in ids]
    mini = {
        "built_utc": man.get("built_utc"), "window_s": man.get("window_s"),
        "n_clips": len({i["clip"] for i in items}), "n_new_clips": 0,
        "n_existing_clips": 0, "n_windows": len(items),
        "total_hours": round(len(items) * 10.0 / 3600.0, 4), "items": items,
    }
    json.dump(mini, open(MINI, "w"))
    print(f"mini manifest: {len(items)} windows -> {MINI}", flush=True)

    if not os.path.exists(RUNB):
        env = dict(os.environ)
        env["JADAL_MANIFEST"] = MINI
        env["JADAL_CONFIG"] = os.path.join(REPO, "configs", "ensemble_v1_sparse.json")
        env["OMP_NUM_THREADS"] = "1"
        env["JADAL_GPU_SLOTS"] = env.get("JADAL_GPU_SLOTS", "4")
        cmd = ["bash", "tools/pod/gpu_run.sh", "python3",
               "eval/exp_opt_scale/run_opt_scale.py", "--workers", "1",
               "--no-compare", "--pixel-fps", "3.33", "--eg2-fps", "1",
               "--early-exit-s", "4", "--window-stride", "1", "--skip-mosaic",
               "--out", RUNB]
        print("running B:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, env=env, check=True)

    b = {}
    for ln in open(RUNB):
        r = json.loads(ln)
        b[r["window_id"]] = r
    a = {r["window_id"]: r for r in picks}
    out = {"experiment": "R1.04-determinism", "n": len(ids), "windows": {}}
    ok_content = ok_digest = ok_score = ok_raw = 0
    for i in ids:
        la = a[i]["contract"]
        lb = b[i]["contract"]
        ca, cb = content_sha(la), content_sha(lb)
        na, nb = normalized_receipt(la), normalized_receipt(lb)
        same_digest = la["features_digest"] == lb["features_digest"]
        same_score = la["score"] == lb["score"] and la["verdict"] == lb["verdict"]
        same_content = ca == cb
        same_raw = na == nb
        ok_content += same_content
        ok_digest += same_digest
        ok_score += same_score
        ok_raw += same_raw
        out["windows"][i] = {
            "clip": a[i]["clip"], "score": la["score"], "verdict": la["verdict"],
            "features_digest_a": la["features_digest"], "features_digest_b": lb["features_digest"],
            "content_sha256_a": ca, "content_sha256_b": cb,
            "normalized_receipt_a": na, "normalized_receipt_b": nb,
            "receipt_sha256_a": la["receipt_sha256"], "receipt_sha256_b": lb["receipt_sha256"],
            "same_content": same_content, "same_digest": same_digest,
            "same_score_verdict": same_score, "same_normalized_receipt": same_raw,
        }
    out["summary"] = {
        "n": len(ids), "same_content_receipt": ok_content,
        "same_features_digest": ok_digest, "same_score_verdict": ok_score,
        "same_normalized_receipt_excl_latency": ok_raw,
        "note": ("content_sha256 and normalized_receipt exclude latency_ms; the raw "
                 "receipt_sha256 includes latency_ms by contract design and is therefore "
                 "not expected to match across runs"),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
