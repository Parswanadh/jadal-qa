#!/usr/bin/env python3
"""S1.01 — shared helpers for deterministic sharding + aggregation.

Pure-stdlib + the repo contract helpers. No GPU, no network: importable from the
shard worker, the aggregator, the scaling harness and the tests.

Key ideas
---------
* **Deterministic shard plan.** Clips (never windows) are the sharding unit, so a
  clip is decoded by exactly one worker and no window is split. Assignment is a
  longest-processing-time (LPT) greedy over clip window counts, ties broken by
  the sorted clip name -> byte-stable for a given (subset, K).
* **Shard-invariance digest.** Each row carries non-deterministic wall timings
  (``latency_ms``, ``stage_wall`` ...). ``det_projection`` strips those, leaving
  the scored content (verdict / score / features / reasons / versions). The
  combined digest is a SHA-256 over ``(window_id, row_digest)`` pairs sorted by
  ``window_id``, so it is independent of shard count and of row order.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
BUNDLE = os.environ.get("JADAL_BUNDLE", "jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "queues", "manifest_scale.json")
SHARD_ROOT = os.path.join(REPO, "eval", "results", "exp_shard")
SHARD_DIR = os.path.join(SHARD_ROOT, "shards")

# non-deterministic / run-local fields stripped before hashing
TIMING_KEYS = (
    "latency_ms", "stage_wall", "t_wall0", "t_wall1", "gpu",
    "decode_compare", "write", "wall_s", "cpu_s",
)

import sys  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))
from contract import canonical_bytes, sha256_hex  # noqa: E402


# --------------------------------------------------------------------- manifest
def load_items(manifest=MANIFEST):
    """(manifest, items) with items in the canonical (clip, start_s, id) order."""
    man = json.load(open(manifest))
    items = man["items"]
    for it in items:
        it.setdefault("window_id", it["id"])
    items = sorted(items, key=lambda i: (i["clip"], i["start_s"], i["window_id"]))
    return man, items


def clip_counts(items):
    return Counter(it["clip"] for it in items)


def select_items(items, subset):
    """Deterministic subset selection.

    ``subset`` is one of ``all`` / ``first<N>`` / ``smallest<N>`` / ``largest<N>``
    (N clips). ``first`` = the first N clips in sorted order; ``smallest`` /
    ``largest`` = the N clips with the fewest / most windows (ties by name).
    """
    if not subset or subset == "all":
        return list(items)
    m = re.match(r"^(first|smallest|largest)(\d+)$", subset)
    if not m:
        raise ValueError(f"bad subset spec: {subset!r}")
    kind, n = m.group(1), int(m.group(2))
    cnt = clip_counts(items)
    clips = sorted(cnt)
    if kind == "first":
        chosen = clips[:n]
    elif kind == "smallest":
        chosen = sorted(clips, key=lambda c: (cnt[c], c))[:n]
    else:  # largest
        chosen = sorted(clips, key=lambda c: (-cnt[c], c))[:n]
    keep = set(chosen)
    return [it for it in items if it["clip"] in keep]


# ------------------------------------------------------------------- shard plan
def make_plan(items, k):
    """Deterministic, window-balanced clip->shard assignment (LPT greedy).

    Returns a JSON-serialisable plan; identical for a given (items, k).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    cnt = clip_counts(items)
    clips = sorted(cnt)
    order = sorted(clips, key=lambda c: (-cnt[c], c))
    loads = [0] * k
    assign = {}
    for c in order:
        s = min(range(k), key=lambda i: (loads[i], i))
        assign[c] = s
        loads[s] += cnt[c]
    return {
        "K": k,
        "assignment": assign,
        "loads_windows": loads,
        "clip_windows": {c: cnt[c] for c in clips},
        "n_windows": sum(cnt.values()),
        "n_clips": len(clips),
    }


def plan_path(root, k):
    return os.path.join(root, f"plan_K{k}.json")


def write_plan(plan, root):
    os.makedirs(root, exist_ok=True)
    p = plan_path(root, plan["K"])
    with open(p, "w") as f:
        f.write(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    return p


def read_plan(path):
    return json.load(open(path))


def items_for_shard(items, plan, shard_index):
    return [it for it in items if plan["assignment"][it["clip"]] == shard_index]


# ------------------------------------------------------------ deterministic hash
def det_projection(row):
    """Row with all run-local timings removed (scored content only)."""
    r = {k: v for k, v in row.items() if k not in TIMING_KEYS}
    line = dict(r.get("contract") or {})
    line.pop("latency_ms", None)
    line.pop("receipt_sha256", None)   # receipt hashes latency_ms -> run-local
    r["contract"] = line
    return r


def row_digest(row):
    return sha256_hex(canonical_bytes(det_projection(row)))


def combined_digest(rows):
    """Order-independent digest over a set of rows (shard-invariance key)."""
    pairs = sorted((r["window_id"], row_digest(r)) for r in rows)
    return sha256_hex(canonical_bytes(pairs))


def shards_digest(per_shard):
    """Digest of the per-shard digests: sha256(sorted [(shard, digest)])."""
    pairs = sorted((str(k), v) for k, v in per_shard.items())
    return sha256_hex(canonical_bytes(pairs))


def verdict_dist(rows):
    return dict(sorted(Counter(r["verdict"] for r in rows).items()))


def read_jsonl(path):
    out = []
    for ln in open(path):
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
