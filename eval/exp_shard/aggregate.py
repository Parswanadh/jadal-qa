#!/usr/bin/env python3
"""S1.01 — deterministic aggregation of isolated shard outputs.

Protocol (order-independent, shard-invariant):

    shard_i.jsonl  --(strip run-local timings)-->  per-row digest
                   --(sort by window_id)--------->  shard_i digest
    all shard digests --> combined digest = sha256(canonical(sorted (window_id, row_digest)))

The combined digest is a function of the *scored content* only, so it is the same
whether the windows were produced by 1 process or K isolated processes, and
whether they arrive in any order. ``--expect-digest`` fails loudly on mismatch
(the shard-invariance test).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_shard"))

import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--out", required=True, help="combined result JSONL")
    ap.add_argument("--digest-out", required=True, help="combined digest JSON")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--manifest", default=common.MANIFEST)
    ap.add_argument("--expect-digest", default=None)
    args = ap.parse_args()

    # canonical window order (from the same subset spec the shards used)
    _, items = common.load_items(args.manifest)
    items = common.select_items(items, args.subset)
    order = {it["window_id"]: i for i, it in enumerate(items)}

    rows, per_shard, dupes = [], {}, []
    seen = set()
    for i in range(args.shards):
        p = os.path.join(args.shard_dir, f"shard_{i}.jsonl")
        srows = common.read_jsonl(p)
        d = common.combined_digest(srows)
        per_shard[i] = d
        for r in srows:
            wid = r["window_id"]
            if wid in seen:
                dupes.append(wid)
            seen.add(wid)
            rows.append(r)
    if dupes:
        raise SystemExit(f"ERROR: {len(dupes)} duplicate window_id(s) across shards: {dupes[:5]}")

    combined = common.combined_digest(rows)
    # integrity: each shard's self-reported digest must match a fresh recompute
    integrity = {}
    for i in range(args.shards):
        mp = os.path.join(args.shard_dir, "state", f"shard_{i}",
                          f"shard_{i}.meta.json")
        if os.path.exists(mp):
            reported = json.load(open(mp)).get("shard_digest")
            integrity[i] = bool(reported == per_shard[i])

    rows.sort(key=lambda r: order.get(r["window_id"], 1 << 30))
    common.write_jsonl(args.out, rows)

    digest_doc = {
        "protocol": "combined = sha256 over sorted (window_id, "
                    "sha256(canonical(det_projection(row)))); partition-independent",
        "det_projection": "contract line minus latency_ms/receipt_sha256; row minus "
                          "latency_ms/stage_wall/t_wall0/t_wall1/gpu/decode_compare",
        "n_shards": args.shards, "n_windows": len(rows),
        "combined_digest": combined,
        "per_shard_digest": {str(k): v for k, v in sorted(per_shard.items())},
        "per_shard_integrity_ok": integrity,
        "verdict_distribution": common.verdict_dist(rows),
        "subset": args.subset,
        "result_jsonl": os.path.relpath(args.out, REPO),
    }
    json.dump(digest_doc, open(args.digest_out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in digest_doc.items()
                      if k != "per_shard_digest"}, indent=1))

    if args.expect_digest:
        ok = combined == args.expect_digest
        print(f"expect-digest: {'MATCH' if ok else 'MISMATCH'} "
              f"(got {combined[:16]} want {args.expect_digest[:16]})")
        if not ok:
            raise SystemExit(2)
    if integrity and not all(integrity.values()):
        raise SystemExit(f"ERROR: per-shard integrity failed: {integrity}")


if __name__ == "__main__":
    main()
