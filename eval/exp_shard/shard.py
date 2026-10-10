#!/usr/bin/env python3
"""S1.01 — one isolated shard worker.

Runs the reused latency runner (``eval/exp_latency_fast/run_fast.py``, the
adopted-optimization pipeline) over exactly the windows of ONE shard:

* own process (single-threaded, ``OMP_NUM_THREADS=1`` -> <=1 core),
* own output JSONL ``<out>``,
* own state/log dir ``<state-dir>``,
* no shared mutable state with any other shard.

The shard plan is deterministic (see ``common.make_plan``); clips are never
split, so a shard is a self-contained set of whole clips.

Run (one GPU slot per pod; wrap the whole fleet, not each shard):
    bash tools/pod/gpu_run.sh python3 eval/exp_shard/shard.py \
        --shard-index 0 --shards 4 --subset smallest8 \
        --out eval/results/exp_shard/shards/shard_0.jsonl \
        --state-dir eval/results/exp_shard/shards/state/shard_0
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_latency_fast"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_shard"))

import common  # noqa: E402
import run_e8 as e8  # noqa: E402
import run_fast  # noqa: E402  (the reused latency runner)

# the adopted-optimization throughput config being scaled (run_both.sh COMMON)
ADOPTED_ENV = {
    "JADAL_PIXEL_FPS": "3.33",
    "JADAL_EG2_FPS": "1",
    "JADAL_NATIVE_FPS": "0.8",
    "JADAL_EARLY_EXIT_S": "4",
    "JADAL_SKIP_MOSAIC": "1",
}


def _cpu_s():
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (s.ru_utime + s.ru_stime) + (c.ru_utime + c.ru_stime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--subset", default="all")
    ap.add_argument("--manifest", default=common.MANIFEST)
    ap.add_argument("--plan", default=None, help="plan JSON (else rebuilt from --subset/--shards)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap windows (debug)")
    args = ap.parse_args()

    # governor: one core per shard, and the adopted knobs
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for k, v in ADOPTED_ENV.items():
        os.environ.setdefault(k, v)

    os.makedirs(args.state_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log = open(os.path.join(args.state_dir, f"shard_{args.shard_index}.log"), "w")

    def say(msg):
        line = f"[shard {args.shard_index}/{args.shards}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    man, items = common.load_items(args.manifest)
    items = common.select_items(items, args.subset)
    if args.limit:
        items = items[:args.limit]
    if args.plan:
        plan = common.read_plan(args.plan)
    else:
        plan = common.make_plan(items, args.shards)
    if plan["K"] != args.shards:
        raise SystemExit(f"plan K={plan['K']} != --shards {args.shards}")
    mine = common.items_for_shard(items, plan, args.shard_index)
    order = {it["window_id"]: i for i, it in enumerate(items)}

    say(f"subset={args.subset} n_items={len(items)} mine={len(mine)} "
        f"clip_windows={sum(plan['clip_windows'][c] for c in plan['clip_windows'])}")
    say(f"windows={[it['window_id'] for it in mine][:3]}...")

    run_fast.init_worker()
    pmap = e8.profiles()
    opts = run_fast.read_opts()
    say(f"opts={opts}")

    # GPU utilisation sampler (GPU-wide; may include other pod jobs -> reported)
    import threading
    th = threading.Thread(target=run_fast.smi_sampler, daemon=True)
    th.start()

    t0 = time.perf_counter()
    cpu0 = _cpu_s()
    rows, errors = [], []
    for k, it in enumerate(mine, 1):
        r = run_fast._worker((it, pmap, False))
        if r["ok"]:
            rows.append(r["row"])
        else:
            errors.append(r)
        if k % 10 == 0 or k == len(mine):
            el = time.perf_counter() - t0
            say(f"  {k}/{len(mine)} ok={len(rows)} err={len(errors)} "
                f"{el:.0f}s {k/max(el,1e-9):.3f} win/s")

    wall = time.perf_counter() - t0
    cpu_s = _cpu_s() - cpu0
    run_fast.smi_stop.set()
    th.join(timeout=5)

    rows.sort(key=lambda r: order.get(r["window_id"], 1 << 30))
    common.write_jsonl(args.out, rows)

    digest = common.combined_digest(rows)
    meta = {
        "shard_index": args.shard_index, "shards": args.shards,
        "subset": args.subset, "plan_K": plan["K"],
        "assigned_clips": sorted(c for c, s in plan["assignment"].items()
                                 if s == args.shard_index),
        "n_windows": len(rows), "n_errors": len(errors), "errors": errors[:10],
        "wall_s": round(wall, 2), "windows_per_s": round(len(rows) / max(wall, 1e-9), 4),
        "cpu_s": round(cpu_s, 2), "cpu_cores": round(cpu_s / max(wall, 1e-9), 3),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "shard_digest": digest,
        "verdict_distribution": common.verdict_dist(rows),
        "optimizations": opts,
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "manifest": os.path.relpath(args.manifest, REPO),
        "out": os.path.relpath(args.out, REPO),
        "smi": run_fast.smi_log,
        "git_sha": e8.git_sha(),
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metap = os.path.join(args.state_dir, f"shard_{args.shard_index}.meta.json")
    json.dump(meta, open(metap, "w"), indent=1, sort_keys=True)
    say(f"DONE wall={wall:.1f}s windows={len(rows)} err={len(errors)} "
        f"digest={digest[:16]}")
    print("META " + json.dumps({k: v for k, v in meta.items() if k != "smi"}),
          flush=True)
    print("METAPATH " + metap, flush=True)


if __name__ == "__main__":
    main()
