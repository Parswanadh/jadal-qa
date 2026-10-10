"""Gap-fill: arm-B T1 features for manifest windows missing from gold_T1.json.

Runs the features worker's extract() READ-ONLY (no edits, no writes to their
paths). Output (harness-owned): eval/results/spectral/b_missing.json in
arm-gold format. The scorer merges it with gold_T1.json as B-T1-150.
Missing windows get score:null (harness LOCO head scores them).

Usage: python -m eval.spectral.rerun_b_missing [--workers 4]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/home/parshu/projects/jadal-qa-wt/spectral-features")

DATA = Path("/home/parshu/projects/jadal-qa-wt/spectral_data")
OUT = HARNESS / "eval" / "results" / "spectral" / "b_missing.json"


def _init():
    sys.path.insert(0, "/home/parshu/projects/jadal-qa-wt/spectral-features")


def task(it: dict):
    from jadal.spectral import features as F
    t0 = time.perf_counter()
    feats = F.extract(it["path"], it["start_s"], it["end_s"])
    dt = time.perf_counter() - t0
    return it["id"], {k: float(v) for k, v in feats.items()}, round(dt, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    man = json.loads((DATA / "manifest_gold.json").read_text())
    t1 = json.loads((DATA / "arms" / "B" / "gold_T1.json").read_text())
    have = set(t1["items"])
    missing = [it for it in man["items"] if it["id"] not in have]
    print(f"manifest={len(man['items'])} have={len(have)} missing={len(missing)}",
          flush=True)
    if not missing:
        return
    from concurrent.futures import ProcessPoolExecutor
    items = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for i, feats, dt in ex.map(task, missing):
            items[i] = {"score": None, "features": feats, "latency_s": dt}
    OUT.write_text(json.dumps({"arm": "B", "tier": "T1-gapfill",
                               "version": t1.get("version"),
                               "items": items}, indent=1))
    lats = [v["latency_s"] for v in items.values()]
    print(f"wrote {OUT} n={len(items)} mean_lat={sum(lats)/len(lats):.2f}s",
          flush=True)


if __name__ == "__main__":
    main()
