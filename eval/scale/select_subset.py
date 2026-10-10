"""Seeded diverse subset selection for Egocentric-10K tars (no download).

Rule SELECT-53 (seed 53):
  1. Load per-tar sizes from eval/results/scale/repo_listing.json.
  2. Shuffle the 85 factories with random.Random(53).
  3. Walk the shuffled factories round-robin, taking part00.tar first, then
     part01, ... (one extra tar per factory per round) until adding another
     tar would exceed BUDGET_BYTES (30 GB).
  4. One worker per factory in this repo, so factory-diversity == worker-diversity.

Rationale: maximises distinct factories/workers (site/worker generalisation)
with few files per worker, under the 30 GB disk cap.

Output: eval/results/scale/selection.json {rule, seed, budget, status, files[]}
Download is a separate step (eval/scale/download_subset.py) because the
network has timed out before: resumable, retried, ffprobe-verified.
"""
import json
import random

LISTING = "eval/results/scale/repo_listing.json"
OUT = "eval/results/scale/selection.json"
SEED = 53
BUDGET_BYTES = 30_000_000_000  # 30 GB cap (disk has ~115 GB free; keep >= 60 GB free)


def main():
    with open(LISTING) as f:
        listing = json.load(f)
    tars = listing["tars"]
    by_factory = {}
    for t in tars:
        fac = t["path"].split("/")[0]
        by_factory.setdefault(fac, []).append(t)
    for fac in by_factory:
        by_factory[fac].sort(key=lambda t: t["path"])

    rng = random.Random(SEED)
    factories = sorted(by_factory)
    rng.shuffle(factories)

    chosen, used = [], 0
    round_idx = 0
    while True:
        progressed = False
        for fac in factories:
            files = by_factory[fac]
            if round_idx < len(files):
                cand = files[round_idx]
                if used + cand["size_bytes"] <= BUDGET_BYTES:
                    chosen.append(cand)
                    used += cand["size_bytes"]
                    progressed = True
        round_idx += 1
        if not progressed:
            break

    facs = sorted({c["path"].split("/")[0] for c in chosen})
    workers = sorted({"/".join(c["path"].split("/")[:2]) for c in chosen})
    sel = {
        "rule": "SELECT-53: seeded shuffle of factories, round-robin partNN, 30 GB cap",
        "seed": SEED,
        "budget_bytes": BUDGET_BYTES,
        "status": "pending_download (token 403 on gated content as of 2026-10-10; see report)",
        "n_files": len(chosen),
        "total_bytes": used,
        "total_GB": round(used / 1e9, 2),
        "n_factories": len(facs),
        "n_workers": len(workers),
        "factories": facs,
        "files": [c["path"] for c in chosen],
    }
    with open(OUT, "w") as f:
        json.dump(sel, f, indent=1)
    print(f"chose {sel['n_files']} tars, {sel['total_GB']} GB, "
          f"{sel['n_factories']} factories/workers -> {OUT}")


if __name__ == "__main__":
    main()
