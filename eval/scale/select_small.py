"""Seeded SMALL diverse selection: <= 8 GB, one shard per factory first.

Rule SELECT-8G (seed 8):
  1. Load per-tar sizes from eval/results/scale/repo_listing.json.
  2. Shuffle the 85 factories with random.Random(8).
  3. Round 1: smallest tar of each factory in shuffled order, while <= budget.
  4. Round 2+: next-smallest not-yet-chosen tar of each factory, while <= budget.
One shard per factory first => maximises factory/worker diversity per GB.
Output: eval/results/scale/selection_small.json (status pending_download).
"""
import json
import random

LISTING = "eval/results/scale/repo_listing.json"
OUT = "eval/results/scale/selection_small.json"
SEED = 8
BUDGET_BYTES = 8_000_000_000


def main():
    tars = json.load(open(LISTING))["tars"]
    by_factory = {}
    for t in tars:
        by_factory.setdefault(t["path"].split("/")[0], []).append(t)
    for fac in by_factory:
        by_factory[fac].sort(key=lambda t: t["size_bytes"])
    rng = random.Random(SEED)
    factories = sorted(by_factory)
    rng.shuffle(factories)
    chosen, used, idx = [], 0, 0
    while True:
        progressed = False
        for fac in factories:
            files = by_factory[fac]
            if idx < len(files) and used + files[idx]["size_bytes"] <= BUDGET_BYTES:
                chosen.append(files[idx])
                used += files[idx]["size_bytes"]
                progressed = True
        idx += 1
        if not progressed:
            break
    sel = {
        "rule": "SELECT-8G: seeded factory shuffle, smallest-first round-robin, 8 GB cap",
        "seed": SEED, "budget_bytes": BUDGET_BYTES, "status": "downloading",
        "n_files": len(chosen), "total_bytes": used, "total_GB": round(used / 1e9, 2),
        "n_factories": len({c["path"].split("/")[0] for c in chosen}),
        "n_workers": len({"/".join(c["path"].split("/")[i] for i in (0, 2)) for c in chosen}),
        "files": [{"path": c["path"], "size_bytes": c["size_bytes"]} for c in chosen],
    }
    json.dump(sel, open(OUT, "w"), indent=1)
    print(f"chose {sel['n_files']} tars, {sel['total_GB']} GB, "
          f"{sel['n_factories']} factories, {sel['n_workers']} workers")


if __name__ == "__main__":
    main()
