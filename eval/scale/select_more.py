"""Seeded EXTENDED diverse selection of Ego10K tars, biased to NEW factories.

Rule SELECT-MORE (seed 42):
  1. Load per-tar sizes from eval/results/exp_corpus/repo_listing_more.json.
  2. Exclude the 47 tars already downloaded (queues/selection_small.json).
  3. Round-robin, smallest-first, across factories NOT yet in the corpus
     ("new" factories) until the byte budget is exhausted; this maximises the
     number of NEW factories per GB and, at ~42 GB, reaches all 38 remaining
     factories (~6 tars each).
  4. Any leftover budget is filled round-robin from already-seen factories.

Output: eval/results/exp_corpus/selection_more.json (status pending_download).
"""
import json
import random

LISTING = "eval/results/exp_corpus/repo_listing_more.json"
EXISTING = "/workspace/jadal_pod_bundle/queues/selection_small.json"
OUT = "eval/results/exp_corpus/selection_more.json"
SEED = 42
BUDGET_BYTES = 42_000_000_000
MIN_FREE_BYTES = 15_000_000_000


def main():
    tars = json.load(open(LISTING))["tars"]
    existing = json.load(open(EXISTING))
    ex_paths = {f["path"] for f in existing["files"]}
    ex_facs = {p.split("/")[0] for p in ex_paths}
    avail = [t for t in tars if t["path"] not in ex_paths]

    by_factory = {}
    for t in avail:
        by_factory.setdefault(t["path"].split("/")[0], []).append(t)
    for fac in by_factory:
        by_factory[fac].sort(key=lambda t: t["size_bytes"])

    rng = random.Random(SEED)
    new_facs = sorted(f for f in by_factory if f not in ex_facs)
    old_facs = sorted(f for f in by_factory if f in ex_facs)
    rng.shuffle(new_facs)
    rng.shuffle(old_facs)

    chosen, used = [], 0

    def round_robin(facs):
        nonlocal used
        idx = 0
        while True:
            progressed = False
            for fac in facs:
                files = by_factory[fac]
                if idx < len(files) and used + files[idx]["size_bytes"] <= BUDGET_BYTES:
                    chosen.append(files[idx])
                    used += files[idx]["size_bytes"]
                    progressed = True
            idx += 1
            if not progressed:
                break

    round_robin(new_facs)
    round_robin(old_facs)

    sel = {
        "rule": "SELECT-MORE: new-factories-first seeded round-robin, smallest-first, 42 GB cap",
        "seed": SEED, "budget_bytes": BUDGET_BYTES, "status": "pending_download",
        "n_files": len(chosen), "total_bytes": used, "total_GB": round(used / 1e9, 2),
        "n_factories": len({c["path"].split("/")[0] for c in chosen}),
        "n_new_factories": len({c["path"].split("/")[0] for c in chosen} - ex_facs),
        "n_workers": len({"/".join(c["path"].split("/")[i] for i in (0, 2)) for c in chosen}),
        "files": [{"path": c["path"], "size_bytes": c["size_bytes"]} for c in chosen],
    }
    json.dump(sel, open(OUT, "w"), indent=1)
    print(f"chose {sel['n_files']} tars, {sel['total_GB']} GB, "
          f"{sel['n_factories']} factories ({sel['n_new_factories']} new), "
          f"{sel['n_workers']} workers")


if __name__ == "__main__":
    main()
