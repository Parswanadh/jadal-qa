"""Determinism proof for the frozen artifact and the CPU reference path.

Proves, with real runs:
  1. 10 repeated `score()` calls on the same window give identical
     `features_digest`, `usable_prob` and `verdict`.
  2. The same holds with OMP_NUM_THREADS=1 vs 8 (run this script twice with
     different thread counts; the digests are compared across the two runs).
  3. The representation itself is byte-stable: extracting the same window twice
     gives an identical tensor digest.
  4. Which windows (if any) differ between the thread counts -- listed by id, not
     summarised away.

Receipts in this project are CONTENT-HASHED (SHA-256 of canonical JSON). They are
NOT cryptographically signed. This script proves reproducibility, not authenticity.

Usage
  OMP_NUM_THREADS=1 python eval/rhythm/determinism.py --out /tmp/det_t1.json
  OMP_NUM_THREADS=8 python eval/rhythm/determinism.py --out /tmp/det_t8.json
  python eval/rhythm/determinism.py --compare /tmp/det_t1.json /tmp/det_t8.json \
      --out eval/results/rhythm/determinism.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jadal.rhythm import features as F  # noqa: E402
from jadal.rhythm.scorer import score  # noqa: E402

GOLD = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")


def pick_windows(n: int = 5):
    """A deterministic spread across clips and labels."""
    items = json.load(open(GOLD))["items"]
    good = [i for i in items if i["label"] == 1]
    bad = [i for i in items if i["label"] == 0]
    picks = [good[0], good[len(good) // 2], bad[0], bad[len(bad) // 2], items[-1]]
    return picks[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None,
                    help="two per-thread JSONs to compare against each other")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    if args.compare:
        a = json.load(open(args.compare[0]))
        b = json.load(open(args.compare[1]))
        diffs = []
        keys = set(a["windows"]) | set(b["windows"])
        for k in sorted(keys):
            wa, wb = a["windows"].get(k), b["windows"].get(k)
            if wa is None or wb is None:
                diffs.append({"id": k, "reason": "missing in one run"})
                continue
            for field in ("features_digest", "usable_prob", "verdict"):
                if wa[field] != wb[field]:
                    diffs.append({"id": k, "field": field, "t_a": wa[field], "t_b": wb[field]})
        out = {
            "threads_a": a.get("threads"), "threads_b": b.get("threads"),
            "n_windows_compared": len(keys),
            "n_differing_fields": len(diffs),
            "differences": diffs,
            "verdict": "IDENTICAL across thread counts" if not diffs else "DIFFERENCES FOUND",
            "claim_scope": ("same machine, pinned stack, CPU reference path. Cross-device and "
                            "GPU-vs-CPU identity are NOT claimed."),
        }
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(args.out, "w"), indent=1)
        print(json.dumps({k: out[k] for k in
                          ("threads_a", "threads_b", "n_windows_compared",
                           "n_differing_fields", "verdict")}, indent=1))
        return 0

    picks = pick_windows()
    res = {
        "threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "repeats": args.repeats,
        "command": " ".join(sys.argv),
        "windows": {},
        "tensor_stability": {},
    }
    for it in picks:
        recs = []
        for _ in range(args.repeats):
            r = score(it["path"], it["start_s"], it["end_s"])
            recs.append((r["features_digest"], r["usable_prob"], r["verdict"]))
        digests = {d for d, _, _ in recs}
        probs = {p for _, p, _ in recs}
        verdicts = {v for _, _, v in recs}
        res["windows"][it["id"]] = {
            "clip": it["clip"], "label": it["label"], "reason": it.get("reason", ""),
            "features_digest": recs[0][0], "usable_prob": recs[0][1], "verdict": recs[0][2],
            "n_repeats": args.repeats,
            "distinct_digests": len(digests), "distinct_probs": len(probs),
            "distinct_verdicts": len(verdicts),
            "stable": len(digests) == 1 and len(probs) == 1 and len(verdicts) == 1,
        }
        # representation byte-stability, extracted twice from scratch
        d1 = F.tensor_digest(*[F.extract_window(it["path"], it["start_s"], it["end_s"])[k]
                               for k in ("X", "g")])
        d2 = F.tensor_digest(*[F.extract_window(it["path"], it["start_s"], it["end_s"])[k]
                               for k in ("X", "g")])
        res["tensor_stability"][it["id"]] = {"digest_1": d1, "digest_2": d2, "identical": d1 == d2}

    res["all_windows_stable"] = all(w["stable"] for w in res["windows"].values())
    res["all_tensors_identical"] = all(t["identical"] for t in res["tensor_stability"].values())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
    print(f"threads={res['threads']} windows={len(res['windows'])} "
          f"all_stable={res['all_windows_stable']} tensors_identical={res['all_tensors_identical']}")
    for k, w in res["windows"].items():
        print(f"  {k} {w['clip']:11s} digest {w['features_digest'][:12]} "
              f"p={w['usable_prob']} {w['verdict']:7s} stable={w['stable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
