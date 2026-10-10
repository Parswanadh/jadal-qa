"""Merge per-shard fold results into one pooled LOCO result file.

Usage
  python eval/rhythm/merge_folds.py --shards eval/results/rhythm/loco_shard0.json \
      eval/results/rhythm/loco_shard1.json eval/results/rhythm/loco_shard2.json \
      --out eval/results/rhythm/loco_main.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    folds, cfgs, cmds = [], [], []
    for s in args.shards:
        d = json.load(open(s))
        folds.extend(d["folds"])
        cfgs.append(d.get("config"))
        cmds.append(d.get("command"))
    folds.sort(key=lambda f: f["fold"])

    out = {
        "label_source": "human gold (150 windows, 7 clips, one labeller) + synthetic-by-construction negatives",
        "protocol": "leave-one-clip-out over 7 clips; members selected by inner-LOCO on training clips only",
        "commands": cmds,
        "shards": [str(s) for s in args.shards],
        "configs": cfgs,
        "n_folds": len(folds),
        "held_out_clips": [f["held_out_clip"] for f in folds],
        "mean_fold_auroc": sum(f["auroc"] for f in folds) / len(folds),
        "folds": folds,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"merged {len(folds)} folds -> {args.out}")
    print(f"mean fold AUROC {out['mean_fold_auroc']:.3f}")
    for f in folds:
        print(f"  fold {f['fold']} held={f['held_out_clip']:11s} n={f['n_test']:3d} "
              f"bad={f['n_bad_test']:2d} AUROC={f['auroc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
