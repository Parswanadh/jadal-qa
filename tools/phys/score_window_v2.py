#!/usr/bin/env python3
"""Packaged deterministic scorer for the v2 physics rule model.

Scores one 10-s window from a video clip (or from a pre-extracted feature JSON) and
prints the ACCEPT/REVIEW/REJECT verdict with full per-threshold evidence and a
canonical sha256 receipt.

  python3 tools/phys/score_window_v2.py --clip VIDEO --start-s 0 --duration 10
  python3 tools/phys/score_window_v2.py --features features.json --window-id g000
  python3 tools/phys/score_window_v2.py --clip VIDEO --domain-mode requantile \
      --target-features ood_features.json      # label-free target-side re-quantiling

The decision path is integer-only at Q=4 (see tools/phys/model_v2.py). No RNG, no
learned parameter, no float comparison after the quantisation boundary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.phys import model_v2 as M  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "phys_rules_v2.json")


def requantile(features: dict, target_rows: list[dict]) -> dict:
    """Label-free per-domain normalisation: replace each value by its within-target
    rank (uniform in (0,1)), using ONLY the target's own unlabelled values."""
    names = sorted({k for r in target_rows for k in r})
    out = {}
    for k in names:
        vals = np.array([float(r[k]) for r in target_rows
                         if k in r and np.isfinite(float(r[k]))])
        if vals.size == 0 or k not in features:
            continue
        v = float(features[k])
        if not np.isfinite(v):
            continue
        out[k] = float((int((vals < v).sum()) + 0.5) / vals.size)
    return out


def rank_thresholds(table: dict) -> dict:
    """Thresholds in rank space: the frozen quantile LEVELS, integer at Q=4."""
    levels = table.get("quantile_levels", [0.01, 0.05, 0.10, 0.25])
    out = {}
    for r in table["rules"]:
        out[r["rule"] + "|" + r["feature"]] = (
            [M.quantise(1.0 - l) for l in levels] if r["side"] == "high"
            else [M.quantise(l) for l in levels])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic v2 physics scorer.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--features", default=None, help="pre-extracted feature JSON")
    ap.add_argument("--window-id", default="")
    ap.add_argument("--domain-mode", choices=["absolute", "requantile"],
                    default="absolute")
    ap.add_argument("--target-features", default=None,
                    help="JSON list of unlabelled target-domain feature records "
                         "(required for --domain-mode requantile)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    table = M.load_table(args.config)
    if args.features:
        feats = json.load(open(args.features))
        feats = feats.get("features", feats)
    elif args.clip:
        from tools.phys.extract_features import extract_window_features
        feats = extract_window_features(args.clip, args.start_s, args.duration)
    else:
        print("need --clip or --features", file=sys.stderr)
        return 2

    if args.domain_mode == "requantile":
        if not args.target_features:
            print("--domain-mode requantile needs --target-features", file=sys.stderr)
            return 2
        target = json.load(open(args.target_features))
        target = [t.get("features", t) for t in target]
        feats = requantile(feats, target)
        table = dict(table)
        table["rules"] = [dict(r, thresholds_q=rank_thresholds(table)[
            r["rule"] + "|" + r["feature"]]) for r in table["rules"]]

    rec = M.decide(feats, table, window_id=args.window_id or
                   (os.path.basename(args.clip) if args.clip else ""))
    rec["domain_mode"] = args.domain_mode
    text = json.dumps(rec, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0 if rec["verdict"] != "REJECT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
