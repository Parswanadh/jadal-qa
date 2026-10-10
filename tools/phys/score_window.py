#!/usr/bin/env python3
"""CLI: score one window with the deterministic physics model.

Inputs (choose one):
  --features-json FILE   a cached feature record (see README for accepted shapes)
  --clip VIDEO --start-s S --duration D   decode + extract features on the fly

Optional deterministic logistic fallback (hybrid mode):
  --fallback-prob P      resolve a REVIEW window with the frozen ensemble prob
  --fallback-json FILE   look the prob up by --window-id

Emits the canonical JSON receipt on stdout and exits
  0 = ACCEPT, 1 = REJECT, 2 = REVIEW (2 only when no fallback resolved it).

Examples:
  python3 tools/phys/score_window.py --features-json rec.json --window-id g000
  python3 tools/phys/score_window.py --clip a.mp4 --start-s 0 --duration 10 \
      --fallback-prob 0.73 --pretty
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.phys import model  # noqa: E402


def _load_features(path: str, window_id: str | None) -> tuple[str, dict]:
    """Accept {features:{...}}, a flat dict, or a run_features-style record."""
    d = json.load(open(path))
    if isinstance(d, dict) and "features" in d:
        feats = dict(d["features"])
        wid = window_id or d.get("window_id")
    elif isinstance(d, dict) and ("full" in d):
        feats = {f"full.{k}": v for k, v in d["full"].items()}
        if "phys" in d:
            feats.update({f"phys.{k}": v for k, v in d["phys"].items()})
        wid = window_id or d.get("id")
    elif isinstance(d, dict) and isinstance(d.get("windows"), dict):
        wid = window_id or next(iter(d["windows"]))
        rec = d["windows"][wid]
        feats = {f"full.{k}": v for k, v in rec["full"].items()}
        if "phys" in rec:
            feats.update({f"phys.{k}": v for k, v in rec["phys"].items()})
    else:
        feats = dict(d)
        wid = window_id
    return (wid or "window"), feats


def _load_fallback(path: str, window_id: str) -> float | None:
    d = json.load(open(path))
    if isinstance(d, dict) and "scores" in d:
        for s in d["scores"]:
            if str(s.get("id")) == window_id:
                return float(s.get("usable_prob"))
        return None
    if isinstance(d, dict):
        if window_id in d:
            v = d[window_id]
            return float(v["usable_prob"]) if isinstance(v, dict) else float(v)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic physics model scorer.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--features-json", help="cached feature record JSON")
    src.add_argument("--clip", help="video path (decoded on the fly)")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--window-id", default=None)
    ap.add_argument("--rules", default=model.DEFAULT_RULES)
    ap.add_argument("--fallback-prob", type=float, default=None)
    ap.add_argument("--fallback-json", default=None)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    if args.features_json:
        wid, feats = _load_features(args.features_json, args.window_id)
    else:
        from tools.phys.extract_features import extract_window_features
        wid = args.window_id or os.path.basename(args.clip)
        feats = extract_window_features(args.clip, args.start_s, args.duration)

    rules = model.load_rules(args.rules)

    fb = args.fallback_prob
    fb_sha = None
    if fb is None and args.fallback_json:
        fb = _load_fallback(args.fallback_json, wid)
        if fb is not None:
            fb_sha = model.sha256_file(args.fallback_json)

    rec = model.receipt(wid, feats, rules, fallback_prob=fb,
                        fallback_sha256=fb_sha)
    if args.pretty:
        print(json.dumps(rec, indent=2, sort_keys=True))
    else:
        print(model.canonical_json(rec))
    return model.decision_exit_code(rec["decision"])


if __name__ == "__main__":
    raise SystemExit(main())
