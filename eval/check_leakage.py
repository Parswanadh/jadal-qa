"""Prove that no near-duplicate pair (pHash Hamming <= threshold) spans two splits.

Usage (from the repo root):

    /home/parshu/envs/jadal/bin/python -m eval.check_leakage \
        --splits eval/results/splits.json --cache eval/results/phash_cache.json

Prints the number of cross-split pairs at Hamming <= threshold (must be 0)
and exits nonzero otherwise. Uses the same multi-index blocking as
eval.make_splits so the full 30k x 30k comparison stays cheap.

NOT DONE: checks pHash leakage only; semantic near-duplicates with very
different hashes are out of scope.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from eval.make_splits import _block_keys, hamming

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def count_cross_split_pairs(
    splits: dict[str, list[str]],
    hashes: dict[str, str],
    threshold: int,
    names: Sequence[str] = ("train", "calibration", "test"),
) -> tuple[int, list]:
    """Count pairs with Hamming <= threshold whose members sit in different splits."""
    split_of: dict[str, str] = {}
    for name in names:
        for fid in splits[name]:
            split_of[fid] = name
    buckets: dict[tuple[int, int], list[str]] = {}
    for fid, h in hashes.items():
        if fid not in split_of:
            continue
        for key in _block_keys(h):
            buckets.setdefault(key, []).append(fid)
    cross = 0
    examples: list = []
    seen: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a_pos in range(len(members)):
            for b_pos in range(a_pos + 1, len(members)):
                a, b = members[a_pos], members[b_pos]
                if a == b or (a, b) in seen or (b, a) in seen:
                    continue
                seen.add((a, b))
                if split_of[a] == split_of[b]:
                    continue
                d = hamming(hashes[a], hashes[b])
                if d <= threshold:
                    cross += 1
                    if len(examples) < 5:
                        examples.append(
                            {"a": a, "split_a": split_of[a], "b": b, "split_b": split_of[b], "hamming": d}
                        )
    return cross, examples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check for cross-split near-duplicate leakage.")
    p.add_argument("--splits", type=Path, default=RESULTS_DIR / "splits.json")
    p.add_argument("--cache", type=Path, default=RESULTS_DIR / "phash_cache.json")
    p.add_argument("--threshold", type=int, default=4)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.splits.read_text())
    split_ids = {name: manifest["splits"][name]["frame_ids"] for name in ("train", "calibration", "test")}
    cache = json.loads(args.cache.read_text())
    hashes = cache["rows"] if isinstance(cache, dict) and "rows" in cache else cache
    cross, examples = count_cross_split_pairs(split_ids, hashes, args.threshold)
    total = sum(len(v) for v in split_ids.values())
    print(f"[check_leakage] {total} frames, threshold={args.threshold}: "
          f"cross-split pairs at Hamming <= threshold: {cross}")
    for ex in examples:
        print(f"  LEAK {ex['a']} ({ex['split_a']}) <-> {ex['b']} ({ex['split_b']}) hamming={ex['hamming']}")
    return 0 if cross == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
