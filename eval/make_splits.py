"""Jadal QA evaluation harness: dataset split construction.

Usage (from the repo root):

    /home/parshu/envs/jadal/bin/python -m eval.make_splits --report-only
    /home/parshu/envs/jadal/bin/python -m eval.make_splits
    /home/parshu/envs/jadal/bin/python -m eval.make_splits --help

What it does
------------
1. Reads the three Parquet files in the ego10k evaluation directory
   (/home/parshu/projects/awing/data/ego10k_eval by default) and prints the
   dataset structure: per-file row counts and the label distributions
   (``hand_count`` and ``active_labor``) per ``source_dataset``.
2. Streams the images in small batches (memory-safe: decode with PIL,
   immediately convert to 64x64 grayscale for the pHash, close the image),
   computes a pHash per frame.
3. Removes near-duplicate frames (pHash Hamming distance <= 4) BEFORE
   splitting, so no near-duplicate pair can end up in two different splits.
4. Assigns the surviving frames to train / calibration / test with a
   deterministic seeded hash order inside each stratum
   (source_dataset x hand_count x active_labor), so the split is stratified
   by source and hand count and keeps the label balance.
5. Writes the manifests and a report JSON under ``eval/results/``.

The ``active_labor`` labels in these files were produced by gemini-2.5-flash
and were NOT human-checked. See ``eval/README.md``.

NOT DONE:
- No model baselines live here yet (see ``eval/README.md``); this module only
  builds the split manifests.
- ``jadal.certify`` does not exist on the branch this was written against, so
  nothing in ``eval`` imports it yet except an optional try/except inside
  ``eval/calibration_report.py``.
- The pHash cache is invalidated by ``PHASH_VERSION`` only; if the input
  parquet files change, delete ``eval/results/phash_cache.json`` or pass
  ``--no-cache``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image
import imagehash
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/home/parshu/projects/awing/data/ego10k_eval")
PARQUET_FILES = ("ego4d.parquet", "egocentric_10k.parquet", "epic_kitchens.parquet")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

COLUMNS = ["frame_id", "image", "source_dataset", "hand_count", "active_labor"]
HAND_COUNTS = (0, 1, 2)
LABEL_VALUES = ("yes", "no")

# Bump when the pHash recipe changes; the on-disk cache is then ignored.
PHASH_VERSION = "imagehash-4.3.2/phash/gray64-lanczos-hash8"

CACHE_NAME = "phash_cache.json"
DEFAULT_SEED = 1234
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_DUP_THRESHOLD = 4
DEFAULT_WORKERS = 4
DEFAULT_BATCH_SIZE = 256
CHUNK = 32  # rows submitted to the thread pool at once (bounds decode memory)

SPLIT_NAMES = ("train", "calibration", "test")


def log(msg: str) -> None:
    print(f"[make_splits] {msg}", flush=True)


# ---------------------------------------------------------------------------
# stage 1: metadata (no images touched)
# ---------------------------------------------------------------------------


def _dist(values: Iterable[object]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def scan_metadata(data_dir: Path, files: Sequence[str] = PARQUET_FILES) -> dict:
    """Row counts and label distributions per file, without decoding images."""
    report: dict = {"data_dir": str(data_dir), "files": {}, "frame_id_duplicates": []}
    seen_frames: dict[str, str] = {}
    for name in files:
        path = data_dir / name
        pf = pq.ParquetFile(path)
        rows = 0
        hand_counts: list[int] = []
        labels: list[str] = []
        sources: list[str] = []
        id_lengths: list[int] = []
        for batch in pf.iter_batches(batch_size=2048, columns=COLUMNS):
            for r in batch.to_pylist():
                rows += 1
                hand_counts.append(int(r["hand_count"]))
                labels.append(str(r["active_labor"]))
                sources.append(str(r["source_dataset"]))
                id_lengths.append(len(r["frame_id"]))
                prev = seen_frames.get(r["frame_id"])
                if prev is not None and prev != name:
                    report["frame_id_duplicates"].append({"frame_id": r["frame_id"], "files": [prev, name]})
                seen_frames[r["frame_id"]] = name
        cross: dict[str, dict[str, int]] = {}
        for src, hand, label in zip(sources, hand_counts, labels):
            key = f"{src}|hand_count={hand}"
            cross.setdefault(key, {})
            cross[key][f"active_labor={label}"] = cross[key].get(f"active_labor={label}", 0) + 1
        report["files"][name] = {
            "rows": rows,
            "distinct_source_dataset": _dist(sources),
            "hand_count": _dist(hand_counts),
            "active_labor": _dist(labels),
            "hand_count_x_active_labor": {k: dict(sorted(v.items())) for k, v in sorted(cross.items())},
            "frame_id_len_min": min(id_lengths) if id_lengths else 0,
            "frame_id_len_max": max(id_lengths) if id_lengths else 0,
        }
    report["total_rows"] = sum(f["rows"] for f in report["files"].values())
    report["distinct_frame_ids"] = len(seen_frames)
    return report


def print_metadata_report(report: dict) -> None:
    print("\n=== dataset report (metadata only, no images decoded) ===")
    print(f"data_dir: {report['data_dir']}")
    header = f"{'file':<22}{'rows':>8}  {'sources':<20}{'hand_count':<20}{'active_labor':<20}"
    print(header)
    print("-" * len(header))
    for name, f in report["files"].items():
        sources = ",".join(f"{k}:{v}" for k, v in f["distinct_source_dataset"].items())
        hands = ",".join(f"{k}:{v}" for k, v in f["hand_count"].items())
        labels = ",".join(f"{k}:{v}" for k, v in f["active_labor"].items())
        print(f"{name:<22}{f['rows']:>8}  {sources:<20}{hands:<20}{labels:<20}")
    print(f"{'TOTAL':<22}{report['total_rows']:>8}  distinct frame_ids: {report['distinct_frame_ids']}")
    if report["frame_id_duplicates"]:
        print(f"WARNING: {len(report['frame_id_duplicates'])} frame_id value(s) appear in more than one file")
    print("\n--- label distribution by source_dataset (rows) ---")
    hcols = ["hand_count=0", "hand_count=1", "hand_count=2", "active_labor=yes", "active_labor=no", "rows"]
    print(f"{'source_dataset':<18}" + "".join(f"{c:>19}" for c in hcols))
    print("-" * (18 + 19 * len(hcols)))
    per_source: dict[str, dict[str, int]] = {}
    for name, f in report["files"].items():
        for key, labels in f["hand_count_x_active_labor"].items():
            src, hand = key.split("|")
            cell = per_source.setdefault(src, {})
            cell[hand] = cell.get(hand, 0) + sum(labels.values())
            for label, n in labels.items():
                cell[label] = cell.get(label, 0) + n
            cell["rows"] = cell.get("rows", 0) + sum(labels.values())
    for src in sorted(per_source):
        cell = per_source[src]
        print(f"{src:<18}" + "".join(f"{cell.get(c, 0):>19}" for c in hcols))
    print(f"{'TOTAL':<18}" + "".join(f"{sum(per_source[s].get(c, 0) for s in per_source):>19}" for c in hcols))
    print("")


# ---------------------------------------------------------------------------
# stage 2: pHash streaming (memory-safe)
# ---------------------------------------------------------------------------


def _hash_row(row: dict) -> dict:
    """Decode one image, downscale immediately, compute the pHash, close the image."""
    with Image.open(BytesIO(row["image"]["bytes"])) as img:
        width, height = img.size
        gray = img.convert("L").resize((64, 64), Image.LANCZOS)
        phash = imagehash.phash(gray)
    return {
        "frame_id": row["frame_id"],
        "source_dataset": str(row["source_dataset"]),
        "hand_count": int(row["hand_count"]),
        "active_labor": str(row["active_labor"]),
        "phash": str(phash),
        "width": int(width),
        "height": int(height),
    }


def hash_parquet(
    path: Path,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    only_frames: set[str] | None = None,
) -> list[dict]:
    """Stream one parquet file and return one record per frame (no images kept).

    ``only_frames`` restricts decoding to those frame_ids (frames already in the
    cache are skipped, so a re-run only pays for the missing ones).
    """
    out: list[dict] = []
    pf = pq.ParquetFile(path)
    seen = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
            rows = [
                r
                for r in batch.to_pylist()
                if only_frames is None or r["frame_id"] in only_frames
            ]
            if limit is not None and seen + len(rows) > limit:
                rows = rows[: limit - seen]
            for start in range(0, len(rows), CHUNK):
                out.extend(pool.map(_hash_row, rows[start : start + CHUNK]))
            seen += len(rows)
            if limit is not None and seen >= limit:
                break
    return out


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("phash_version") != PHASH_VERSION:
        return {}
    return data.get("rows", {})


def save_cache(path: Path, records: Iterable[dict]) -> None:
    payload = {
        "phash_version": PHASH_VERSION,
        "created_utc": _utc_now(),
        "rows": {r["frame_id"]: r["phash"] for r in records},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def hash_dataset(
    data_dir: Path,
    files: Sequence[str] = PARQUET_FILES,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    use_cache: bool = True,
    cache_path: Path | None = None,
) -> list[dict]:
    """pHash every frame of every file, using/refreshing the on-disk cache.

    One cheap metadata pass decides which frames still need decoding, a second
    streaming pass decodes only those.
    """
    cache_path = cache_path or (RESULTS_DIR / CACHE_NAME)
    cache = load_cache(cache_path) if use_cache else {}
    records: list[dict] = []
    for name in files:
        path = data_dir / name
        pf = pq.ParquetFile(path)
        total = pf.metadata.num_rows if limit is None else min(limit, pf.metadata.num_rows)
        cached_rows: list[dict] = []
        need: set[str] = set()
        seen = 0
        for batch in pf.iter_batches(batch_size=2048, columns=COLUMNS):
            for r in batch.to_pylist():
                seen += 1
                if limit is not None and seen > limit:
                    break
                if r["frame_id"] in cache:
                    cached_rows.append(
                        {
                            "frame_id": r["frame_id"],
                            "source_dataset": str(r["source_dataset"]),
                            "hand_count": int(r["hand_count"]),
                            "active_labor": str(r["active_labor"]),
                            "phash": cache[r["frame_id"]],
                            "width": -1,
                            "height": -1,
                        }
                    )
                else:
                    need.add(r["frame_id"])
            if limit is not None and seen >= limit:
                break
        log(f"  {name}: {total} frames, {len(need)} to decode, {len(cached_rows)} from cache")
        fresh = hash_parquet(path, workers=workers, batch_size=batch_size, limit=limit, only_frames=need or None)
        records.extend(cached_rows)
        records.extend(fresh)
    if use_cache:
        save_cache(cache_path, records)
    return records


# ---------------------------------------------------------------------------
# stage 3: near-duplicate removal (pHash Hamming <= threshold)
# ---------------------------------------------------------------------------


def hamming(a: str, b: str) -> int:
    """Hamming distance between two 16-hex-digit pHash strings."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _block_keys(phash: str, blocks: int = 4) -> Iterator[tuple[int, int]]:
    """Blocking keys: with 4 blocks, Hamming <= 4 forces one block to differ by <= 1 bit."""
    value = int(phash, 16)
    bits = 64 // blocks
    mask = (1 << bits) - 1
    for b in range(blocks):
        block = (value >> (b * bits)) & mask
        yield (b, block)
        for i in range(bits):
            yield (b, block ^ (1 << i))


def dedup(records: Sequence[dict], threshold: int = DEFAULT_DUP_THRESHOLD) -> tuple[list[dict], dict]:
    """Collapse near-duplicate frames (pHash Hamming <= threshold) into one representative.

    Returns (kept_records, report). The representative of a duplicate group is the
    smallest frame_id in that group, so the result does not depend on input order.
    """
    n = len(records)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, rec in enumerate(records):
        for key in _block_keys(rec["phash"]):
            buckets.setdefault(key, []).append(idx)

    pairs_checked = 0
    bucket_skipped = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > 400:  # degenerate bucket (all-identical hashes); keep it cheap
            bucket_skipped += 1
            continue
        for pos in range(len(members)):
            for other in range(pos + 1, len(members)):
                i, j = members[pos], members[other]
                pairs_checked += 1
                if hamming(records[i]["phash"], records[j]["phash"]) <= threshold:
                    union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    kept: list[dict] = []
    removed = 0
    clusters = 0
    removed_map: list[dict] = []
    for root in sorted(groups):
        members = groups[root]
        if len(members) > 1:
            clusters += 1
        # representative = smallest frame_id
        ordered = sorted(members, key=lambda i: (records[i]["frame_id"], records[i]["source_dataset"]))
        keep_idx = ordered[0]
        kept.append(records[keep_idx])
        for other in ordered[1:]:
            removed += 1
            removed_map.append(
                {
                    "frame_id": records[other]["frame_id"],
                    "kept_frame_id": records[keep_idx]["frame_id"],
                    "hamming": hamming(records[other]["phash"], records[keep_idx]["phash"]),
                }
            )
    report = {
        "input_rows": n,
        "kept_rows": len(kept),
        "removed_rows": removed,
        "duplicate_groups": clusters,
        "hash_pairs_checked": pairs_checked,
        "skipped_large_buckets": bucket_skipped,
        "threshold_hamming": threshold,
        "removed": removed_map,
    }
    return kept, report


# ---------------------------------------------------------------------------
# stage 4: stratified split
# ---------------------------------------------------------------------------


def _bucket(frame_id: str, seed: int, cell: tuple) -> str:
    key = f"{seed}|{cell[0]}|{cell[1]}|{cell[2]}|{frame_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def assign_splits(
    records: Sequence[dict],
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    min_cell_for_split: int = 4,
) -> dict:
    """Assign each record to train/calibration/test, stratified per (source, hand_count, label).

    Deterministic: the order inside a stratum comes from a seeded SHA-256 of the
    frame_id, so the same records + seed always give the same assignment.
    """
    cells: dict[tuple, list[dict]] = {}
    for rec in records:
        cell = (rec["source_dataset"], int(rec["hand_count"]), str(rec["active_labor"]))
        cells.setdefault(cell, []).append(rec)

    assignment: dict[str, str] = {}
    strata: dict[str, dict[str, int]] = {}
    for cell in sorted(cells):
        members = sorted(cells[cell], key=lambda r: (_bucket(r["frame_id"], seed, cell), r["frame_id"]))
        n = len(members)
        r_train, r_cal, r_test = (float(x) for x in ratios)
        if n < min_cell_for_split:
            counts = {"train": n, "calibration": 0, "test": 0}
        else:
            n_test = min(n, math.ceil(r_test * n))
            n_cal = min(n - n_test, math.ceil(r_cal * n))
            counts = {"train": n - n_cal - n_test, "calibration": n_cal, "test": n_test}
        strata["|".join(str(c) for c in cell)] = counts
        pos = 0
        for split in SPLIT_NAMES:
            for _ in range(counts[split]):
                assignment[members[pos]["frame_id"]] = split
                pos += 1
    return {"assignment": assignment, "strata": strata}


def split_distribution(records: Sequence[dict], assignment: dict[str, str]) -> dict:
    """Per-split counts by source_dataset, hand_count and active_labor."""
    by_frame = {r["frame_id"]: r for r in records}
    out: dict[str, dict] = {name: {"n": 0, "by_source": {}, "by_hand_count": {}, "by_active_labor": {}, "by_cell": {}} for name in SPLIT_NAMES}
    for frame_id, split in assignment.items():
        rec = by_frame[frame_id]
        cell = out[split]
        cell["n"] += 1
        for field, target in (
            ("source_dataset", "by_source"),
            ("hand_count", "by_hand_count"),
            ("active_labor", "by_active_labor"),
        ):
            key = str(rec[field])
            cell[target][key] = cell[target].get(key, 0) + 1
        ckey = f"{rec['source_dataset']}|hand_count={rec['hand_count']}|active_labor={rec['active_labor']}"
        cell["by_cell"][ckey] = cell["by_cell"].get(ckey, 0) + 1
    for cell in out.values():
        for target in ("by_source", "by_hand_count", "by_active_labor", "by_cell"):
            cell[target] = dict(sorted(cell[target].items()))
    return out


def build_manifest(
    records: Sequence[dict],
    assignment: dict[str, str],
    seed: int,
    ratios: Sequence[float],
    dup_report: dict,
    metadata_report: dict,
    data_dir: Path,
    files: Sequence[str],
    dup_threshold: int,
) -> dict:
    per_split = split_distribution(records, assignment)
    frame_ids = {name: sorted(fid for fid, s in assignment.items() if s == name) for name in SPLIT_NAMES}
    total = sum(len(v) for v in frame_ids.values())
    assert total == len(assignment)
    sizes = {name: (round(len(frame_ids[name]) / total, 4) if total else 0.0) for name in SPLIT_NAMES}
    manifest = {
        "created_utc": _utc_now(),
        "tool": "eval/make_splits.py",
        "data_dir": str(data_dir),
        "files": list(files),
        "seed": int(seed),
        "ratios": {"train": ratios[0], "calibration": ratios[1], "test": ratios[2]},
        "phash_version": PHASH_VERSION,
        "dedup": {k: v for k, v in dup_report.items() if k != "removed"},
        "duplicate_map_entries": len(dup_report.get("removed", [])),
        "dataset_report": {
            "total_rows": metadata_report["total_rows"],
            "distinct_frame_ids": metadata_report["distinct_frame_ids"],
            "files": {
                name: {
                    "rows": f["rows"],
                    "distinct_source_dataset": f["distinct_source_dataset"],
                    "hand_count": f["hand_count"],
                    "active_labor": f["active_labor"],
                    "hand_count_x_active_labor": f["hand_count_x_active_labor"],
                }
                for name, f in metadata_report["files"].items()
            },
        },
        "label_source": "gemini-2.5-flash auto-labels, NOT human-checked (see eval/README.md)",
        "strata_counts": assignment["strata"],
        "splits": {
            name: {
                "count": len(frame_ids[name]),
                "fraction": sizes[name],
                "frame_ids": frame_ids[name],
                "by_source": per_split[name]["by_source"],
                "by_hand_count": per_split[name]["by_hand_count"],
                "by_active_labor": per_split[name]["by_active_labor"],
                "by_cell": per_split[name]["by_cell"],
            }
            for name in SPLIT_NAMES
        },
        "sizes": sizes,
    }
    return manifest


def print_split_report(manifest: dict) -> None:
    print("\n=== split report ===")
    print(f"seed={manifest['seed']} ratios={manifest['ratios']} phash={manifest['phash_version']}")
    print(f"rows in dataset: {manifest['dataset_report']['total_rows']}, "
          f"kept after dedup: {manifest['dedup']['kept_rows']} "
          f"(removed {manifest['dedup']['removed_rows']} near-duplicates in "
          f"{manifest['dedup']['duplicate_groups']} groups, hamming <= {manifest['dedup']['threshold_hamming']})")
    print(f"\n{'split':<14}{'count':>8}{'fraction':>10}  " + f"{'sources':<46}{'hand_count':<20}{'active_labor':<16}")
    print("-" * 112)
    for name in SPLIT_NAMES:
        s = manifest["splits"][name]
        src = ",".join(f"{k.split('_')[0][:6]}:{v}" for k, v in s["by_source"].items())
        hand = ",".join(f"{k}:{v}" for k, v in s["by_hand_count"].items())
        lab = ",".join(f"{k}:{v}" for k, v in s["by_active_labor"].items())
        print(f"{name:<14}{s['count']:>8}{s['fraction']:>10}  {src:<46}{hand:<20}{lab:<16}")
    print("")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stratified train/calibration/test manifests for the ego10k eval set.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--ratios", type=str, default="70,15,15", help="train,calibration,test percentages")
    p.add_argument("--dup-threshold", type=int, default=DEFAULT_DUP_THRESHOLD, help="pHash Hamming distance treated as duplicate")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--limit", type=int, default=None, help="only use the first N rows of each file (smoke tests)")
    p.add_argument("--report-only", action="store_true", help="print the dataset report and skip image decoding")
    p.add_argument("--no-cache", action="store_true", help="ignore (and do not write) the pHash cache")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ratios = [float(x) / 100.0 for x in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"--ratios must be three numbers summing to 100, got {args.ratios!r}")

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log("stage 1/3: scanning parquet metadata (no images decoded)")
    metadata = scan_metadata(args.data_dir)
    print_metadata_report(metadata)
    (args.out_dir / "dataset_report.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    if args.report_only:
        log(f"report only, wrote {args.out_dir / 'dataset_report.json'} in {time.time() - started:.1f}s")
        return 0

    log("stage 2/3: computing pHashes (streaming, images downscaled immediately)")
    records = hash_dataset(
        args.data_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        use_cache=not args.no_cache,
        cache_path=args.out_dir / CACHE_NAME,
    )
    log(f"  hashed {len(records)} frames in {time.time() - started:.1f}s")

    log("stage 3/3: dedup + stratified split")
    kept, dup_report = dedup(records, threshold=args.dup_threshold)
    log(f"  kept {len(kept)}, removed {dup_report['removed_rows']} near-duplicates "
        f"({dup_report['duplicate_groups']} groups, {dup_report['hash_pairs_checked']} pairs checked)")
    split = assign_splits(kept, seed=args.seed, ratios=ratios)
    manifest = build_manifest(
        kept,
        split,
        seed=args.seed,
        ratios=ratios,
        dup_report=dup_report,
        metadata_report=metadata,
        data_dir=args.data_dir,
        files=PARQUET_FILES,
        dup_threshold=args.dup_threshold,
    )
    out_path = args.out_dir / "splits.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (args.out_dir / "duplicates_removed.json").write_text(
        json.dumps({"threshold_hamming": args.dup_threshold, "removed": dup_report["removed"]}, indent=2)
    )
    print_split_report(manifest)
    log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB) in {time.time() - started:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
