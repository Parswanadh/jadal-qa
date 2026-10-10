"""Funnel round 3 (issue #56): real L0 signals for the human gold windows.

Reads the READ-ONLY snapshot ``eval/results/gold_labels_snapshot.json``
(copied from the live labeller file with its sha256) joined to
``eval/gold/segments.json`` (150 ten-second windows, 7 IndEgo clips).

Phase 1 cuts [start_s, end_s) and runs the REAL ``jadal.signals``
``compute_signals`` in a multiprocessing pool (``--jobs`` <= 6, CPU
only; mediapipe models via ``JADAL_MODELS_DIR``). Near-duplicate
resolution stays sequential in manifest order (exact reuse of
``build_gold_records`` logic via injected ``cut_fn``/``signals_fn``).

Writes:
  eval/results/gold_l0_signals.json  (per-segment signals + backend versions)
  eval/results/funnel_calibration_gold.json (calibrate_two_sided output with
      schema-checked human_provenance + sidecar, same shape as
      calibrate_funnel.py --from-gold)

Usage:
    JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models \\
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \\
        eval/compute_gold_l0.py --jobs 6
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.calibrate_funnel import (  # noqa: E402
    build_gold_records,
    check_gold_labels,
    file_sha256,
    markdown_tables,
)
from jadal import funnel  # noqa: E402


def _one(job: dict) -> dict:
    """Cut one window and compute its L0 signals (pool worker)."""
    from jadal import signals as signals_mod
    from eval.calibrate_funnel import cut_segment
    cut = cut_segment(job["path"], job["start_s"], job["end_s"],
                      Path(job["out"]), recut=False)
    signals = dict(signals_mod.compute_signals(str(cut)))
    signals["duration_s"] = round(job["end_s"] - job["start_s"], 1)
    return {"segment_id": job["segment_id"], "signals": signals}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "eval" / "results"
                        / "gold_labels_snapshot.json")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "eval" / "gold" / "segments.json")
    parser.add_argument("--cuts-dir", type=Path,
                        default=ROOT / "data_runtime" / "gold_cuts")
    parser.add_argument("--signals-out", type=Path,
                        default=ROOT / "eval" / "results"
                        / "gold_l0_signals.json")
    parser.add_argument("--calibration-out", type=Path,
                        default=ROOT / "eval" / "results"
                        / "funnel_calibration_gold.json")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--min-accepted", type=int, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 6:
        raise SystemExit("--jobs must be in [1, 6]")

    manifest = json.loads(args.manifest.read_text())
    labels = json.loads(args.snapshot.read_text())
    gold_check = check_gold_labels(labels)
    if not gold_check["ok"]:
        raise SystemExit(
            f"snapshot {args.snapshot} failed the gold schema check "
            f"({'; '.join(gold_check['reasons'])})")
    latest = labels["latest"]
    jobs = [{
        "segment_id": s["segment_id"], "path": s["path"],
        "start_s": float(s["start_s"]), "end_s": float(s["end_s"]),
        "out": str(args.cuts_dir / f"{s['segment_id']}.mp4"),
    } for s in manifest["segments"] if s["segment_id"] in latest]
    print(f"labeled windows: {len(jobs)} / {len(manifest['segments'])}")

    from jadal import signals as signals_mod
    versions = signals_mod.backend_versions()
    print(f"signal backends: {versions} (jobs={args.jobs})")
    with Pool(args.jobs) as pool:
        results = pool.map(_one, jobs)
    cached = {r["segment_id"]: r["signals"] for r in results}
    args.signals_out.parent.mkdir(parents=True, exist_ok=True)
    args.signals_out.write_text(json.dumps({
        "labels_snapshot_sha256": file_sha256(args.snapshot),
        "n_windows": len(cached),
        "signal_backend_versions": versions,
        "signals": cached,
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.signals_out} ({len(cached)} windows)")

    # Exact reuse of the --from-gold join: cuts exist, signals cached.
    def _cut_fn(src, start_s, end_s, out_path, recut=False):
        assert Path(out_path).is_file(), f"missing cut {out_path}"
        return Path(out_path)

    def _signals_fn(cut, segment):
        key = Path(cut).stem
        if key not in cached:
            raise KeyError(f"no cached signals for {key}")
        return dict(cached[key])

    records = build_gold_records(manifest, labels, signals_fn=_signals_fn,
                                 cuts_dir=args.cuts_dir, cut_fn=_cut_fn)
    result = funnel.calibrate_two_sided(records, {
        "target_alpha": args.alpha, "target_beta": args.beta,
        "delta": args.delta, "min_support": args.min_support,
        "min_accepted": args.min_accepted, "seed": args.seed})
    result["meta"]["provenance"] = f"gold-join-snapshot:{args.snapshot}"
    result["meta"]["human_provenance"] = {
        "ok": True,
        "schema": "gold-labels-v1",
        "file": str(args.snapshot),
        "sha256": file_sha256(args.snapshot),
        "labellers": gold_check["labellers"],
        "n_entries": gold_check["n_entries"],
    }
    result["sidecar"] = {
        "produced_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.calibration_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(markdown_tables(result))
    print(f"wrote {args.calibration_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
