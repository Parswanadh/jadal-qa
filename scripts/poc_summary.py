"""Regenerate poc_data/summary.json from a deduped snapshot of windows.jsonl.

Reads <poc_data>/windows.jsonl, drops malformed lines, dedupes by window_id
(keeping the LATEST occurrence), writes <poc_data>/windows.clean.jsonl, then
writes <poc_data>/summary.json from the clean set.

Honesty: all rates are descriptive counts on UNLABELLED Egocentric-10K video,
NOT accuracy. partial=false only when --complete is passed (run finished);
otherwise partial=true.

Usage:
  OMP_NUM_THREADS=2 python scripts/poc_summary.py [--complete]
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

POC_DATA = os.environ.get("POC_DATA_DIR", "/home/parshu/projects/jadal-qa-wt/poc_data")
WINDOWS = os.path.join(POC_DATA, "windows.jsonl")
CLEAN = os.path.join(POC_DATA, "windows.clean.jsonl")
SUMMARY = os.path.join(POC_DATA, "summary.json")


def pct(a, b):
    return round(a / max(b, 1), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complete", action="store_true",
                    help="run worker finished: mark partial=false")
    ap.add_argument("--data-dir", default=POC_DATA)
    args = ap.parse_args()
    base = args.data_dir
    windows = os.path.join(base, "windows.jsonl")
    clean = os.path.join(base, "windows.clean.jsonl")
    summary = os.path.join(base, "summary.json")

    order = []
    latest = {}
    malformed = 0
    total_lines = 0
    with open(windows) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                o = json.loads(line)
                wid = o["window_id"]
            except Exception:
                malformed += 1
                continue
            if wid not in latest:
                order.append(wid)
            latest[wid] = o
    recs = [latest[w] for w in order]

    with open(clean, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    by_verdict = Counter()
    by_stage = Counter()
    by_reason = Counter()
    by_factory = {}
    by_worker = {}
    dec_paths = Counter()
    label_sources = Counter()
    rtfs = []
    lat_feat, lat_total = [], []
    missing_thumbs = 0
    for r in recs:
        by_verdict[r.get("verdict")] += 1
        by_stage[r.get("stage")] += 1
        for reason in r.get("reasons", []):
            by_reason[reason.get("code")] += 1
        f = by_factory.setdefault(r.get("factory", "?"),
                                  {"n": 0, "accept": 0, "reject": 0, "review": 0})
        f["n"] += 1
        if r.get("verdict") in f:
            f[r["verdict"]] += 1
        wkey = f"{r.get('factory', '?')}/{r.get('worker', '?')}"
        w = by_worker.setdefault(wkey, {"n": 0, "accept": 0, "reject": 0, "review": 0})
        w["n"] += 1
        if r.get("verdict") in w:
            w[r["verdict"]] += 1
        dec_paths[r.get("decode_path", "?")] += 1
        lat = (r.get("latency_ms") or {})
        if isinstance(lat.get("features"), (int, float)):
            lat_feat.append(lat["features"])
        if isinstance(lat.get("total"), (int, float)):
            lat_total.append(lat["total"])
        dur = (r.get("end_s", 0) or 0) - (r.get("start_s", 0) or 0)
        tot = lat.get("total")
        if isinstance(tot, (int, float)) and dur > 0:
            rtfs.append(round(dur * 1000.0 / tot, 3))
        for t in r.get("thumbs", []):
            if not os.path.exists(os.path.join(base, t)):
                missing_thumbs += 1
    rtfs.sort()
    hours = round(sum((r.get("end_s", 0) or 0) - (r.get("start_s", 0) or 0)
                      for r in recs) / 3600, 3)
    summ = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_note": "descriptive rates on unlabelled video, NOT accuracy",
        "label_source": "none",
        "label_sources_seen": dict(label_sources),
        "n_windows": len(recs),
        "n_lines_read": total_lines,
        "n_duplicates_dropped": total_lines - len(recs),
        "n_malformed_dropped": malformed,
        "hours": hours,
        "by_verdict": dict(by_verdict),
        "by_stage": dict(by_stage),
        "by_reason": dict(by_reason),
        "by_factory": by_factory,
        "by_worker": by_worker,
        "decode_path": dict(dec_paths),
        "latency_ms_p50": {
            "features": (round(sorted(lat_feat)[len(lat_feat) // 2], 1) if lat_feat else None),
            "total": (round(sorted(lat_total)[len(lat_total) // 2], 1) if lat_total else None)},
        "reject_share": pct(by_verdict.get("reject", 0), len(recs)),
        "review_share": pct(by_verdict.get("review", 0), len(recs)),
        "sequential_rtf_per_window": {
            "n": len(rtfs),
            "p50": rtfs[len(rtfs) // 2] if rtfs else None,
            "note": ("wall-clock seconds of 10-s window per second of "
                     "sequential (single-process, no batching) latency; "
                     ">1 means faster than realtime"),
        },
        "missing_thumbs": missing_thumbs,
        "throughput": {"gpu": "RTX 4070 Laptop (NOT datacenter, NOT a Pi)"},
        "partial": not args.complete,
    }
    with open(summary, "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: summ[k] for k in (
        "n_windows", "hours", "by_verdict", "by_stage", "by_reason",
        "n_duplicates_dropped", "n_malformed_dropped", "missing_thumbs",
        "partial", "reject_share")}, indent=1))
    print(f"wrote {clean} ({len(recs)} windows) and {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
