"""Headline counts for the PoC batch run, computed from windows.jsonl.

Writes eval/results/poc/run_summary.json (path overridable). Deduplicates by
window_id (keeps the LAST occurrence, as scripts/poc_summary.py does), so the
numbers match the dedupe even if the run appended a window twice.

Environment (all optional):
  JADAL_POC_DATA  directory holding windows.jsonl
                  (default: <repo>/../poc_data, the sibling of the worktree)
  JADAL_POC_SUMMARY_OUT  output JSON
                  (default: <repo>/eval/results/poc/run_summary.json)

Usage:
  python3 eval/poc/summarize.py [--complete]

--complete sets partial=false: the run finished its clip list. Without it the
file says partial=true. Verdict rates are descriptive counts on UNLABELLED
video, not accuracy.
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.environ.get("JADAL_POC_DATA", os.path.join(ROOT, "..", "poc_data"))
OUT = os.environ.get("JADAL_POC_SUMMARY_OUT",
                     os.path.join(ROOT, "eval", "results", "poc", "run_summary.json"))


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round(q / 100.0 * (len(sorted_vals) - 1))))
    return round(sorted_vals[idx], 1)


def load_unique(path):
    latest, order, lines, malformed = {}, [], 0, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                rec = json.loads(line)
                wid = rec["window_id"]
            except (ValueError, KeyError):
                malformed += 1
                continue
            if wid not in latest:
                order.append(wid)
            latest[wid] = rec
    return [latest[w] for w in order], lines, malformed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complete", action="store_true", help="mark partial=false")
    ap.add_argument("--windows", default=None, help="windows.jsonl path")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    path = args.windows or os.path.join(DATA, "windows.jsonl")

    recs, n_lines, malformed = load_unique(path)
    verdicts = Counter(r.get("verdict") for r in recs)
    reasons = Counter(rc.get("code") for r in recs for rc in r.get("reasons", []))
    stages = Counter(r.get("stage") for r in recs)
    factories = {r.get("factory") for r in recs}
    clips = {r.get("clip") for r in recs}
    secs = sum((r.get("end_s") or 0) - (r.get("start_s") or 0) for r in recs)
    total_ms = sorted(r["latency_ms"]["total"] for r in recs
                      if isinstance((r.get("latency_ms") or {}).get("total"), (int, float)))
    feat_ms = sorted(r["latency_ms"]["features"] for r in recs
                     if isinstance((r.get("latency_ms") or {}).get("features"), (int, float)))
    n = len(recs)
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": os.path.basename(path),
        "label_note": "descriptive rates on unlabelled video, NOT accuracy",
        "partial": not args.complete,
        "lines_read": n_lines,
        "duplicates_dropped": n_lines - malformed - n,
        "malformed_dropped": malformed,
        "unique_windows": n,
        "clips": len(clips),
        "factories": len(factories),
        "hours": round(secs / 3600, 3),
        "verdicts": {k: verdicts.get(k, 0) for k in ("accept", "reject", "review")},
        "verdict_share": {k: round(verdicts.get(k, 0) / max(n, 1), 4)
                          for k in ("accept", "reject", "review")},
        "stages": {k: stages.get(k, 0) for k in ("S0", "S1", "S2")},
        "reasons": dict(sorted(reasons.items())),
        "throughput": {
            "video_h_per_compute_h": round(secs / max(sum(total_ms) / 1000.0, 1e-9), 2),
            "note": ("video hours divided by the summed per-window sequential compute "
                     "time (latency_ms.total); concurrency and wall-clock are not included"),
        },
        "latency_ms": {
            "total_p50": percentile(total_ms, 50),
            "total_p90": percentile(total_ms, 90),
            "total_p99": percentile(total_ms, 99),
            "features_p50": percentile(feat_ms, 50),
            "features_p90": percentile(feat_ms, 90),
            "n": len(total_ms),
            "note": "sequential single-process per-window compute time from latency_ms",
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: out[k] for k in ("unique_windows", "verdicts", "hours",
                                           "factories", "partial")}))


if __name__ == "__main__":
    main()
