"""Replay summary: stream_events.jsonl -> eval/results/poc/replay_summary.json.

Reads the live 2-second-chunk replay written by scripts/poc_stream_replay.py
(poc_up.sh --replay) and reports: event count, chunk latency p50 and max, the
first early_reject time against the scripted blackout onset, and peak RAM.

The blackout is injected by the replay script (--degrade blackout --degrade-at 12);
it is not an event in the footage. The onset is therefore an input (--blackout-at),
not something this file can verify.

Usage:
  python3 eval/poc/replay_summary.py [--events PATH] [--blackout-at 12] [--out PATH]

--events defaults to <resolved poc_data dir>/stream_events.jsonl
(scripts/poc_paths.py: JADAL_POC_DATA, then the sibling layout).
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import poc_paths  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "eval", "results", "poc", "replay_summary.json")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows, blackout_at):
    if not rows:
        raise ValueError("no events in stream_events.jsonl")
    lat = [float(r["latency_ms"]) for r in rows]
    ram = [float(r["ram_mb"]) for r in rows]
    decisions = {}
    for r in rows:
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
    rejects = [r for r in rows if r["decision"] == "early_reject"]
    first = min(rejects, key=lambda r: r["t"]) if rejects else None
    first_rule = None
    if first is not None:
        first_rule = ((first.get("reason") or {}).get("s0") or {}).get("rule")
    chunk_s = None
    ts = sorted(float(r["t"]) for r in rows)
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    if gaps:
        chunk_s = statistics.median(gaps)
    return {
        "events": len(rows),
        "window_ids": sorted({r["window_id"] for r in rows}),
        "chunk_s": chunk_s,
        "decisions": decisions,
        "chunk_latency_ms": {
            "p50": round(statistics.median(lat), 1),
            "max": round(max(lat), 1),
            "mean": round(statistics.fmean(lat), 1),
            "method": "median and max over every event; chunk compute only",
        },
        "first_early_reject": None if first is None else {
            "t_s": first["t"],
            "chunk": first["chunk"],
            "s0_rule": first_rule,
        },
        "scripted_blackout": {
            "t_s": blackout_at,
            "note": "injected by scripts/poc_stream_replay.py --degrade blackout; not from the footage",
        },
        "first_reject_minus_blackout_s": (
            None if first is None else round(float(first["t"]) - float(blackout_at), 3)),
        "ram_mb": {"max": round(max(ram), 1), "last": round(ram[-1], 1)},
        "decode": sorted({r.get("decode") for r in rows if r.get("decode")}),
        "s0_source": sorted({r.get("s0_source") for r in rows if r.get("s0_source")}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=None)
    ap.add_argument("--blackout-at", type=float, default=12.0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    events = args.events
    if events is None:
        try:
            events = os.path.join(poc_paths.resolve_poc_data(ROOT), "stream_events.jsonl")
        except poc_paths.PathError as e:
            print(f"replay_summary: {e}", file=sys.stderr)
            return 2
    rows = load(events)
    out = summarize(rows, args.blackout_at)
    out = {"source": os.path.basename(events), "label_note": "replay of a real clip with a scripted blackout; not accuracy",
           **out}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: out[k] for k in ("events", "chunk_latency_ms", "first_early_reject",
                                          "first_reject_minus_blackout_s", "ram_mb")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
