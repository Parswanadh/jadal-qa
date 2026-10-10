#!/usr/bin/env python3
"""Stream-replay CLI for the PoC demo (issue #64). See jadal/poc_stream/replay.py.

Example:
  scripts/poc_stream_replay.py --clip <ego10k mp4> --speed 20 --max-chunks 30 \\
      --out /home/parshu/projects/jadal-qa-wt/poc_data/stream_events.jsonl
  scripts/poc_stream_replay.py --clip <mp4> --degrade blackout --degrade-at 6 \\
      --max-chunks 12 --pico
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jadal.poc_stream.replay import replay


def main():
    ap = argparse.ArgumentParser(description="Replay a clip as a live worker stream.")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="/home/parshu/projects/jadal-qa-wt/poc_data/stream_events.jsonl")
    ap.add_argument("--chunk-s", type=float, default=2.0)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--degrade", default="none",
                    choices=["none", "blackout", "freeze", "blur"])
    ap.add_argument("--degrade-at", type=float, default=None)
    ap.add_argument("--accept-thr", type=float, default=0.5)
    ap.add_argument("--pico", action="store_true")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of starting a fresh file")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    summary = replay(args.clip, args.out, chunk_s=args.chunk_s,
                     window_s=args.window_s, speed=args.speed,
                     max_chunks=args.max_chunks, degrade=args.degrade,
                     degrade_at=args.degrade_at, accept_thr=args.accept_thr,
                     use_pico=args.pico, fps=args.fps, width=args.width,
                     use_cuda=args.cuda, append=args.append)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
