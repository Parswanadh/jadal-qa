#!/usr/bin/env python
"""Simulated head-camera dock: copy clips into the bucket one at a time.

Mimics the crowd-worker device that syncs clips to the hub: each clip is copied
into ``data_runtime/bucket/`` (the directory the service watches) with a delay
between clips, and after every copy a ``POST /ingest/event`` is sent for the
docked file. Run the service first, e.g.::

    /home/parshu/envs/jadal/bin/python -m uvicorn jadal.service:app
    /home/parshu/envs/jadal/bin/python sim/dock_sim.py --clips path/to/clips

Nothing happens at import time; all I/O is inside main(). No secrets, no API keys.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def post_event(base_url: str, path: Path, worker_id: str, site_id: str) -> dict:
    payload = json.dumps({
        "path": str(path), "worker_id": worker_id, "site_id": site_id,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/ingest/event", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulated clip dock for Jadal QA")
    parser.add_argument("--clips", required=True, help="directory of source clips")
    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="service base URL")
    parser.add_argument("--bucket", default=str(ROOT / "data_runtime" / "bucket"),
                        help="bucket directory to copy into")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds to wait between clips")
    parser.add_argument("--site", default="site-sim")
    parser.add_argument("--worker", default="w-sim")
    parser.add_argument("--loop", action="store_true",
                        help="keep cycling the clip directory forever")
    args = parser.parse_args(argv)

    src_dir = Path(args.clips)
    bucket = Path(args.bucket)
    bucket.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in src_dir.iterdir()
                   if p.is_file() and (p.suffix.lower() in VIDEO_SUFFIXES
                                       or p.name.endswith(".jadalclip")))
    if not clips:
        print(f"no clips found in {src_dir}")
        return 1

    print(f"docking {len(clips)} clips from {src_dir} -> {bucket} "
          f"(delay {args.delay}s), posting to {args.url}")
    while True:
        for clip in clips:
            dest = bucket / clip.name
            shutil.copy2(clip, dest)
            print(f"copied {clip.name} -> {dest}")
            try:
                result = post_event(args.url, dest, args.worker, args.site)
                batch = result.get("batch")
                print(f"  ingested {clip.name}: duplicate={result.get('duplicate')} "
                      f"batch={batch['batch_id'] if batch else None} "
                      f"({batch['batch_decision'] if batch else 'pending'})")
            except (urllib.error.URLError, OSError, ValueError) as exc:
                print(f"  ingest failed for {clip.name}: {exc}")
            time.sleep(args.delay)
        if not args.loop:
            break
    print("dock run finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
