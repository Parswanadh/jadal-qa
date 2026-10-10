"""E4.03(a): re-encode OOD queue-1 1080p/30 HEVC source windows to the IndEgo
spec (480x480, 10 fps, H.264) and re-run the deterministic descriptors.

The point is to test whether matching the *container* spec shrinks the
descriptor gap to IndEgo (it is expected not to: the gap is content).

Writes JSONL: {window_id, set:'ood1_reenc10', clip, label, ok, desc, src}
        + the encoded file path.
Frames are NOT committed; encoded clips go to a scratch dir.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from descriptors import describe_window

Q = "/workspace/jadal_pod_bundle/queues/label_queue_random.json"
LAB = "/workspace/jadal_pod_bundle/labels/ood_labels.json"
SPEC = "scale=480:480:force_original_aspect_ratio=decrease,pad=480:480:(ow-iw)/2:(oh-ih)/2,fps=10"


def encode(src, start, dur, out):
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", src,
           "-t", f"{dur:.3f}", "-vf", SPEC, "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-an", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r.returncode == 0, r.stderr[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/tmp/opencode/e4_reenc")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    segs = json.load(open(Q))["segments"]
    lab = json.load(open(LAB))["latest"]
    if a.limit:
        segs = segs[:a.limit]
    recs = []
    for i, s in enumerate(segs):
        sid = s["segment_id"]
        v = lab.get(sid, {}).get("verdict")
        label = 1 if v == "good" else (0 if v == "bad" else None)
        dur = s["end_s"] - s["start_s"]
        out = os.path.join(a.scratch, f"{sid}.mp4")
        ok, err = encode(s["source_path"], s["source_start_s"], dur, out)
        if not ok:
            recs.append({"window_id": sid, "set": "ood1_reenc10", "clip": s["clip"],
                         "label": label, "ok": False, "error": err})
            continue
        try:
            d = describe_window(out, 0.0, dur)
            recs.append({"window_id": sid, "set": "ood1_reenc10", "clip": s["clip"],
                         "label": label, "ok": True, "desc": d, "src": s["source_path"],
                         "encoded": out})
        except Exception as e:  # noqa: BLE001
            recs.append({"window_id": sid, "set": "ood1_reenc10", "clip": s["clip"],
                         "label": label, "ok": False, "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(segs)}", flush=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("ok", sum(1 for r in recs if r["ok"]), "/", len(recs), "->", a.out, flush=True)


if __name__ == "__main__":
    main()
