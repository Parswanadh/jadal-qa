#!/usr/bin/env python3
"""Backup recording for the PoC demo (issue #64): renders a ~60-90 s mp4 from
stream_events.jsonl WITHOUT screen capture.

Each chunk becomes a title card (PIL-drawn, 1280x720): running-score
sparkline, chunk timeline coloured by decision, decision banner, evidence
text, and an honesty footer. Cards are stitched with ffmpeg to
docs/screens/poc_demo_backup.mp4.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image, ImageDraw  # noqa: E402

W, H = 1280, 720
COLORS = {"continue": (200, 160, 40), "accept": (60, 180, 80),
          "early_reject": (210, 60, 50)}


def draw_card(ev, history, n_total, out_path):
    img = Image.new("RGB", (W, H), (18, 20, 26))
    d = ImageDraw.Draw(img)
    dec = ev.get("decision", "continue")
    col = COLORS.get(dec, (150, 150, 150))
    d.rectangle([0, 0, W, 86], fill=col)
    d.text((24, 18), f"CHUNK {ev.get('chunk')}  t={ev.get('t')}s   "
                     f"{str(dec).upper()}   score={ev.get('running_score')}",
           fill=(10, 10, 10))
    # sparkline of running_score
    x0, y0, bw, bh = 60, 150, 1160, 220
    d.rectangle([x0, y0, x0 + bw, y0 + bh], outline=(90, 90, 100))
    d.text((x0, y0 - 34), "running_score (descriptive, unlabelled video)",
           fill=(200, 200, 200))
    pts = []
    scores = [h[0] for h in history]
    for i, h in enumerate(scores):
        x = x0 + (i / max(1, n_total - 1)) * bw
        y = y0 + bh - max(0.0, min(1.0, h)) * bh
        pts.append((x, y))
    for a, b in zip(pts, pts[1:]):
        d.line([a, b], fill=(90, 200, 255), width=3)
    # chunk timeline
    ty = 450
    cw = 1160 / max(1, n_total)
    for i in range(n_total):
        c = COLORS.get(history[i][1], (150, 150, 150)) if i < len(history) \
            else (60, 60, 66)
        d.rectangle([60 + i * cw, ty, 60 + (i + 1) * cw - 2, ty + 44], fill=c)
    d.text((60, ty + 56), "chunk timeline (amber=watching, green=accept, red=reject)",
           fill=(200, 200, 200))
    # evidence
    reason = ev.get("reason", {})
    s0 = reason.get("s0", {})
    ev_txt = ("evidence: " + (f"S0 rule {s0.get('rule')} "
                              f"{s0.get('feature')}={s0.get('value')} "
                              f"{s0.get('op')} {s0.get('threshold')} "
                              f"[{s0.get('scope', reason.get('scope', ''))}]")
              if s0 else f"reason: {json.dumps(reason)[:110]}")
    d.text((60, 560), ev_txt, fill=(230, 230, 230))
    d.text((60, 600), f"decode={ev.get('decode')} latency={ev.get('latency_ms')}ms "
                      f"engine=spectral-stream-or-local-fallback", fill=(150, 150, 150))
    d.text((60, 646), "DEMO RENDER from stream_events.jsonl: descriptive rates on "
                      "unlabelled Egocentric-10K video - NOT accuracy.",
           fill=(140, 140, 140))
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="/home/parshu/projects/jadal-qa-wt/poc_data/stream_events.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hold-s", type=float, default=5.0,
                    help="seconds per chunk card (padded to >=60 s total)")
    args = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    out = args.out or os.path.join(root, "docs", "screens", "poc_demo_backup.mp4")
    evs = [json.loads(l) for l in open(args.events) if l.strip()]
    if not evs:
        raise SystemExit(f"no events in {args.events}")
    hist = [(e.get("running_score", 0), e.get("decision", "continue")) for e in evs]
    total = max(60.0, len(evs) * args.hold_s)
    hold = total / len(evs)
    fps = 30
    with tempfile.TemporaryDirectory(dir=os.path.join(root, ".scratch")) as tmp:
        sizes = []
        for i, e in enumerate(evs):
            card = os.path.join(tmp, f"card_{i:03d}.png")
            draw_card(e, hist[:i + 1], len(evs), card)
            sizes.append((card, int(round(hold * fps))))
        # concat via per-card loop counts
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as f:
            for card, nfr in sizes:
                f.write(f"file '{card}'\nduration {nfr / fps:.3f}\n")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "concat", "-safe", "0", "-i", lst,
               "-vf", "format=yuv420p", "-c:v", "libx264",
               "-r", str(fps), out]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise SystemExit(f"ffmpeg failed: {r.stderr[:500]}")
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", out],
        capture_output=True, text=True).stdout.strip())
    print(f"wrote {out} ({dur:.0f}s, {len(evs)} cards)")


if __name__ == "__main__":
    main()
