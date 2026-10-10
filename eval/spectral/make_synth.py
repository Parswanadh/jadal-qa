#!/usr/bin/env python3
"""SYNTH generator: synthetic degradation set (by construction labels).

Reads /home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json (label==1 only),
picks ~6 sources deterministically (sorted by id, greedy distinct-clip cover, 2 per clip),
extracts 10-s clean controls + 9 degradations each.

Outputs:
  videos:   /home/parshu/projects/jadal-qa-wt/spectral_data/synth/*.mp4
  manifest: /home/parshu/projects/jadal-qa-wt/spectral_data/manifest_synth.json
  marker:   /home/parshu/projects/jadal-qa-wt/spectral_data/READY_synth
Temp: .scratch/synth/ under this worktree.

Video spec: 480x480 @ 10 fps, exactly 10 s (100 frames), no audio,
libx264 -crf 18 -preset veryfast -pix_fmt yuv420p. Verified with ffprobe.

Python: /home/parshu/envs/jadal/bin/python. Deps: cv2, numpy, ffmpeg subprocess only.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

SEED = 0
W, H, FPS, DUR, NFRAMES = 480, 480, 10, 10, 100
GOLD = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")
SYNTH_DIR = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/synth")
MANIFEST_OUT = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_synth.json")
READY = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/READY_synth")
SCRATCH = Path(__file__).resolve().parents[1] / ".scratch" / "synth"

DEGRADATIONS = ["blur", "darken", "overexpose", "frozen", "black",
                "shake", "noise", "lowbitrate", "static"]

PARAMS = {
    "clean": {},
    "blur": {"filter": "gblur=sigma=8"},
    "darken": {"filter": "eq=brightness=-0.7"},
    "overexpose": {"filter": "eq=brightness=0.6:contrast=1.1"},
    "frozen": {"method": "first-frame-loop"},
    "black": {"filter": "geq=r=0:g=0:b=0"},
    "shake": {"dx_dy_range_px": [-14, 14], "borderMode": "REFLECT"},
    "noise": {"sigma": 35, "distribution": "gaussian"},
    "lowbitrate": {"b:v": "70k", "maxrate": "70k", "bufsize": "140k"},
    "static": {"distribution": "uniform-RGB"},
}


def item_seed(synth_id: str) -> int:
    return int(hashlib.sha256(synth_id.encode()).hexdigest()[:8], 16)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"CMD FAILED: {' '.join(cmd)}\nSTDERR:\n{r.stderr[-3000:]}")
    return r


def probe(path: Path):
    r = run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_read_frames,codec_name",
             "-of", "json", str(path)])
    s = json.loads(r.stdout)["streams"][0]
    return s


def verify(path: Path):
    s = probe(path)
    n = int(s.get("nb_read_frames", -1))
    w, h = int(s["width"]), int(s["height"])
    dur = float(s.get("duration", -1))
    fps_txt = s.get("avg_frame_rate", "?")
    ok = (n == NFRAMES and w == W and h == H and abs(dur - DUR) < 0.6)
    return ok, f"frames={n} {w}x{h} dur={dur:.2f} fps={fps_txt} codec={s.get('codec_name')}"


def pick_sources(items):
    good = sorted([it for it in items if it["label"] == 1], key=lambda d: d["id"])
    # greedy distinct-clip cover: round-robin 2 per clip
    clips = []
    for it in good:
        if it["clip"] not in clips:
            clips.append(it["clip"])
    chosen = []
    for rnd in range(3):  # up to 3 per clip
        for c in clips:
            cand = [it for it in good if it["clip"] == c and it not in chosen]
            if cand and len([x for x in chosen if x["clip"] == c]) < (2 if len(clips) >= 3 else 3):
                chosen.append(cand[0])
            if len(chosen) >= 6:
                break
        if len(chosen) >= 6:
            break
    return chosen[:6]


def extract_clean(src: str, start: float, out: Path):
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(DUR), "-i", src,
         "-vf", f"scale={W}:{H},fps={FPS}", "-an", "-r", str(FPS),
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         str(out)])


def ffmpeg_filter(clean: Path, out: Path, vf_extra: str, extra_args=()):
    vf = f"scale={W}:{H},fps={FPS}" + (f",{vf_extra}" if vf_extra else "")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(clean),
         "-vf", vf, "-an", "-r", str(FPS),
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         *extra_args, str(out)])


def make_frozen(clean: Path, out: Path):
    f0 = SCRATCH / (out.stem + "_f0.png")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(clean), "-vf", "select=eq(n\\,0)",
         "-v:frames", "1", str(f0)])
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", str(FPS), "-t", str(DUR),
         "-i", str(f0), "-vf", f"scale={W}:{H},fps={FPS}", "-an", "-r", str(FPS),
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         str(out)])
    f0.unlink(missing_ok=True)


def read_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def pipe_to_h264(frames_bgr, out: Path):
    assert len(frames_bgr) == NFRAMES, f"expected {NFRAMES} frames, got {len(frames_bgr)}"
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
           "-an", "-r", str(FPS), "-c:v", "libx264", "-crf", "18",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames_bgr:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    err = p.stderr.read().decode()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg pipe failed for {out}:\n{err[-3000:]}")


def make_shake(clean: Path, out: Path, seed: int):
    rng = np.random.default_rng(seed)
    frames = read_frames(clean)
    assert len(frames) == NFRAMES, f"clean has {len(frames)} frames"
    out_frames = []
    for f in frames:
        dx = int(rng.integers(-14, 15))
        dy = int(rng.integers(-14, 15))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out_frames.append(cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT))
    pipe_to_h264(out_frames, out)


def make_noise(clean: Path, out: Path, seed: int):
    rng = np.random.default_rng(seed)
    frames = read_frames(clean)
    assert len(frames) == NFRAMES, f"clean has {len(frames)} frames"
    out_frames = []
    for f in frames:
        n = rng.normal(0, 35, f.shape).astype(np.float32)
        out_frames.append(np.clip(f.astype(np.float32) + n, 0, 255).astype(np.uint8))
    pipe_to_h264(out_frames, out)


def make_static(out: Path, seed: int):
    rng = np.random.default_rng(seed)
    frames = [(rng.random((H, W, 3)) * 256).astype(np.uint8) for _ in range(NFRAMES)]
    pipe_to_h264(frames, out)


def make_lowbitrate(clean: Path, out: Path):
    run(["ffmpeg", "-y", "-v", "error", "-i", str(clean),
         "-vf", f"scale={W}:{H},fps={FPS}", "-an", "-r", str(FPS),
         "-c:v", "libx264", "-b:v", "70k", "-maxrate", "70k", "-bufsize", "140k",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)])


def main():
    t0 = time.time()
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    gold = json.loads(GOLD.read_text())
    sources = pick_sources(gold["items"])
    print(f"sources picked: {[(s['id'], s['clip']) for s in sources]}", flush=True)

    manifest_items = []
    failures = []
    for src in sources:
        sid = src["id"]
        clean_path = SYNTH_DIR / f"syn_{sid}_clean.mp4"
        try:
            extract_clean(src["path"], src["start_s"], clean_path)
            ok, info = verify(clean_path)
            print(f"[{clean_path.name}] {info} {'OK' if ok else 'FAIL'}", flush=True)
            if not ok:
                raise RuntimeError(f"verify failed: {info}")
        except Exception as e:
            print(f"FAIL clean {sid}: {e}", flush=True)
            failures.append(f"syn_{sid}_clean: {e}")
            continue
        manifest_items.append({"id": f"syn_{sid}_clean", "source_id": sid,
                               "degradation": "clean", "params": dict(PARAMS["clean"]),
                               "path": str(clean_path), "label": 1, "seed": item_seed(f"syn_{sid}_clean")})
        for deg in DEGRADATIONS:
            oid = f"syn_{sid}_{deg}"
            out = SYNTH_DIR / f"{oid}.mp4"
            seed = item_seed(oid)
            try:
                if deg == "blur":
                    ffmpeg_filter(clean_path, out, "gblur=sigma=8")
                elif deg == "darken":
                    ffmpeg_filter(clean_path, out, "eq=brightness=-0.7")
                elif deg == "overexpose":
                    ffmpeg_filter(clean_path, out, "eq=brightness=0.6:contrast=1.1")
                elif deg == "frozen":
                    make_frozen(clean_path, out)
                elif deg == "black":
                    ffmpeg_filter(clean_path, out, "geq=r=0:g=0:b=0")
                elif deg == "shake":
                    make_shake(clean_path, out, seed)
                elif deg == "noise":
                    make_noise(clean_path, out, seed)
                elif deg == "lowbitrate":
                    make_lowbitrate(clean_path, out)
                elif deg == "static":
                    make_static(out, seed)
                ok, info = verify(out)
                print(f"[{out.name}] {info} {'OK' if ok else 'FAIL'}", flush=True)
                if not ok:
                    raise RuntimeError(f"verify failed: {info}")
                params = dict(PARAMS[deg])
                params["seed"] = seed
                manifest_items.append({"id": oid, "source_id": sid, "degradation": deg,
                                       "params": params, "path": str(out), "label": 0, "seed": seed})
            except Exception as e:
                print(f"FAIL {oid}: {e}", flush=True)
                failures.append(f"{oid}: {e}")
                if out.exists():
                    out.unlink()

    # keep only verified files in manifest (already the case); write manifest
    wall = time.time() - t0
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": SEED,
                "sources": [s["id"] for s in sources],
                "source_detail": {s["id"]: {"clip": s["clip"], "path": s["path"],
                                            "start_s": s["start_s"], "end_s": s["end_s"]} for s in sources},
                "items": manifest_items,
                "failures": failures,
                "wall_s": wall}
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=1))

    # summary table per degradation
    from collections import defaultdict
    groups = defaultdict(list)
    for it in manifest_items:
        groups[it["degradation"]].append(os.path.getsize(it["path"]))
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'degradation':<12}{'n':>4}{'mean_bytes':>12}", flush=True)
    for deg in ["clean"] + DEGRADATIONS:
        sizes = groups.get(deg, [])
        mean = int(sum(sizes) / len(sizes)) if sizes else 0
        print(f"{deg:<12}{len(sizes):>4}{mean:>12}", flush=True)
    n_clean = sum(1 for it in manifest_items if it["label"] == 1)
    n_deg = sum(1 for it in manifest_items if it["label"] == 0)
    print(f"total: {len(manifest_items)} files ({n_clean} clean/label1, {n_deg} degraded/label0)", flush=True)
    print(f"failures: {failures if failures else 'none'}", flush=True)
    print(f"wall time: {wall:.1f}s", flush=True)

    if failures:
        print("FAILURES PRESENT — not touching READY marker", flush=True)
        sys.exit(1)
    READY.write_text("")
    print(f"wrote {MANIFEST_OUT} + touched {READY}", flush=True)


if __name__ == "__main__":
    main()
