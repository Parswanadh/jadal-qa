"""Arm A: baseline OpenCV+MediaPipe L0 signals (harness worker).

Reuses the signals worker's code READ-ONLY (sys.path import, no edits).
Outputs under /home/parshu/projects/jadal-qa-wt/spectral_data/arms/A/:
  gold.json      per-gold-segment {score:null, features, latency_s}
  synth.json     per-synth-item ditto
  dups.json      per-pair {similarity (pHash), latency_s}
  chunks.json    per-item prefix features [2s,4s,6s,8s,10s] (streaming sim)
  allframes.json whole-video single-pass 160px cheap signals + RTF
  cost.json      single-core calibration latency + peak RAM + cpu benchmark

Usage: python -m eval.spectral.run_arm_a [--workers 4] [--calibrate-only]
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/home/parshu/projects/jadal-qa-wt/signals")
os.environ.setdefault("JADAL_MODELS_DIR", "/home/parshu/projects/jadal-qa/models")

DATA = Path("/home/parshu/projects/jadal-qa-wt/spectral_data")
OUTDIR = DATA / "arms" / "A"
SCRATCH = HARNESS / ".scratch" / "armA"

NUMERIC_FEATURES = ["blur_var", "brightness_mean", "under_exposed_frac",
                    "over_exposed_frac", "motion_mean", "hands_visible_frac",
                    "hand_count_mode", "faces_found", "faces_blurred"]


def version() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, cwd=str(HARNESS))
    h = r.stdout.strip()
    r2 = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True,
                        cwd="/home/parshu/projects/jadal-qa-wt/signals")
    return f"harness:{h}/signals:{r2.stdout.strip()}"


def cut_segment(src: str, start: float, dur: float, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(start), "-t", str(dur), "-i", src,
         "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(dst)], check=True)


def cut_prefix(src_seg: Path, dur: float, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-t", str(dur), "-i", str(src_seg),
         "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(dst)], check=True)


def _features_of(path: str) -> tuple[dict, float, dict]:
    from jadal import signals as S
    t0 = time.perf_counter()
    sig = S.compute_signals(path)
    dt = time.perf_counter() - t0
    feats = {k: (float(sig[k]) if isinstance(sig[k], (int, float)) else sig[k])
             for k in NUMERIC_FEATURES}
    feats["decodable"] = bool(sig.get("decodable", False))
    extra = {"phash": sig.get("phash", ""), "dup_of": sig.get("dup_of")}
    return feats, dt, extra


def gold_task(it: dict, tmpdir: str):
    seg = Path(tmpdir) / f"{it['id']}.mp4"
    cut_segment(it["path"], it["start_s"], it["end_s"] - it["start_s"], seg)
    feats, dt, extra = _features_of(str(seg))
    prefixes = []
    for u in (2, 4, 6, 8, 10):
        p = Path(tmpdir) / f"{it['id']}_p{u}.mp4"
        cut_prefix(seg, u, p)
        pf, pdt, _ = _features_of(str(p))
        prefixes.append({"until_s": u, "features": pf, "latency_s": round(pdt, 3)})
        p.unlink(missing_ok=True)
    seg.unlink(missing_ok=True)
    return it["id"], feats, dt, extra, prefixes


def synth_task(it: dict, tmpdir: str):
    feats, dt, extra = _features_of(it["path"])
    prefixes = []
    for u in (2, 4, 6, 8, 10):
        p = Path(tmpdir) / f"{it['id']}_p{u}.mp4"
        cut_prefix(Path(it["path"]), u, p)
        pf, pdt, _ = _features_of(str(p))
        prefixes.append({"until_s": u, "features": pf, "latency_s": round(pdt, 3)})
        p.unlink(missing_ok=True)
    return it["id"], feats, dt, extra, prefixes


def _init_worker():
    sys.path.insert(0, "/home/parshu/projects/jadal-qa-wt/signals")
    os.environ.setdefault("JADAL_MODELS_DIR", "/home/parshu/projects/jadal-qa/models")


def allframes_task(args) -> dict:
    """Single-pass O(1)-memory every-frame 160px cheap signals (single core)."""
    import cv2
    import numpy as np
    item_id, src, start, dur = args
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as td:
        seg = Path(td) / "seg.mp4"
        if start is None:
            segp = src
        else:
            cut_segment(src, start, dur, seg)
            segp = str(seg)
        cap = cv2.VideoCapture(segp)
        n = 0
        lap, bri, mot, skin = [], [], [], []
        prev_small = None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            small = cv2.resize(fr, (160, 160), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            lv = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            m = float(gray.mean())
            lap.append(lv / ((m / 128.0 + 1e-3) ** 2))
            bri.append(m)
            if prev_small is not None:
                d = cv2.absdiff(
                    cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), prev_small)
                mot.append(float(d.mean()))
            prev_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 177, 127))
            skin.append(float((mask > 0).mean()))
            n += 1
        cap.release()
    dt = time.perf_counter() - t0
    a = np.array
    feats = {
        "lap_mean": float(np.mean(lap)) if lap else 0.0,
        "lap_std": float(np.std(lap)) if lap else 0.0,
        "bri_mean": float(np.mean(bri)) if bri else 0.0,
        "bri_std": float(np.std(bri)) if bri else 0.0,
        "mot_mean": float(np.mean(mot)) if mot else 0.0,
        "mot_std": float(np.std(mot)) if mot else 0.0,
        "skin_mean": float(np.mean(skin)) if skin else 0.0,
        "skin_std": float(np.std(skin)) if skin else 0.0,
        "n_frames": n,
    }
    return {"id": item_id, "features": feats,
            "latency_s": round(dt, 4), "video_s": round(n / 10.0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--skip-prefixes", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    ver = version()
    print("version:", ver, flush=True)

    from concurrent.futures import ProcessPoolExecutor
    import tempfile

    gold = json.loads((DATA / "manifest_gold.json").read_text())["items"]
    synth = json.loads((DATA / "manifest_synth.json").read_text())["items"]

    if not args.calibrate_only:
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as td:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=_init_worker) as ex:
                g_futs = {ex.submit(gold_task, it, td): it for it in gold}
                s_futs = {ex.submit(synth_task, it, td): it for it in synth}
                gold_items, chunk_items = {}, {}
                for f in g_futs:
                    i, feats, dt, extra, prefs = f.result()
                    gold_items[i] = {"score": None, "features": feats,
                                     "latency_s": round(dt, 3),
                                     "phash": extra["phash"]}
                    if not args.skip_prefixes:
                        chunk_items[i] = prefs
                synth_items = {}
                for f in s_futs:
                    i, feats, dt, extra, prefs = f.result()
                    synth_items[i] = {"score": None, "features": feats,
                                      "latency_s": round(dt, 3),
                                      "phash": extra["phash"]}
                    if not args.skip_prefixes:
                        chunk_items[i] = prefs
        (OUTDIR / "gold.json").write_text(json.dumps(
            {"arm": "A", "version": ver, "items": gold_items}, indent=1))
        (OUTDIR / "synth.json").write_text(json.dumps(
            {"arm": "A", "version": ver, "items": synth_items}, indent=1))
        if chunk_items:
            (OUTDIR / "chunks.json").write_text(json.dumps(
                {"arm": "A", "version": ver,
                 "prefix_until_s": [2, 4, 6, 8, 10],
                 "items": chunk_items}, indent=1))
        print(f"gold={len(gold_items)} synth={len(synth_items)} "
              f"chunks={len(chunk_items)}", flush=True)

        # dups: pHash similarities (cheap, serial)
        from jadal import signals as S
        dups = json.loads((DATA / "manifest_dups.json").read_text())
        ph, ph_lat = {}, {}
        for fid, fpath in dups["files"].items():
            t0 = time.perf_counter()
            ph[fid] = S.phash_segment(fpath)
            ph_lat[fid] = time.perf_counter() - t0

        def ham(a: str, b: str) -> int:
            try:
                return (int(a, 16) ^ int(b, 16)).bit_count()
            except (ValueError, TypeError):
                return 64

        pairs = {}
        for p in dups["pairs"]:
            key = f"{p['a']}|{p['b']}"
            d = ham(ph[p["a"]], ph[p["b"]])
            pairs[key] = {"similarity": round(1.0 - d / 64.0, 4),
                          "hamming": d,
                          "latency_s": round(ph_lat[p["a"]] + ph_lat[p["b"]], 3)}
        (OUTDIR / "dups.json").write_text(json.dumps(
            {"arm": "A", "method": "pHash", "version": ver,
             "pairs": pairs}, indent=1))
        print(f"dups pairs={len(pairs)}", flush=True)

    # single-core calibration: 6 segments, full compute_signals, taskset by caller
    from jadal import signals as S2
    cal = gold[:6]
    lats = []
    with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as td:
        for it in cal:
            seg = Path(td) / "cal.mp4"
            cut_segment(it["path"], it["start_s"],
                        it["end_s"] - it["start_s"], seg)
            t0 = time.perf_counter()
            S2.compute_signals(str(seg))
            lats.append(time.perf_counter() - t0)
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    cost = {"single_core": "caller must taskset -c 0 for validity",
            "n_cal": len(lats),
            "latency_s_per_segment": [round(x, 3) for x in lats],
            "mean_latency_s": round(sum(lats) / len(lats), 3),
            "peak_ram_mb_self": round(peak_ram_mb, 1),
            "backend": S2.backend_info()}
    (OUTDIR / "cost.json").write_text(json.dumps(
        {"arm": "A", "version": ver, "cost": cost}, indent=1))
    print("cost:", json.dumps(cost), flush=True)

    # all-frames variant: run serially here (caller tasksets for RTF honesty)
    af_args = [(it["id"], it["path"], it["start_s"], it["end_s"] - it["start_s"])
               for it in gold]
    af_args += [(it["id"], it["path"], None, None) for it in synth]
    af_items, af_lats = {}, []
    for a in af_args:
        r = allframes_task(a)
        af_items[r["id"]] = {"score": None, "features": r["features"],
                             "latency_s": r["latency_s"], "frames": r["features"].pop("n_frames") if False else r["features"]["n_frames"]}
        af_lats.append((r["latency_s"], r["video_s"]))
    mean_rtf = sum(v / l for l, v in af_lats) / len(af_lats)
    (OUTDIR / "allframes.json").write_text(json.dumps(
        {"arm": "A", "variant": "allframes_160px_singlepass",
         "version": ver, "mean_rtf_single_process": round(mean_rtf, 2),
         "items": af_items}, indent=1))
    print(f"allframes={len(af_items)} mean_rtf={mean_rtf:.1f}", flush=True)


if __name__ == "__main__":
    main()
