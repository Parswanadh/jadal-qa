"""PoC runner: Egocentric-10K clips -> poc_data/windows.jsonl + thumbs + summary.json.

Verdicts are DESCRIPTIVE rates on unlabelled video, NOT accuracy.

Usage:
  OMP_NUM_THREADS=2 python scripts/poc_run.py --clips a.mp4 [...] [--decode cpu|nvdec] [--shard 0/3]
  OMP_NUM_THREADS=2 python scripts/poc_run.py --all [--decode cpu] [--shard 0/3]
  OMP_NUM_THREADS=2 python scripts/poc_run.py --verify WINDOW_ID
  OMP_NUM_THREADS=2 python scripts/poc_run.py --selfcheck 20
  OMP_NUM_THREADS=2 python scripts/poc_run.py --summary   # (re)write summary.json

Resumable: existing window_ids in windows.jsonl are skipped. One line flushed
per window. Chunk with --shard i/n (disjoint clip split) for <=8min GPU-lock
chunks; CPU runs need no lock. <=3 processes, OMP_NUM_THREADS=2.

Performance design (thrashing-box safe): each clip is decoded TWICE,
sequentially, in O(1) memory — pass A at 160x90 for features+S0, pass B at
480x270 keeping 3 thumb frames per 10-s window. No per-window seeks.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eval.gpu.decode import clip_duration  # noqa: E402
from jadal.poc import cascade as C  # noqa: E402
from jadal.poc import receipts as R  # noqa: E402
sys.path.insert(0, HERE)
import poc_paths  # noqa: E402

# Paths resolve in main() via scripts/poc_paths.py (env var, then candidates
# relative to ROOT; the first that exists; loud failure otherwise).
ROOT = os.path.normpath(os.path.join(HERE, ".."))
POC_DATA = WINDOWS = SUMMARY = THUMBS = None


def set_data_dir(path):
    global POC_DATA, WINDOWS, SUMMARY, THUMBS
    POC_DATA = path
    WINDOWS = os.path.join(POC_DATA, "windows.jsonl")
    SUMMARY = os.path.join(POC_DATA, "summary.json")
    THUMBS = os.path.join(POC_DATA, "thumbs")


WIN_S = 10.0
FRAMES_PER_WIN = 300  # 10 s @ 30 fps

try:
    import numpy as np
except ImportError:
    np = None


def ffmpeg_version():
    try:
        q = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                           timeout=15)
        return q.stdout.splitlines()[0][:120]
    except Exception:
        return "unknown"


def code_sha():
    try:
        q = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return q.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def torch_version():
    try:
        import torch
        return f"{torch.__version__}/cuda={torch.cuda.is_available()}"
    except Exception:
        return "unknown"


def vram_peak_mib():
    try:
        q = subprocess.run(["nvidia-smi",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        used = int(q.stdout.split(",")[0].strip())
        vram_peak_mib.peak = max(getattr(vram_peak_mib, "peak", 0), used)
        return used
    except Exception:
        return None


def parse_ids(path):
    m = re.search(r"(factory[_]?(\d+)).*?(worker[_]?(\d+)).*?_(\d+)\.mp4$", path)
    if m:
        return f"factory_{int(m.group(2)):03d}", f"worker_{int(m.group(4)):03d}", m.group(5)
    base = os.path.splitext(os.path.basename(path))[0]
    return "factory_?", "worker_?", base


def window_id(factory, worker, clipnum, start_s):
    f = re.sub(r"\D", "", factory)[-3:].zfill(3)
    w = re.sub(r"\D", "", worker)[-3:].zfill(3)
    return f"f{f}_w{w}_c{clipnum}_t{int(start_s):04d}"


def load_done():
    done = set()
    if os.path.exists(WINDOWS):
        with open(WINDOWS) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["window_id"])
                    except Exception:
                        pass
    return done


def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def stream_raw(path, w, h, pix="rgb24", ch=3):
    """Yield (chunk_index, uint8 array) of FRAMES_PER_WIN frames, sequential."""
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-i", path, "-vf", f"scale={w}:{h},format={pix}",
           "-f", "rawvideo", "-pix_fmt", pix, "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, bufsize=1 << 22)
    fs = w * h * ch
    win = FRAMES_PER_WIN * fs
    idx, buf = 0, b""
    try:
        while True:
            need = win - len(buf)
            ch_ = p.stdout.read(max(need, 1 << 20))
            if not ch_:
                break
            buf += ch_
            while len(buf) >= win:
                raw = buf[:win]
                buf = buf[win:]
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    FRAMES_PER_WIN, h, w, ch) if ch == 3 else \
                    np.frombuffer(raw, dtype=np.uint8).reshape(FRAMES_PER_WIN, h, w)
                yield idx, arr
                idx += 1
        if len(buf) >= fs:
            n = len(buf) // fs
            arr = np.frombuffer(buf[:n * fs], dtype=np.uint8)
            arr = arr.reshape(n, h, w, ch) if ch == 3 else arr.reshape(n, h, w)
            yield idx, arr
    finally:
        try:
            p.wait(timeout=30)
        except Exception:
            p.kill()


def s0_from_luma(luma60):
    """S0 probe from the working-res buffer: 2x2 box to 80x45 gray approx.

    Training probe was ffmpeg scale=80:45 gray; the 2x2 box over the 160x90
    luma is the documented equivalent (same bandlimit, no extra decode).
    """
    g = luma60.reshape(60, 45, 2, 80, 2).mean(axis=(2, 4))
    return {"mean": float(g.mean()), "p5": float(np.percentile(g, 5)),
            "p95": float(np.percentile(g, 95)),
            "spatial_var": float(g[:10].mean(axis=0).var()),
            "temp_e": float(abs(np.diff(g, axis=0)).mean())}


def boxcar_to_10fps(chunk_f64):
    """Anti-alias normalisation: 3-tap boxcar then decimate (30fps->10fps).

    Approximates what a native 10fps camera (the head's training condition)
    would see; avoids aliasing energy that bare [::3] would fold into the
    2-8 Hz hand-motion band.
    """
    t = (len(chunk_f64) // 3) * 3
    x = chunk_f64[:t].reshape(-1, 3, *chunk_f64.shape[1:])
    return x.mean(axis=1)


def compute_record(path, factory, worker, clipnum, start_s, arr,
                   clf, scaler, versions):
    """Deterministic per-window math shared by the run and verify paths.

    arr: (n,H,W,3) uint8 at WORK_W x WORK_H, first frame = start_s.
    """
    n = arr.shape[0]
    wid = window_id(factory, worker, clipnum, start_s)
    if n < FRAMES_PER_WIN:
        pad = np.repeat(arr[-1:], FRAMES_PER_WIN - n, axis=0)
        full = np.concatenate([arr, pad], axis=0)
    else:
        full = arr[:FRAMES_PER_WIN]
    f64 = full.astype(np.float64) / 255.0
    lum = luma_of(f64)
    s0 = s0_from_luma(lum[:60])
    sub = boxcar_to_10fps(f64)
    feats = C.cpu_window_features_alias(lum, sub, f64)
    feats_r = {k: round(float(v), 4) for k, v in feats.items()}
    verdict, stage, reasons, score = C.decide_window(s0, feats, clf, scaler)
    fdigest = R.features_digest(feats_r)
    receipt = R.receipt_for(wid, verdict, reasons, fdigest, versions)
    return {"window_id": wid, "factory": factory, "worker": worker,
            "clip": path, "start_s": round(float(start_s), 1),
            "end_s": round(float(start_s + n / 30.0), 1),
            "verdict": verdict, "stage": stage, "reasons": reasons,
            "score": score, "features": feats_r,
            "features_digest": fdigest, "receipt_sha256": receipt,
            "versions": versions, "thumbs": _thumb_names(wid),
            "label_source": "none", "n_frames": n,
            "effective_fps": C.EFFECTIVE_FPS,
            "decode_path": "cpu-sequential"}


def process_clip(path, done, clf, scaler, head_sha, versions, out_fh, stats):
    """Two sequential passes over one clip; append new windows. Returns n_new."""
    factory, worker, clipnum = parse_ids(path)
    n_new = 0
    # ---- pass A: 160x90 features + S0 + verdicts ----
    thumbs_plan = {}  # win_idx -> list of frame offsets (filled here, used in pass B)
    win_idx = 0
    tA = time.perf_counter()
    for idx, arr in stream_raw(path, C.WORK_W, C.WORK_H):
        n = arr.shape[0]
        start_s = idx * WIN_S
        wid = window_id(factory, worker, clipnum, start_s)
        if n < 10:
            win_idx += 1
            continue
        # edge-pad ragged tail to 300 (reported via n_frames)
        if n < FRAMES_PER_WIN:
            pad = np.repeat(arr[-1:], FRAMES_PER_WIN - n, axis=0)
            full = np.concatenate([arr, pad], axis=0)
        else:
            full = arr
        thumbs_plan[win_idx] = sorted({n // 4, n // 2, 3 * n // 4})
        if wid not in done:
            t0 = time.perf_counter()
            rec = compute_record(path, factory, worker, clipnum, start_s,
                                 arr, clf, scaler, versions)
            feat_ms = (time.perf_counter() - t0) * 1000
            rec["latency_ms"] = {"decode": None, "features": round(feat_ms, 1),
                                 "thumbs": None, "total": round(feat_ms, 1)}
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            done.add(wid)
            n_new += 1
            stats["windows"] += 1
            stats["feat_ms"] += feat_ms
        win_idx += 1
    stats["decodeA_s"] += time.perf_counter() - tA
    if not thumbs_plan:
        return n_new
    # ---- pass B: 480x270 sequential, keep 3 frames per window ----
    import cv2
    tB = time.perf_counter()
    os.makedirs(THUMBS, exist_ok=True)
    need = {}
    for wi, offs in thumbs_plan.items():
        base = wi * FRAMES_PER_WIN
        for k, off in zip("abc", offs):
            need[base + off] = (wi, k)
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-i", path, "-vf", "scale=480:-1,format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, bufsize=1 << 22)
    # probe output height: 270 for 16:9
    W2, H2 = 480, 270
    fs2 = W2 * H2 * 3
    fno, kept = 0, 0
    try:
        while True:
            raw = p.stdout.read(fs2)
            if len(raw) < fs2:
                break
            if fno in need:
                wi, k = need[fno]
                wid = window_id(factory, worker, clipnum, wi * WIN_S)
                fr = np.frombuffer(raw, dtype=np.uint8).reshape(H2, W2, 3)
                bgr = fr[:, :, ::-1]
                cv2.imwrite(os.path.join(THUMBS, f"{wid}_{k}.jpg"), bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                kept += 1
            fno += 1
    finally:
        try:
            p.wait(timeout=30)
        except Exception:
            p.kill()
    stats["decodeB_s"] += time.perf_counter() - tB
    stats["thumbs"] += kept
    return n_new


def _thumb_names(wid):
    return [f"thumbs/{wid}_{k}.jpg" for k in "abc"]


def list_clips():
    clip_root = poc_paths.resolve_clip_root(ROOT)
    cands = sorted(glob.glob(os.path.join(clip_root, "factory_*", "worker_*", "*.mp4")))
    good = []
    for p in cands:
        try:
            if clip_duration(p) > 30:
                good.append(p)
        except Exception:
            continue
    return good


def write_summary(timing_extra=None):
    recs = []
    if os.path.exists(WINDOWS):
        with open(WINDOWS) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    by_verdict, by_reason, by_factory, by_worker = {}, {}, {}, {}
    stages, dec_paths = {}, {}
    lat = {"features": [], "total": []}
    for r in recs:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
        stages[r["stage"]] = stages.get(r["stage"], 0) + 1
        dec_paths[r.get("decode_path", "?")] = dec_paths.get(r.get("decode_path", "?"), 0) + 1
        for reason in r.get("reasons", []):
            by_reason[reason["code"]] = by_reason.get(reason["code"], 0) + 1
        f = by_factory.setdefault(r["factory"], {"n": 0, "accept": 0, "reject": 0, "review": 0})
        f["n"] += 1
        f[r["verdict"]] += 1
        w = by_worker.setdefault(f"{r['factory']}/{r['worker']}",
                                 {"n": 0, "accept": 0, "reject": 0, "review": 0})
        w["n"] += 1
        w[r["verdict"]] += 1
        for k in lat:
            if r["latency_ms"].get(k) is not None:
                lat[k].append(r["latency_ms"][k])
    hours = round(sum(r["end_s"] - r["start_s"] for r in recs) / 3600, 3)
    wall_h = hpwh = None
    if timing_extra and timing_extra.get("wall_s"):
        wall_h = round(timing_extra["wall_s"] / 3600, 4)
        hpwh = round(hours / max(wall_h, 1e-9), 2)
    n = len(recs)
    summ = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label_note": "descriptive rates on unlabelled video, NOT accuracy",
            "n_windows": n, "hours": hours,
            "by_verdict": by_verdict, "by_reason": by_reason,
            "by_factory": by_factory, "by_worker": by_worker,
            "by_stage": stages,
            "s0_exit_frac": round(stages.get("S0", 0) / max(n, 1), 3),
            "s1_decided_frac": round(stages.get("S1", 0) / max(n, 1), 3),
            "s2_review_frac": round(stages.get("S2", 0) / max(n, 1), 3),
            "decode_path": dec_paths,
            "vram_peak_mib": getattr(vram_peak_mib, "peak", None),
            "latency_ms_p50": {k: round(float(np.median(v)), 1) if v else None
                               for k, v in lat.items()},
            "throughput": {"hours_of_video_per_wall_hour": hpwh,
                           "wall_hours": wall_h,
                           "rtf": (round(hours * 3600 / timing_extra["wall_s"], 1)
                                   if timing_extra and timing_extra.get("wall_s") and hours else None),
                           "gpu": "RTX 4070 Laptop (NOT datacenter, NOT a Pi)"},
            "partial": True}
    if timing_extra:
        summ["run_stats"] = timing_extra
    json.dump(summ, open(SUMMARY, "w"), indent=1)
    return summ


def recompute_window(rec, clf, scaler):
    """Re-verify: re-decode the clip SEQUENTIALLY (same as production) to the
    stored window's chunk, recompute verdict+digests with stored versions."""
    target = int(round(rec["start_s"] / WIN_S))
    factory, worker, clipnum = parse_ids(rec["clip"])
    for idx, arr in stream_raw(rec["clip"], C.WORK_W, C.WORK_H):
        if idx == target:
            fresh = compute_record(rec["clip"], factory, worker, clipnum,
                                   rec["start_s"], arr, clf, scaler,
                                   rec["versions"])
            return (fresh["verdict"], fresh["stage"], fresh["reasons"],
                    fresh["score"], fresh["features"],
                    fresh["features_digest"], fresh["receipt_sha256"])
    raise RuntimeError(f"chunk {target} not decoded from {rec['clip']}")


def load_recs():
    d = {}
    if os.path.exists(WINDOWS):
        with open(WINDOWS) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    d[r["window_id"]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--decode", default="cpu", choices=["cpu", "nvdec"])
    ap.add_argument("--shard", default=None, help="i/n disjoint clip split")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--verify", default=None)
    ap.add_argument("--selfcheck", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--rebuild-thumbs", action="store_true",
                    help="re-extract thumbs for stored windows missing jpgs")
    args = ap.parse_args()
    try:
        set_data_dir(poc_paths.resolve_poc_data(ROOT))
    except poc_paths.PathError as e:
        print(f"poc_run: {e}", file=sys.stderr)
        sys.exit(2)

    recs = load_recs()

    if args.summary:
        s = write_summary()
        print(json.dumps({k: s[k] for k in ("n_windows", "hours", "by_verdict")}, indent=1))
        return

    clf, scaler, head_sha, head_doc = C.train_head()
    print(f"head: n_train={head_doc['n_train']} thr={head_doc['threshold']} "
          f"sha={head_sha[:16]}", flush=True)

    if args.selfcheck:
        ids = sorted(recs)[:args.selfcheck]
        assert ids, "no windows to check"
        bad = 0
        for wid in ids:
            old = recs[wid]
            v, s, rea, sco, fr, fd, rc = recompute_window(old, clf, scaler)
            ok = (old["receipt_sha256"] == rc and old["features_digest"] == fd)
            print(f"{wid}: {'MATCH' if ok else 'MISMATCH'} {rc[:16]}", flush=True)
            bad += not ok
        print(f"selfcheck: {len(ids) - bad}/{len(ids)} identical digests on re-run")
        sys.exit(1 if bad else 0)

    if args.verify:
        if args.verify not in recs:
            print(f"{args.verify} not in {WINDOWS}")
            sys.exit(2)
        old = recs[args.verify]
        v, s, rea, sco, fr, fd, rc = recompute_window(old, clf, scaler)
        print(json.dumps({"window_id": args.verify, "stored": old["receipt_sha256"],
                          "recomputed": rc, "match": old["receipt_sha256"] == rc},
                         indent=1))
        sys.exit(0 if old["receipt_sha256"] == rc else 1)

    if args.rebuild_thumbs:
        missing = [r for r in recs.values()
                   if not all(os.path.exists(os.path.join(POC_DATA, t))
                              for t in r.get("thumbs", []))]
        print(f"windows missing thumbs: {len(missing)}", flush=True)
        by_clip = {}
        for r in missing:
            by_clip.setdefault(r["clip"], []).append(r)
        import cv2
        os.makedirs(THUMBS, exist_ok=True)
        for path, rs in by_clip.items():
            want = {}
            for r in rs:
                base = int(round(r["start_s"] / WIN_S)) * FRAMES_PER_WIN
                n = r.get("n_frames", FRAMES_PER_WIN)
                for k, off in zip("abc", sorted({n // 4, n // 2, 3 * n // 4})):
                    want[base + off] = (r["window_id"], k)
            cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
                   "-i", path, "-vf", "scale=480:-1,format=rgb24",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
            W2, H2, fs2, fno = 480, 270, 480 * 270 * 3, 0
            while True:
                raw = p.stdout.read(fs2)
                if len(raw) < fs2:
                    break
                if fno in want:
                    wid, k = want[fno]
                    fr = np.frombuffer(raw, dtype=np.uint8).reshape(H2, W2, 3)
                    cv2.imwrite(os.path.join(THUMBS, f"{wid}_{k}.jpg"),
                                fr[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
                fno += 1
            p.wait()
            print(f"  {os.path.basename(path)}: frames_seen={fno}", flush=True)
        return

    if args.decode != "cpu":
        print("note: --decode nvdec selected per-window; sequential passes use cpu; "
              "recording choice in versions", flush=True)

    if args.clips:
        clips = args.clips
    elif args.all:
        try:
            clips = list_clips()
        except poc_paths.PathError as e:
            print(f"poc_run: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        ap.error("need --clips, --all, --verify, --selfcheck, --summary or --rebuild-thumbs")
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        clips = [c for k, c in enumerate(sorted(clips)) if k % n == i]
    if args.max_clips:
        clips = clips[:args.max_clips]
    print(f"clips: {len(clips)}", flush=True)

    os.makedirs(THUMBS, exist_ok=True)
    done = set(recs)
    code, torch_v, ffm = code_sha(), torch_version(), ffmpeg_version()
    versions = {"code_sha": code, "device": "RTX 4070 Laptop",
                "torch": torch_v, "ffmpeg": ffm,
                # false when any metadata lookup failed ("unknown" above)
                "env_ok": all(v != "unknown" for v in (code, torch_v, ffm)),
                "head_sha256": head_sha, "threshold": C.THRESHOLD,
                "margin": C.MARGIN, "work_res": [C.WORK_W, C.WORK_H],
                "effective_fps": C.EFFECTIVE_FPS, "temporal": "3-tap boxcar + decimate",
                "s0_probe": "2x2 box of 160x90 luma (approx of 80x45 gray probe)",
                "train": "IndEgo 480p/10fps full-fit (descriptive on 1080p/30fps input)"}
    stats = {"windows": 0, "feat_ms": 0.0, "decodeA_s": 0.0, "decodeB_s": 0.0,
             "thumbs": 0}
    t0 = time.perf_counter()
    n_new = 0
    vram_peak_mib()
    with open(WINDOWS, "a") as fh:
        for ci, path in enumerate(clips):
            try:
                c0 = time.perf_counter()
                n_new += process_clip(path, done, clf, scaler, head_sha,
                                      versions, fh, stats)
                vram_peak_mib()
                print(f"[{ci + 1}/{len(clips)}] {os.path.basename(path)} "
                      f"{time.perf_counter() - c0:.0f}s new_total={n_new}", flush=True)
            except Exception as e:
                print(f"CLIP_ERR {path}: {str(e)[:200]}", flush=True)
    wall = time.perf_counter() - t0
    s = write_summary({"wall_s": round(wall, 1), **{k: round(v, 1) for k, v in stats.items()
                                                    if isinstance(v, float)}})
    print(json.dumps({"new_windows": n_new, "wall_s": round(wall, 1),
                      "by_verdict": s["by_verdict"],
                      "n_windows": s["n_windows"]}, indent=1))


if __name__ == "__main__":
    main()
