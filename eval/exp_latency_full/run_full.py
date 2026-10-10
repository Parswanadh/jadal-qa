#!/usr/bin/env python3
"""L2.01 — full-corpus PoC pipeline run with atomic per-stage perf_counter.

Runs the REPO's E8.01 pipeline (``eval/exp_rescore/run_e8.py``: one NVDEC +
``scale_cuda`` three-branch pass per window -> pixel-26, EG2 (2 fps),
E4.01 regime descriptors -> profile router; ``jadal/ensemble/scorer.py`` frozen
two-view logistic ensemble -> verdict + content-hashed receipt) over ALL 1,546
Egocentric-10K scale windows, with ``time.perf_counter`` around every stage.

Stages (one JSONL row per window):
  1. io_probe       open+seek+read 64 KiB at the window byte offset (cold
                    ``ffprobe`` container open is timed separately, once/clip)
  2. decode         repo path = NVDEC(``-hwaccel cuda``) + ``scale_cuda`` three
                    branches in ONE ffmpeg pass (the E8.01 decode path)
                    + a decode COMPARISON (CPU ffmpeg/swscale vs explicit
                    ``-c:v hevc_cuvid``) on the first window of every clip
  3. pixel_features repo ``jadal.gpu.features.cpu_window_features`` (pixel-26)
  4. descriptors    repo ``eval/exp_regimes/descriptors`` on the native frames
  5. eg2_embed      repo EG2 mean embedding, 2 fps (EmbeddingGemma-2)
  6. mosaic         repo ``jadal.mosaic.build`` (only if the routed profile
                    enables ``evidence.mosaic``)
  7. decision       repo ensemble + router + verdict + receipt
  8. write          JSONL append (timed by the parent)

Verdicts/score/features_digest are identical to E8.01 (same code path); only
``latency_ms`` carries more stages, so ``receipt_sha256`` differs (documented).

Run (GPU slot held for the whole job):
  source /workspace/env.sh
  export OMP_NUM_THREADS=1
  bash tools/pod/gpu_run.sh python3 eval/exp_latency_full/run_full.py \
        --workers 4 --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_regimes"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))

import run_e8 as e8  # noqa: E402  (the repo E8.01 pipeline)

from contract import features_block, reasons_for, route_from_decision, verdict_for  # noqa: E402
from jadal.gpu.features import cpu_window_features  # noqa: E402
from jadal.regimes.router import choose_profile  # noqa: E402
from jadal.ensemble.scorer import ensemble_prob  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "queues", "manifest_scale.json")
OUTDIR = os.path.join(REPO, "eval", "results", "exp_latency_full")
W, H = 160, 90
EGW, EGH = 512, 288
NATIVE_W, NATIVE_H = 1920, 1080
WINDOW_S = 10.0
DEVICE = "A40"
STAGE = "L1"
MEGA = 1 << 20

# the pod EG2 weights (E8.01 ran with the remapped pod path; a post-merge commit
# reverted configs/ensemble_v1.json to the laptop path -> patch it in memory).
POD_EG2 = ("/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2"
           "/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")

smi_log = []
smi_stop = threading.Event()


def smi_sampler():
    while not smi_stop.is_set():
        try:
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            u, m = [x.strip() for x in p.stdout.strip().split(",")]
            smi_log.append({"t": time.time(), "gpu_pct": float(u),
                            "vram_mib": float(m)})
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)


# ---------------------------------------------------------------- worker setup
def init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    e8.init_worker()  # sets e8._CFG / e8._CFG_FRAG (head numbers checked)
    if os.path.isdir(POD_EG2):
        e8._CFG["views"]["eg2"]["embed_model_dir"] = POD_EG2


def ms(t0):
    return (time.perf_counter() - t0) * 1000.0


def io_probe(path, start_s, dur_s):
    """open + seek to the window byte offset + read 64 KiB (real disk IO)."""
    t = time.perf_counter()
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        frac = min(max(float(start_s) / max(float(dur_s), 1e-9), 0.0), 1.0)
        off = min(int(frac * size), max(size - 65536, 0))
        f.seek(off)
        f.read(65536)
    return ms(t)


def _decode_nvdec_cuvid(path, start_s, dur_s, w=W, h=H, timeout_s=180):
    """Explicit NVDEC ``-c:v hevc_cuvid`` + ``scale_cuda`` (comparison arm)."""
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-c:v", "hevc_cuvid",
           "-ss", f"{float(start_s):.3f}", "-t", f"{float(dur_s):.3f}", "-i", path,
           "-vf", f"scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12,format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=timeout_s)
    if p.returncode != 0:
        raise RuntimeError(f"cuvid rc={p.returncode}: "
                           f"{p.stderr[-200:].decode('utf-8', 'replace')}")
    return p.stdout


def decode_compare(path, start_s, dur_s):
    """CPU ffmpeg/swscale vs explicit NVDEC hevc_cuvid, same window."""
    out = {}
    from eval.gpu.decode import decode_window
    t = time.perf_counter()
    try:
        rgb, _ = decode_window(path, start_s, dur_s, W, H, "rgb24")
        out["cpu_ms"] = ms(t)
        out["cpu_frames"] = int(rgb.shape[0])
    except Exception as ex:  # noqa: BLE001
        out["cpu_ms"] = ms(t)
        out["cpu_error"] = str(ex)[:160]
    t = time.perf_counter()
    try:
        buf = _decode_nvdec_cuvid(path, start_s, dur_s)
        out["nvdec_cuvid_ms"] = ms(t)
        out["nvdec_cuvid_frames"] = int(len(buf) // (W * H * 3))
    except Exception as ex:  # noqa: BLE001
        out["nvdec_cuvid_ms"] = ms(t)
        out["nvdec_cuvid_error"] = str(ex)[:160]
    return out


# ----------------------------------------------------------------- per window
def score_window_timed(item, profile_map, do_compare=False):
    from PIL import Image  # noqa: F401  (imported for parity with run_e8)
    lat = {}
    wall = {}
    t_all = time.perf_counter()
    path, s0 = item["path"], float(item["start_s"])
    s1 = float(item["end_s"])
    dur = s1 - s0

    # 1. io_probe ----------------------------------------------------------
    t = time.perf_counter()
    w0 = time.time()
    lat["io_probe"] = io_probe(path, s0, item.get("duration_s", dur))
    wall["io_probe"] = [w0, time.time()]

    # 2. decode (repo path: three-branch NVDEC + scale_cuda) ---------------
    t = time.perf_counter()
    w0 = time.time()
    pix_raw, eg_raw, nat_raw = e8.decode_three_branch(path, s0, dur)
    lat["decode"] = ms(t)
    wall["decode"] = [w0, time.time()]

    # 3. pixel-26 features -------------------------------------------------
    t = time.perf_counter()
    w0 = time.time()
    pix = e8.frames_from_raw(pix_raw, H, W).astype(np.float64) / 255.0
    if pix.shape[0] == 0:
        raise RuntimeError("no pixel frames")
    luma = 0.299 * pix[..., 0] + 0.587 * pix[..., 1] + 0.114 * pix[..., 2]
    fps_pix = min(pix.shape[0] / max(dur, 1e-6), 30.0)
    feats = cpu_window_features(luma, pix, fps=fps_pix)
    lat["pixel_features"] = ms(t)
    wall["pixel_features"] = [w0, time.time()]

    # 4. descriptors (E4.01) on the native frames --------------------------
    t = time.perf_counter()
    w0 = time.time()
    meta = e8._get_meta(path)
    nat_frames = list(e8.frames_from_raw(nat_raw, NATIVE_H, NATIVE_W))
    desc = e8.describe_from_native_frames(meta, nat_frames, s0, s1)
    lat["descriptors"] = ms(t)
    wall["descriptors"] = [w0, time.time()]

    # route (deterministic; needed to know if the profile enables a mosaic)
    decision = choose_profile(desc)
    route = route_from_decision(decision, desc)
    prof = profile_map.get(route["profile"], profile_map["default"])
    accept_thr = float(prof["thresholds"]["accept"])
    reject_thr = float(prof["thresholds"]["reject"])
    mosaic_enabled = bool(prof.get("evidence", {}).get("mosaic", False))

    # 5. EG2 window embedding (2 fps) --------------------------------------
    t = time.perf_counter()
    w0 = time.time()
    eg2_mean, n_eg = e8.encode_eg2(e8.frames_from_raw(eg_raw, EGH, EGW), e8._CFG)
    lat["eg2_embed"] = ms(t)
    wall["eg2_embed"] = [w0, time.time()]
    try:
        import torch
        vram_peak = float(torch.cuda.max_memory_allocated() / MEGA)
    except Exception:  # noqa: BLE001
        vram_peak = None

    # 6. mosaic (only if the profile enables it) ---------------------------
    mosaic_info = {"enabled": mosaic_enabled}
    if mosaic_enabled:
        from jadal.mosaic import build as MB
        t = time.perf_counter()
        w0 = time.time()
        try:
            n_tiles = len(nat_frames)
            grid = MB.grid_for_n(n_tiles)
            times = MB.tile_times(n_tiles, WINDOW_S)
            fps_native = float(meta.get("fps", 30.0)) or 30.0
            idx = MB.frame_indices(times, fps_native, n_tiles)
            img, side = MB.build_mosaic(
                [f for f in nat_frames], times, idx, grid, 224,
                window_id=item["window_id"], source=path,
                fps_used=fps_native)
            png = MB.png_bytes(img)
            mosaic_info.update({
                "n": n_tiles, "grid": list(grid), "tile_px": 224,
                "size": list(img.size),
                "sha256": hashlib.sha256(png).hexdigest(),
                "builder_version": MB.BUILDER_VERSION,
            })
        except Exception as ex:  # noqa: BLE001
            mosaic_info["error"] = f"{type(ex).__name__}: {ex}"[:160]
        lat["mosaic"] = ms(t)
        wall["mosaic"] = [w0, time.time()]
    else:
        lat["mosaic"] = 0.0

    # 7. decision / verdict / receipt --------------------------------------
    t = time.perf_counter()
    w0 = time.time()
    prob, p_e, p_g = ensemble_prob(eg2_mean, feats, e8._CFG)
    verdict = verdict_for(prob, accept_thr, reject_thr)
    reasons = reasons_for(prob, verdict, accept_thr, reject_thr, route, feats)
    features, fdig = features_block(feats, eg2_mean, pix.shape[0], n_eg,
                                    fps_pix, p_e, p_g)
    lat["decision"] = ms(t)
    wall["decision"] = [w0, time.time()]
    lat["total"] = ms(t_all)

    versions = {
        "recipe": "ensemble_v1",
        **e8._CFG_FRAG,
        "fusion": "equal_prob_average",
        "config_sha256": e8.sha256_hex(open(e8.CONFIG, "rb").read()),
        "eg2_model": "google/embeddinggemma-2@914f7f8",
        "profile": route["profile"],
        "profile_version": prof["version"],
        "profile_margin": route["margin"],
        "profile_thresholds": {"accept": accept_thr, "reject": reject_thr},
        "router_reasons": route["profile_reasons"],
        "unseen_regime": route["unseen_regime"],
        "descriptors_sha256": e8.sha256_hex(e8.canonical_bytes(desc)),
        "decode_path": "nvdec(cuda)+scale_cuda(bilinear)->rgb24",
        "sampling": {"window_s": WINDOW_S, "pixel_fps": round(fps_pix, 4),
                     "eg2_step_s": 0.5, "descriptor_frames": 8},
        "deterministic": True,
        "device": DEVICE,
        "git_sha": e8.git_sha(),
    }
    line = e8.build_line(item, verdict, STAGE, reasons, prob, fdig, features,
                         versions, lat)

    cmp = decode_compare(path, s0, dur) if do_compare else None
    row = {
        "window_id": item["window_id"], "clip": item["clip"],
        "factory": item["factory"], "worker": item["worker"],
        "start_s": round(s0, 4), "end_s": round(s1, 4),
        "verdict": verdict, "score": round(float(prob), 4),
        "profile": route["profile"],
        "latency_ms": {k: round(v, 3) for k, v in lat.items()},
        "n_frames": {"pixel": int(pix.shape[0]), "eg2": int(n_eg),
                     "native": len(nat_frames)},
        "gpu": {"eg2_vram_peak_mib": vram_peak},
        "mosaic": mosaic_info,
        "decode_compare": cmp,
        "stage_wall": wall,
        "t_wall0": wall["io_probe"][0], "t_wall1": wall["decision"][1],
        "contract": line,
    }
    return row


def _worker(args):
    item, profile_map, do_compare = args
    try:
        return {"ok": True, "row": score_window_timed(item, profile_map, do_compare)}
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "window_id": item["window_id"], "clip": item["clip"],
                "factory": item["factory"], "error": f"{type(ex).__name__}: {ex}"[:300]}


# --------------------------------------------------------------- policy probe
def measure_early_exit(items, profile_map, n=16):
    """Directly time a 4 s early-exit window (io + decode4 + pixel4 + decision)."""
    out = []
    init_worker()
    for it in items[:n]:
        try:
            path, s0 = it["path"], float(it["start_s"])
            dur = 4.0
            t = time.perf_counter()
            io_probe(path, s0, it.get("duration_s", 10.0))
            pix_raw, eg_raw, nat_raw = e8.decode_three_branch(path, s0, dur)
            dec = ms(t)
            t = time.perf_counter()
            pix = e8.frames_from_raw(pix_raw, H, W).astype(np.float64) / 255.0
            luma = 0.299 * pix[..., 0] + 0.587 * pix[..., 1] + 0.114 * pix[..., 2]
            feats = cpu_window_features(luma, pix, fps=min(pix.shape[0] / dur, 30.0))
            feat = ms(t)
            out.append({"window_id": it["window_id"], "decode4_ms": dec,
                        "pixel4_ms": feat, "early_ms": dec + feat})
        except Exception as ex:  # noqa: BLE001
            out.append({"window_id": it["window_id"], "error": str(ex)[:160]})
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "windows.jsonl"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compare-first-per-clip", action="store_true", default=True)
    ap.add_argument("--policy-n", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    man, items = e8.load_manifest()
    if args.limit:
        items = items[:args.limit]
    pmap = e8.profiles()
    order = {it["window_id"]: i for i, it in enumerate(items)}

    # first window of every clip -> decode comparison (CPU vs NVDEC cuvid)
    seen, compare_ids = set(), set()
    for it in items:
        if it["clip"] not in seen:
            seen.add(it["clip"])
            compare_ids.add(it["window_id"])

    partial = args.out + ".partial"
    done = {}
    if args.resume and os.path.exists(partial):
        for ln in open(partial):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if r.get("row"):
                done[r["row"]["window_id"]] = r["row"]
        print(f"resume: {len(done)} rows already done", flush=True)
        items = [i for i in items if i["window_id"] not in done]

    th = threading.Thread(target=smi_sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    wall0 = time.time()
    rows, errors = list(done.values()), []

    tasks = [(it, pmap, it["window_id"] in compare_ids) for it in items]
    if args.workers <= 1:
        init_worker()
        for k, tk in enumerate(tasks, 1):
            r = _worker(tk)
            _handle(r, rows, errors, partial)
            _log(k, len(tasks), len(rows), len(errors), t0)
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=init_worker) as pool:
            with open(partial, "a") as fh:
                for k, r in enumerate(pool.imap(_worker, tasks, chunksize=1), 1):
                    if r["ok"]:
                        rows.append(r["row"])
                        tw = time.perf_counter()
                        fh.write(json.dumps({"row": r["row"]}, sort_keys=True) + "\n")
                        fh.flush()
                        r["row"]["latency_ms"]["write"] = round(ms(tw), 3)
                    else:
                        errors.append(r)
                    _log(k, len(tasks), len(rows), len(errors), t0)

    # optional: measure the 4 s early-exit arm on a spread of windows
    policy_rows = []
    if args.policy_n > 0:
        step = max(1, len(man["items"]) // args.policy_n)
        sample = man["items"][::step][:args.policy_n]
        print(f"policy probe: {len(sample)} windows (4 s early exit)", flush=True)
        policy_rows = measure_early_exit(sample, pmap, n=args.policy_n)

    wall = time.perf_counter() - t0
    smi_stop.set()
    th.join(timeout=5)

    rows.sort(key=lambda r: order.get(r["window_id"], 1 << 30))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    # per-clip cold probe table
    clip_rows = []
    for c in sorted({i["clip"] for i in man["items"]}):
        it = next(i for i in man["items"] if i["clip"] == c)
        row = {"clip": c, "factory": it["factory"], "path": it["path"],
               "duration_s": it.get("duration_s"),
               "size_bytes": it.get("size_bytes")}
        try:
            from eval.gpu.decode import clip_duration
            t = time.perf_counter()
            row["io_probe_cold_ms"] = ms(t) if False else None
            t = time.perf_counter()
            d = clip_duration(it["path"])
            row["ffprobe_cold_ms"] = ms(t)
            row["clip_dur_s"] = d
        except Exception as ex:  # noqa: BLE001
            row["ffprobe_error"] = str(ex)[:120]
        clip_rows.append(row)
    with open(os.path.join(OUTDIR, "clips.jsonl"), "w") as f:
        for r in clip_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    import resource
    n_windows = len(rows)
    verdicts = Counter(r["verdict"] for r in rows)
    profiles = Counter(r["profile"] for r in rows)
    meta = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(wall, 2),
        "wall_start_epoch": wall0,
        "workers": args.workers,
        "manifest": {"n_windows": man["n_windows"], "n_clips": man["n_clips"],
                     "total_hours": man["total_hours"]},
        "n_requested": len(tasks) + len(done), "n_windows": n_windows,
        "n_errors": len(errors), "errors": errors[:20],
        "verdict_distribution": dict(sorted(verdicts.items())),
        "profile_distribution": dict(sorted(profiles.items())),
        "windows_per_s": round(n_windows / wall, 4) if wall else None,
        "footage_h_per_wall_h": round(man["total_hours"] / (wall / 3600.0), 3)
        if wall else None,
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "device": DEVICE,
        "pod_eg2_dir": POD_EG2,
        "policy_probe": policy_rows,
        "smi_samples": len(smi_log),
        "smi": smi_log,
        "git_sha": e8.git_sha(),
    }
    json.dump(meta, open(os.path.join(OUTDIR, "run_meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("errors", "smi", "policy_probe")}, indent=1))
    print(f"DONE wall={wall:.0f}s windows={n_windows} errors={len(errors)}")


def _handle(r, rows, errors, partial):
    if r["ok"]:
        rows.append(r["row"])
        tw = time.perf_counter()
        with open(partial, "a") as fh:
            fh.write(json.dumps({"row": r["row"]}, sort_keys=True) + "\n")
        r["row"]["latency_ms"]["write"] = round(ms(tw), 3)
    else:
        errors.append(r)


def _log(k, n, nok, nerr, t0):
    if k % 25 == 0 or k == n:
        el = time.perf_counter() - t0
        print(f"  {k}/{n} ok={nok} err={nerr} {el:.0f}s {k/el:.2f} win/s "
              f"eta={(n-k)/max(k/el,1e-9)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
