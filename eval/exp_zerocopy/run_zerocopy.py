#!/usr/bin/env python3
"""Z1 — mosaic on/off delta + zero-copy GPU path attempt + decode verdict.

Runs the repo E8.01 pipeline (imported from ``eval/exp_rescore/run_e8.py``) on a
deterministic **30-window sample spread across factories**, with
``time.perf_counter`` around every stage, and compares:

  * mosaic stage ON vs OFF (same decode, only the mosaic build differs);
  * the repo hwdownload path vs a GPU-resident attempt
    (torchcodec NVDEC->VRAM -> repo GPU feature kernels -> GPU EG2 forward);
  * NVDEC vs CPU decode at 1080p30 HEVC and 480p H.264, with/without hwdownload.

Nothing is faked: if a step cannot be done with the repo's code it is recorded
as a blocker in the output JSON.

Run (GPU slot held for the whole job):
  source /workspace/env.sh
  export OMP_NUM_THREADS=2 JADAL_GPU_SLOTS=4
  bash tools/pod/gpu_run.sh python3 eval/exp_zerocopy/run_zerocopy.py --sample-n 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_regimes"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))

import run_e8 as e8  # noqa: E402
from contract import features_block, reasons_for, route_from_decision, verdict_for  # noqa: E402
from jadal.gpu.features import cpu_window_features  # noqa: E402
from jadal.regimes.router import choose_profile  # noqa: E402
from jadal.ensemble.scorer import ensemble_prob  # noqa: E402

import zerocopy as zc  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "queues", "manifest_scale.json")
OUTDIR = os.path.join(REPO, "eval", "results", "exp_zerocopy")
W, H = 160, 90
EGW, EGH = 512, 288
NATIVE_W, NATIVE_H = 1920, 1080
WINDOW_S = 10.0
DEVICE = "A40"
MEGA = 1 << 20

# 480p H.264 IndEgo clip for the decode verdict (pod path)
INDEGO_480 = os.path.join(BUNDLE, "data", "indego", "1_Assembly", "mp4_480",
                          "User_15_410_0411_2_480.mp4")

POD_EG2 = ("/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2"
           "/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")


def ms(t0):
    return (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------- sample
def build_sample(n=30):
    """Deterministic n-window sample spread across factories (one window/factory).

    Round-robins over factories sorted by name and picks, inside each factory's
    first clip, the window nearest to t=60 s so the decoder must seek.
    """
    man, items = e8.load_manifest()
    by_factory = {}
    for it in items:
        by_factory.setdefault(it["factory"], []).append(it)
    factories = sorted(by_factory)
    if n >= len(factories):
        chosen_f = factories
    else:
        # evenly spaced over the sorted factory list
        idx = np.linspace(0, len(factories) - 1, n).round().astype(int)
        chosen_f = [factories[i] for i in sorted(set(idx.tolist()))]
    sample = []
    for f in chosen_f:
        fitems = by_factory[f]
        clips = sorted({i["clip"] for i in fitems})
        c0 = clips[0]
        citems = [i for i in fitems if i["clip"] == c0]
        it = min(citems, key=lambda i: abs(float(i["start_s"]) - 60.0))
        sample.append(it)
    sample = sample[:n]
    return man, sample


# ------------------------------------------------------------------- baseline
def decode_pipeline(path, s0, dur):
    return e8.decode_three_branch(path, s0, dur)


def _verdict_block(feats, eg2_mean, pix_n, n_eg, fps_pix, desc, pmap):
    decision = choose_profile(desc)
    route = route_from_decision(decision, desc)
    prof = pmap.get(route["profile"], pmap["default"])
    accept_thr = float(prof["thresholds"]["accept"])
    reject_thr = float(prof["thresholds"]["reject"])
    prob, p_e, p_g = ensemble_prob(eg2_mean, feats, e8._CFG)
    verdict = verdict_for(prob, accept_thr, reject_thr)
    reasons = reasons_for(prob, verdict, accept_thr, reject_thr, route, feats)
    features, fdig = features_block(feats, eg2_mean, pix_n, n_eg, fps_pix, p_e, p_g)
    return {"verdict": verdict, "score": round(float(prob), 4),
            "profile": route["profile"], "p_e": round(float(p_e), 4),
            "p_g": round(float(p_g), 4), "features_digest": fdig,
            "accept_thr": accept_thr, "reject_thr": reject_thr}


def baseline_stages(item, pmap, do_mosaic=True):
    """Repo path with per-stage timers; returns (row, raw buffers, feats, eg2)."""
    from PIL import Image  # noqa: F401
    path, s0 = item["path"], float(item["start_s"])
    s1 = float(item["end_s"])
    dur = s1 - s0
    lat, wall = {}, {}
    t_all = time.perf_counter()

    # io_probe
    t = time.perf_counter(); w0 = time.time()
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        frac = min(max(s0 / max(float(item.get("duration_s", dur)), 1e-9), 0.0), 1.0)
        fh.seek(min(int(frac * size), max(size - 65536, 0))); fh.read(65536)
    lat["io_probe"] = ms(t); wall["io_probe"] = [w0, time.time()]

    # decode (3-branch NVDEC + hwdownload)
    t = time.perf_counter(); w0 = time.time()
    pix_raw, eg_raw, nat_raw = decode_pipeline(path, s0, dur)
    lat["decode"] = ms(t); wall["decode"] = [w0, time.time()]

    # pixel-26 (CPU)
    t = time.perf_counter(); w0 = time.time()
    pix = e8.frames_from_raw(pix_raw, H, W).astype(np.float64) / 255.0
    luma = 0.299 * pix[..., 0] + 0.587 * pix[..., 1] + 0.114 * pix[..., 2]
    fps_pix = min(pix.shape[0] / max(dur, 1e-6), 30.0)
    feats = cpu_window_features(luma, pix, fps=fps_pix)
    lat["pixel_features"] = ms(t); wall["pixel_features"] = [w0, time.time()]

    # descriptors (CPU, native frames)
    t = time.perf_counter(); w0 = time.time()
    meta = e8._get_meta(path)
    nat_frames = list(e8.frames_from_raw(nat_raw, NATIVE_H, NATIVE_W))
    desc = e8.describe_from_native_frames(meta, nat_frames, s0, s1)
    lat["descriptors"] = ms(t); wall["descriptors"] = [w0, time.time()]

    # EG2 (CPU frames -> PIL -> encode on GPU)
    t = time.perf_counter(); w0 = time.time()
    eg_frames = e8.frames_from_raw(eg_raw, EGH, EGW)
    eg2_mean, n_eg = e8.encode_eg2(eg_frames, e8._CFG)
    lat["eg2_embed"] = ms(t); wall["eg2_embed"] = [w0, time.time()]

    # mosaic
    mosaic_info = {"enabled": bool(do_mosaic)}
    if do_mosaic:
        from jadal.mosaic import build as MB
        t = time.perf_counter(); w0 = time.time()
        try:
            n_tiles = len(nat_frames)
            grid = MB.grid_for_n(n_tiles)
            times = MB.tile_times(n_tiles, WINDOW_S)
            fps_native = float(meta.get("fps", 30.0)) or 30.0
            idx = MB.frame_indices(times, fps_native, n_tiles)
            img, side = MB.build_mosaic(
                list(nat_frames), times, idx, grid, 224,
                window_id=item["window_id"], source=path, fps_used=fps_native)
            png = MB.png_bytes(img)
            mosaic_info.update({"n": n_tiles, "grid": list(grid),
                                "sha256": hashlib.sha256(png).hexdigest(),
                                "builder_version": MB.BUILDER_VERSION})
        except Exception as ex:  # noqa: BLE001
            mosaic_info["error"] = f"{type(ex).__name__}: {ex}"[:160]
        lat["mosaic"] = ms(t); wall["mosaic"] = [w0, time.time()]
    else:
        lat["mosaic"] = 0.0

    # decision
    t = time.perf_counter(); w0 = time.time()
    vb = _verdict_block(feats, eg2_mean, pix.shape[0], n_eg, fps_pix, desc, pmap)
    lat["decision"] = ms(t); wall["decision"] = [w0, time.time()]
    lat["total"] = ms(t_all)

    row = {
        "window_id": item["window_id"], "clip": item["clip"],
        "factory": item["factory"], "start_s": round(s0, 4),
        "latency_ms": {k: round(v, 3) for k, v in lat.items()},
        "n_frames": {"pixel": int(pix.shape[0]), "eg2": int(n_eg),
                     "native": len(nat_frames)},
        "mosaic": mosaic_info,
        "bytes": {
            "pix": int(pix.shape[0] * W * H * 3),
            "eg2": int(len(eg_frames) * EGW * EGH * 3),
            "native": int(len(nat_frames) * NATIVE_W * NATIVE_H * 3),
        },
        "stage_wall": wall,
        **vb,
    }
    raw = {"pix_raw": pix_raw, "eg_raw": eg_raw, "nat_raw": nat_raw,
           "pix": pix, "eg_frames": eg_frames, "nat_frames": nat_frames,
           "feats": feats, "eg2": eg2_mean, "desc": desc}
    return row, raw


# ------------------------------------------------------------------ gpu path
def gpu_stages(item, pmap, model, mode="torchcodec"):
    """GPU-resident attempt. Returns (row, raw).

    mode:
      * ``torchcodec``  — in-process NVDEC -> VRAM (true zero-copy for pix+eg2)
      * ``upload``      — ffmpeg hwdownload -> host, then ONE small upload of the
                          pix+eg2 frames to VRAM (partial; decode still crosses PCIe)
    """
    import torch
    path, s0 = item["path"], float(item["start_s"])
    s1 = float(item["end_s"])
    dur = s1 - s0
    lat, wall, notes = {}, {}, {}
    t_all = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    # io_probe
    t = time.perf_counter(); w0 = time.time()
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        frac = min(max(s0 / max(float(item.get("duration_s", dur)), 1e-9), 0.0), 1.0)
        fh.seek(min(int(frac * size), max(size - 65536, 0))); fh.read(65536)
    lat["io_probe"] = ms(t); wall["io_probe"] = [w0, time.time()]

    # ---- decode ----
    t = time.perf_counter(); w0 = time.time()
    if mode == "torchcodec":
        frames, fps, dmeta = zc.decode_torchcodec(path, s0, dur, device="cuda")
        notes["decode_backend"] = "torchcodec(VideoDecoder, device=cuda)"
        download = 0
        upload = 0
        eg_frames = frames
        eg_fps = fps
        pix_frames = frames
        nat_cpu = None
    else:
        pix_raw, eg_raw, nat_raw = decode_pipeline(path, s0, dur)
        pix_cpu = np.array(e8.frames_from_raw(pix_raw, H, W))       # (T,90,160,3)
        eg_cpu = np.array(e8.frames_from_raw(eg_raw, EGH, EGW))     # (n,288,512,3)
        nat_cpu = e8.frames_from_raw(nat_raw, NATIVE_H, NATIVE_W)   # (8,1080,1920,3)
        pix_frames = torch.from_numpy(pix_cpu).cuda()
        eg_frames = torch.from_numpy(eg_cpu).cuda()
        eg_fps = 2.0
        fps = 30.0
        download = int(pix_cpu.nbytes + eg_cpu.nbytes + nat_cpu.nbytes)
        upload = int(pix_cpu.nbytes + eg_cpu.nbytes)
        notes["decode_backend"] = "ffmpeg hwdownload -> host -> upload (pix+eg2)"
    lat["decode"] = ms(t); wall["decode"] = [w0, time.time()]
    notes["decode_native_frames"] = int(pix_frames.shape[0])

    # ---- pixel-26 on GPU ----
    t = time.perf_counter(); w0 = time.time()
    gfeats, gluma, grgb = zc.gpu_pixel_features(pix_frames, fps, w=W, h=H)
    lat["pixel_features"] = ms(t); wall["pixel_features"] = [w0, time.time()]

    # ---- descriptors: native frames must reach the CPU (cv2/numpy) ----
    t = time.perf_counter(); w0 = time.time()
    meta = e8._get_meta(path)
    if nat_cpu is None:  # torchcodec: download the 8 sampled native frames
        k = 8
        times = [s0 + (s1 - s0) * (i + 0.5) / k for i in range(k)]
        idxs = [max(0, min(int(round(tt * fps)), int(frames.shape[0]) - 1)) for tt in times]
        nat_cpu = frames[idxs].detach().to("cpu").numpy()
        download += int(nat_cpu.size)
    desc = e8.describe_from_native_frames(meta, list(nat_cpu), s0, s1)
    lat["descriptors"] = ms(t); wall["descriptors"] = [w0, time.time()]

    # ---- EG2 on GPU ----
    t = time.perf_counter(); w0 = time.time()
    gem2, n_eg = zc.gpu_eg2_embed(
        eg_frames, model, fps=eg_fps, step_s=0.5, cache=gpu_stages._cache,
        max_soft_tokens=e8._CFG["views"]["eg2"]["max_soft_tokens"])
    lat["eg2_embed"] = ms(t); wall["eg2_embed"] = [w0, time.time()]

    # ---- decision ----
    t = time.perf_counter(); w0 = time.time()
    vb = _verdict_block(gfeats, gem2, gluma.shape[0], n_eg, min(fps, 30.0), desc, pmap)
    lat["decision"] = ms(t); wall["decision"] = [w0, time.time()]
    lat["mosaic"] = 0.0
    lat["total"] = ms(t_all)
    vram = float(torch.cuda.max_memory_allocated() / MEGA)

    row = {
        "window_id": item["window_id"], "clip": item["clip"],
        "factory": item["factory"], "start_s": round(s0, 4),
        "latency_ms": {k: round(v, 3) for k, v in lat.items()},
        "n_frames": {"pixel": int(gluma.shape[0]), "eg2": int(n_eg)},
        "mode": mode, "notes": notes,
        "gpu_vram_peak_mib": vram,
        "bytes": {"download": int(download), "upload": int(upload)},
        "stage_wall": wall,
        **vb,
    }
    raw = {"feats": gfeats, "eg2": gem2, "desc": desc}
    return row, raw


gpu_stages._cache = {}


# --------------------------------------------------------------- decode probe
def _ff_time(cmd, timeout=180):
    t = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=timeout)
    dt = ms(t)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-200:].decode("utf-8", "replace"))
    return dt


def _nvdec_cmd(path, off, dur, w, h, download=True, null=False):
    vf = f"scale_cuda={w}:{h}:format=nv12"
    if download:
        vf += ",hwdownload,format=nv12,format=rgb24"
    out = ["-f", "null", "-"] if null else ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return (["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
             "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
             "-c:v", "hevc_cuvid",
             "-ss", f"{off:.3f}", "-t", f"{dur:.3f}", "-i", path,
             "-vf", vf] + out)


def _nvdec_cmd_generic(path, off, dur, w, h, codec, download=True, null=False):
    vf = f"scale_cuda={w}:{h}:format=nv12"
    if download:
        vf += ",hwdownload,format=nv12,format=rgb24"
    out = ["-f", "null", "-"] if null else ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    if codec:
        cmd += ["-c:v", codec]
    cmd += ["-ss", f"{off:.3f}", "-t", f"{dur:.3f}", "-i", path, "-vf", vf] + out
    return cmd


def _cpu_cmd(path, off, dur, w, h):
    return ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
            "-ss", f"{off:.3f}", "-t", f"{dur:.3f}", "-i", path,
            "-vf", f"scale={w}:{h},format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]


def decode_probe(sample, n_decode=8):
    """NVDEC vs CPU at 1080p30 HEVC and 480p H.264, with/without hwdownload."""
    from eval.gpu.decode import decode_window  # noqa: F401
    sample = sample[:max(1, min(n_decode, len(sample)))]
    out = {"windows": sample, "arms": []}
    # 1080p HEVC: use the sample clips (native 1920x1080). Time at native + 160x90.
    hevc = [(it, float(it["start_s"]), 10.0) for it in sample]
    # 480p H.264: 480x480 @10fps IndEgo
    h264 = [(None, float(60 + i * 12), 10.0) for i in range(min(len(sample), 8))]

    def run_arm(name, path, off, dur, w, h, codec, do_cpu=True):
        rec = {"arm": name, "path": path, "off": off, "w": w, "h": h,
               "res": f"{w}x{h}"}
        if do_cpu:
            try:
                rec["cpu_ms"] = _ff_time(_cpu_cmd(path, off, dur, w, h))
            except Exception as e:  # noqa: BLE001
                rec["cpu_error"] = str(e)[:120]
        try:
            rec["nvdec_hwdownload_ms"] = _ff_time(
                _nvdec_cmd_generic(path, off, dur, w, h, codec, download=True))
        except Exception as e:  # noqa: BLE001
            rec["nvdec_hwdownload_error"] = str(e)[:120]
        try:
            rec["nvdec_no_download_null_ms"] = _ff_time(
                _nvdec_cmd_generic(path, off, dur, w, h, codec, download=False, null=True))
        except Exception as e:  # noqa: BLE001
            rec["nvdec_no_download_error"] = str(e)[:120]
        out["arms"].append(rec)

    # aggregate over the sample windows
    agg = {}
    for it, off, dur in hevc:
        for w, h, tag in ((1920, 1080, "1080p_native"), (160, 90, "1080p_160x90")):
            key = ("hevc", tag)
            a = agg.setdefault(key, {"cpu": [], "dl": [], "nodl": [], "n": 0})
            p = it["path"]
            try:
                a["cpu"].append(_ff_time(_cpu_cmd(p, off, dur, w, h)))
            except Exception:  # noqa: BLE001
                pass
            try:
                a["dl"].append(_ff_time(_nvdec_cmd_generic(p, off, dur, w, h, "hevc_cuvid", download=True)))
            except Exception:  # noqa: BLE001
                pass
            try:
                a["nodl"].append(_ff_time(_nvdec_cmd_generic(p, off, dur, w, h, "hevc_cuvid", download=False, null=True)))
            except Exception:  # noqa: BLE001
                pass
            a["n"] += 1
    for _, off, dur in h264:
        for w, h, tag in ((480, 480, "480p_native"), (160, 90, "480p_160x90")):
            key = ("h264", tag)
            a = agg.setdefault(key, {"cpu": [], "dl": [], "nodl": [], "n": 0})
            try:
                a["cpu"].append(_ff_time(_cpu_cmd(INDEGO_480, off, dur, w, h)))
            except Exception:  # noqa: BLE001
                pass
            try:
                a["dl"].append(_ff_time(_nvdec_cmd_generic(INDEGO_480, off, dur, w, h, "h264_cuvid", download=True)))
            except Exception:  # noqa: BLE001
                pass
            try:
                a["nodl"].append(_ff_time(_nvdec_cmd_generic(INDEGO_480, off, dur, w, h, "h264_cuvid", download=False, null=True)))
            except Exception:  # noqa: BLE001
                pass
            a["n"] += 1

    def stats(v):
        v = np.asarray(v, float)
        if not len(v):
            return None
        return {"n": int(len(v)), "mean_ms": round(float(v.mean()), 1),
                "p50_ms": round(float(np.percentile(v, 50)), 1)}

    out["aggregate"] = {}
    for (codec, tag), a in sorted(agg.items()):
        cpu, dl, nodl = stats(a["cpu"]), stats(a["dl"]), stats(a["nodl"])
        rec = {"codec": codec, "res": tag, "n": a["n"], "cpu": cpu,
               "nvdec_hwdownload": dl, "nvdec_no_download": nodl}
        if cpu and dl:
            rec["nvdec_speedup_over_cpu"] = round(cpu["p50_ms"] / max(dl["p50_ms"], 1e-9), 3)
        if dl and nodl:
            rec["hwdownload_cost_p50_ms"] = round(dl["p50_ms"] - nodl["p50_ms"], 1)
        out["aggregate"][f"{codec}_{tag}"] = rec
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=30)
    ap.add_argument("--exp", choices=("all", "mosaic", "zerocopy", "decode"),
                    default="all")
    ap.add_argument("--gpu-mode", choices=("torchcodec", "upload"),
                    default="torchcodec")
    ap.add_argument("--out-dir", default=OUTDIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    e8.init_worker()
    if os.path.isdir(POD_EG2):
        e8._CFG["views"]["eg2"]["embed_model_dir"] = POD_EG2
    pmap = e8.profiles()

    man, sample = build_sample(args.sample_n)
    print(f"sample: {len(sample)} windows over "
          f"{len({i['factory'] for i in sample})} factories", flush=True)
    json.dump({"n": len(sample), "items": sample, "manifest": MANIFEST},
              open(os.path.join(args.out_dir, "sample.json"), "w"), indent=1)

    t_start = time.time()
    if args.exp in ("all", "mosaic"):
        run_mosaic(sample, pmap, args.out_dir)
    if args.exp in ("all", "decode"):
        print("decode probe ...", flush=True)
        res = decode_probe(sample)
        json.dump(res, open(os.path.join(args.out_dir, "decode_verdict.json"), "w"),
                  indent=1)
        print("wrote decode_verdict.json", flush=True)
    if args.exp in ("all", "zerocopy"):
        run_zerocopy(sample, pmap, args.out_dir, args.gpu_mode)

    print(f"total wall {time.time() - t_start:.0f}s", flush=True)


def run_mosaic(sample, pmap, out_dir):
    rows = []
    for i, it in enumerate(sample, 1):
        # decode once; only the mosaic build differs between the two passes
        row_on, raw = baseline_stages(it, pmap, do_mosaic=True)
        # pass OFF: reuse the decode by timing the same pipeline without mosaic
        row_off, _ = baseline_stages(it, pmap, do_mosaic=False)
        rows.append({"on": row_on, "off": row_off})
        print(f"  mosaic {i}/{len(sample)} {it['window_id']} "
              f"mosaic={row_on['latency_ms']['mosaic']:.0f}ms "
              f"on_total={row_on['latency_ms']['total']:.0f} "
              f"off_total={row_off['latency_ms']['total']:.0f}", flush=True)
    json.dump({"n": len(rows), "rows": rows, "device": DEVICE},
              open(os.path.join(out_dir, "mosaic.json"), "w"), indent=1)
    print("wrote mosaic.json", flush=True)


def run_zerocopy(sample, pmap, out_dir, gpu_mode):
    import torch
    model = e8._get_model(e8._CFG)[1]  # SentenceTransformer on cuda
    have_tc = zc.have_torchcodec()
    blockers = {}
    mode = gpu_mode
    if mode == "torchcodec" and not have_tc:
        blockers["torchcodec"] = "torchcodec not importable"
        mode = "upload"
    if mode == "torchcodec":
        # runtime probe: does in-process NVDEC -> VRAM actually work here?
        try:
            _f, _fps, _m = zc.decode_torchcodec(sample[0]["path"],
                                                float(sample[0]["start_s"]), 1.0,
                                                device="cuda")
            del _f
            blockers["torchcodec_probe"] = "ok"
        except Exception as ex:  # noqa: BLE001
            blockers["torchcodec_runtime"] = f"{type(ex).__name__}: {ex}"[:200]
            mode = "upload"
    print(f"zerocopy: mode={mode} torchcodec_available={have_tc} "
          f"blockers={blockers}", flush=True)
    rows = []
    for i, it in enumerate(sample, 1):
        try:
            base_row, base_raw = baseline_stages(it, pmap, do_mosaic=False)
        except Exception as ex:  # noqa: BLE001
            blockers.setdefault("baseline", []).append(
                {"window_id": it["window_id"], "error": f"{type(ex).__name__}: {ex}"[:200]})
            print(f"  zc {i}/{len(sample)} {it['window_id']} BASELINE ERROR {ex}", flush=True)
            continue
        try:
            gpu_row, gpu_raw = gpu_stages(it, pmap, model, mode=mode)
        except Exception as ex:  # noqa: BLE001
            blockers.setdefault("gpu_path", []).append(
                {"window_id": it["window_id"], "error": f"{type(ex).__name__}: {ex}"[:200]})
            print(f"  zc {i}/{len(sample)} {it['window_id']} GPU ERROR {ex}", flush=True)
            continue
        # feature/embedding deltas (correctness, not timing)
        order = e8._CFG["views"]["pixel"]["feature_names"]
        fd = {k: round(float(gpu_raw["feats"][k] - base_raw["feats"][k]), 5) for k in order}
        cos = float(np.dot(gpu_raw["eg2"], base_raw["eg2"]) /
                    max(np.linalg.norm(gpu_raw["eg2"]) * np.linalg.norm(base_raw["eg2"]), 1e-12))
        rows.append({
            "window_id": it["window_id"], "clip": it["clip"],
            "factory": it["factory"], "mode": mode,
            "base": {k: base_row[k] for k in
                     ("latency_ms", "verdict", "score", "features_digest", "bytes")},
            "gpu": {k: gpu_row[k] for k in
                    ("latency_ms", "verdict", "score", "features_digest", "bytes",
                     "notes", "gpu_vram_peak_mib")},
            "feature_delta": fd,
            "eg2_cosine": round(cos, 5),
            "verdict_agree": gpu_row["verdict"] == base_row["verdict"],
        })
        print(f"  zc {i}/{len(sample)} {it['window_id']} mode={mode} "
              f"base_total={base_row['latency_ms']['total']:.0f} "
              f"gpu_total={gpu_row['latency_ms']['total']:.0f} "
              f"cos={cos:.4f} agree={gpu_row['verdict']==base_row['verdict']}", flush=True)
        json.dump({"n": len(rows), "device": DEVICE, "gpu_mode": mode,
                   "torchcodec_available": have_tc, "blockers": blockers,
                   "rows": rows,
                   "pixel_order": e8._CFG["views"]["pixel"]["feature_names"]},
                  open(os.path.join(out_dir, "zerocopy.json"), "w"), indent=1)
    res = {"n": len(rows), "device": DEVICE, "gpu_mode": mode,
           "torchcodec_available": have_tc, "blockers": blockers, "rows": rows,
           "pixel_order": e8._CFG["views"]["pixel"]["feature_names"]}
    json.dump(res, open(os.path.join(out_dir, "zerocopy.json"), "w"), indent=1)
    print("wrote zerocopy.json", flush=True)


if __name__ == "__main__":
    main()
