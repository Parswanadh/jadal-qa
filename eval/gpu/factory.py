"""Factory-1080p/30fps-HEVC end-to-end (orchestrator order, replaces looped corpus).

Corpus: /home/parshu/projects/awing/data/ego10k_scale/clips/factory_*/worker_*/*.mp4
(20-min 1920x1080 HEVC 30fps; use only complete files: size>500MB + ffprobe>1100s).

Modes (each ONE lock hold; thread caps set by caller env):
  --sweep    F1: CPU-1core HEVC baseline + NVDEC 1/2/4 sessions + batch/VRAM sweep
  --sustain [minutes]  F2: end-to-end NVDEC->GPU-convert->kernels->S0/S1 sustained run
Writes eval/results/gpu/factory.json (+ ladder rungs).
Throttle: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2, <=3 procs.
"""
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "results", "gpu", "factory.json")
LAD = os.path.join(HERE, "..", "results", "gpu", "ladder.json")
CLIP_GLOB = "/home/parshu/projects/awing/data/ego10k_scale/clips/factory_*/worker_*/*.mp4"
W, H, FPS, T = 160, 90, 30, 300  # 10-s window at 30 fps


def complete_clips():
    """Complete files only (downloader still running). Light: size prefilter."""
    cands = sorted(glob.glob(CLIP_GLOB))
    big = [c for c in cands if os.path.getsize(c) > 500 * 1024 * 1024]
    ok = []
    for c in big:
        try:
            p = subprocess.run(["ffprobe", "-hide_banner", "-v", "error",
                                "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1", c],
                               capture_output=True, text=True, timeout=30)
            if float(p.stdout.strip()) > 1100:
                ok.append(c)
        except Exception:
            pass
    return ok


def nvdec_cmd(path, off, dur, w=W, h=H):
    return (["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
             "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
             "-c:v", "hevc_cuvid",
             "-ss", str(off)] + (["-t", str(dur)] if dur else []) +
            ["-i", path, "-vf",
             f"scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12",
             "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"])


def mode_sweep():
    res = {"device": "RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU",
           "regenerate": "flock ... /home/parshu/envs/jadal/bin/python eval/gpu/factory.py --sweep"}
    clips = complete_clips()
    res["corpus"] = {"n_complete": len(clips), "files": [os.path.basename(c) for c in clips[:8]]}
    if not clips:
        res["error"] = "no complete clips yet"
        json.dump(res, open(OUT, "w"), indent=1)
        print("no complete clips"); return
    f0 = clips[0]
    # (a) CPU 1-core HEVC baseline, 60 s video
    cmd = ["taskset", "-c", "7", "ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-ss", "60", "-t", "60", "-i", f0,
           "-vf", f"scale={W}:{H},format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=300)
    wall = time.perf_counter() - t0
    frames = len(p.stdout) // (W * H)
    res["cpu_1core"] = {"rc": p.returncode, "frames": frames,
                        "frames_per_s": round(frames / wall, 1),
                        "video_s_per_wall_s": round(frames / FPS / wall, 2),
                        "video_s": 60}
    # (b) NVDEC sessions 1/2/4, 60 s video each
    expect = 60 * FPS * W * H * 3 // 2
    dec = {}
    for conc in [1, 2, 4]:
        t0 = time.perf_counter()
        procs = [subprocess.Popen(nvdec_cmd(clips[i % len(clips)], 60 + i * 90, 60),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  stdin=subprocess.DEVNULL) for i in range(conc)]
        tots, rcs = [], []
        for p in procs:
            out, _ = p.communicate()
            tots.append(len(out))
            rcs.append(p.returncode)
        wall = time.perf_counter() - t0
        frames = sum(tots) / (W * H * 3 // 2)
        ok = all(r == 0 for r in rcs) and all(t == expect for t in tots)
        dec[f"x{conc}"] = {"frames_per_s": round(frames / wall, 1) if ok else None,
                           "video_s_per_wall_s": round(frames / FPS / wall, 2) if ok else None,
                           "exact_bytes": ok, "wall_s": round(wall, 1)}
    res["nvdec_sessions"] = dec
    # (c) kernel batch/VRAM sweep (fp16, T=300) + (d) convert sanity
    import torch
    from jadal.gpu.features import gpu_window_features_batch
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    sw = {}
    for B in [4, 8, 16, 32]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        try:
            lb = torch.randn(B, T, H, W, dtype=torch.float16, device="cuda:0")
            rb = torch.randn(B, T, H, W, 3, dtype=torch.float16, device="cuda:0")
            gpu_window_features_batch(lb, rb, fps=30.0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(3):
                    gpu_window_features_batch(lb, rb, fps=30.0)
            torch.cuda.synchronize()
            wps = B * 3 / (time.perf_counter() - t0)
            vram = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            sw[f"B{B}"] = {"windows_s": round(wps, 1),
                           "hours_per_hour": round(wps * 10, 1),
                           "vram_mib": vram,
                           "over_budget": vram > 4608}
            del lb, rb
        except RuntimeError as e:
            sw[f"B{B}"] = {"error": str(e)[:120]}
    res["kernel_sweep_T300"] = sw
    json.dump(res, open(OUT, "w"), indent=1)
    lad = json.load(open(LAD))
    lad["rungs"].append({"rung": "F1_factory_sweep",
                         "cpu_1core_hevc_rtf": res["cpu_1core"]["video_s_per_wall_s"],
                         "nvdec_rtf": {k: v["video_s_per_wall_s"] for k, v in dec.items()},
                         "kernels_hph": {k: (v.get("hours_per_hour"), v.get("vram_mib"))
                                         for k, v in sw.items()},
                         "why": "1080p30 HEVC reality: CPU-1core vs NVDEC sessions; "
                                "batch/VRAM ceiling at T=300"})
    json.dump(lad, open(LAD, "w"), indent=1)
    print(json.dumps(res, indent=1)[:2500])


def mode_sustain(minutes=12, ndec=3, B=8):
    import numpy as np
    import torch
    from eval.gpu.stream import S0_RULES
    from eval.gpu.telemetry import now_str
    from jadal.gpu.features import gpu_window_features_batch, FEATURE_NAMES
    from sklearn.linear_model import LogisticRegression
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    clips = complete_clips()
    res = {"device": "RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU",
           "regenerate": "flock ... /home/parshu/envs/jadal/bin/python eval/gpu/factory.py --sustain",
           "n_complete": len(clips)}
    if len(clips) < 2:
        res["error"] = "need >=2 complete clips"
        json.dump(res, open(OUT, "w"), indent=1)
        print("not enough clips"); return
    # S1 head: trained on IndEgo 10fps features -> EXPLORATORY on 30fps factory
    feats = json.load(open(os.path.join(HERE, "..", "results", "gpu", "features.json")))["windows"]
    names = FEATURE_NAMES
    X = np.stack([v["full"] for v in feats.values() if "full" in v])
    ids = [k for k, v in feats.items() if "full" in v]
    gold = {it["id"]: it["label"] for it in
            json.load(open("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"))["items"]}
    gid = [i for i in ids if i in gold]
    Xg = np.stack([[feats[i]["full"][k] for k in names] for i in gid])
    yg = np.array([gold[i] for i in gid])
    mu, sd = Xg.mean(0), Xg.std(0) + 1e-12
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xg - mu) / sd, yg)

    raw_q = queue.Queue(maxsize=8)
    stop_ev = threading.Event()

    def reader(path, off):
        p = subprocess.Popen(nvdec_cmd(path, off, None), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        fs = W * H * 3 // 2
        buf, win = b"", []
        try:
            while not stop_ev.is_set():
                ch = p.stdout.read(1 << 20)
                if not ch:
                    break
                buf += ch
                while len(buf) >= fs and not stop_ev.is_set():
                    win.append(buf[:fs])
                    buf = buf[fs:]
                    if len(win) == T:
                        try:
                            raw_q.put(win, timeout=5)
                        except queue.Full:
                            pass
                        win = []
        finally:
            try:
                p.kill()
            except Exception:
                pass

    for i in range(ndec):
        threading.Thread(target=reader, args=(clips[i % len(clips)], 120 + i * 150),
                         daemon=True).start()
    h2d_stream = torch.cuda.Stream()
    def_stream = torch.cuda.current_stream()
    t_start = now_str()
    t0 = time.perf_counter()
    exits = {"S0": 0, "S1_accept": 0, "S1_reject": 0, "S2": 0}
    s1_note = "EXPLORATORY: head trained on IndEgo 480p/10fps, applied to factory 1080p/30fps"
    done = padded = 0
    h2d = 0
    while time.perf_counter() - t0 < minutes * 60:
        try:
            win = raw_q.get(timeout=10)
        except queue.Empty:
            break
        if len(win) < T:  # ragged tail: pad with last frame bytes
            win = win + [win[-1]] * (T - len(win))
            padded += 1
        yb = np.stack([np.frombuffer(f[:W * H], dtype=np.uint8).reshape(H, W) for f in win])
        uvb = np.stack([np.frombuffer(f[W * H:], dtype=np.uint8).reshape(H // 2, W // 2, 2)
                        for f in win])
        with torch.cuda.stream(h2d_stream):
            y = torch.from_numpy(yb).cuda(non_blocking=True)
            uv = torch.from_numpy(uvb).cuda(non_blocking=True)
        h2d += y.numel() + uv.numel()
        def_stream.wait_stream(h2d_stream)
        with torch.no_grad():
            u = uv[..., 0].repeat_interleave(2, 1).repeat_interleave(2, 2).float()
            v = uv[..., 1].repeat_interleave(2, 1).repeat_interleave(2, 2).float()
            c = y.float() - 16.0
            d, e = u - 128.0, v - 128.0
            rgb = torch.stack([1.164 * c + 1.596 * e, 1.164 * c - 0.391 * d - 0.813 * e,
                               1.164 * c + 2.018 * d], -1).clamp(0, 255) / 255.0
            luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            F = gpu_window_features_batch(luma.half(), rgb.half(), fps=30.0).float().cpu().numpy()
        # S0 on first 2 s (GPU luma already resident)
        g2 = luma[:60].float()
        probe = {"s0p95": float(torch.quantile(g2, 0.95)),
                 "s0p50": float(torch.quantile(g2, 0.50)),
                 "s0var": 1.0,
                 "s0bright": 1.0 if float(torch.quantile(g2, 0.95)) > 0.95 else 0.0,
                 "s0mot": float((g2[1:] - g2[:-1]).abs().mean())}
        hit = None
        for name, key, op, thr in S0_RULES:
            v = probe[key]
            if (op == "<" and v < thr) or (op == ">" and v > thr):
                hit = name
                break
        if hit:
            exits["S0"] += 1
        else:
            p = float(clf.predict_proba((F - mu) / sd)[0, 1])
            if p >= 0.85:
                exits["S1_accept"] += 1
            elif p <= 0.35:
                exits["S1_reject"] += 1
            else:
                exits["S2"] += 1
        done += 1
        del y, uv, rgb, luma, F
    wall = time.perf_counter() - t0
    stop_ev.set()
    torch.cuda.synchronize()
    wps = done / wall
    res.update({"sustain_min": round(wall / 60, 2), "windows": done, "padded": padded,
                "video_min": round(done * 10 / 60, 1), "decoders": ndec, "batch": 1,
                "windows_s": round(wps, 2), "hours_per_hour": round(wps * 10, 1),
                "video_s_per_wall_s": round(wps * 10, 1),
                "h2d_gbps": round((h2d / 1e9) / wall, 3),
                "exits": exits, "s1_note": s1_note,
                "gpu_peak_vram_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                "vram_process_budget_mib": 4608,
                "t_start": t_start, "t_end": now_str()})
    rtf = wps * 10
    res["extrapolation"] = {"rtf": round(rtf, 1),
                            "gpus_for_10kh_in_2h": round(5000 / rtf, 1),
                            "gpus_for_10kh_in_3h": round(3333 / rtf, 1)}
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out.update(res)
    json.dump(out, open(OUT, "w"), indent=1)
    lad = json.load(open(LAD))
    lad["rungs"].append({"rung": "F2_factory_sustain",
                         "hours_per_hour": res["hours_per_hour"],
                         "why": "real 1080p30 HEVC end-to-end: NVDEC + GPU convert + "
                                "kernels + S0/exploratory-S1, sustained"})
    json.dump(lad, open(LAD, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("windows", "video_min", "windows_s",
                                          "hours_per_hour", "exits", "gpu_peak_vram_mib",
                                          "h2d_gbps", "t_start", "t_end")}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--sweep":
        mode_sweep()
    elif sys.argv[1] == "--sustain":
        mode_sustain(float(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        raise SystemExit("usage: factory.py --sweep | --sustain [minutes]")
