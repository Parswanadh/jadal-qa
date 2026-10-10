"""Factory 1080p30 HEVC end-to-end (REAL video, NOT looped IndEgo).

Corpus: /home/parshu/projects/awing/data/ego10k_scale/clips/factory_*/worker_*/*.mp4
(20-min 1920x1080 HEVC 30fps; use only complete files: size>500MB + ffprobe>1100s).

Modes (thread caps set inside; run under lock):
  --sweep    F1 (~6 min): HEVC CPU 1-core baseline; NVDEC 1/2/4 sessions;
             kernel batch/VRAM sweep at T=300 (my proc <= 4.5 GiB)
  --sustain [minutes] F2: end-to-end NVDEC readers -> GPU convert -> batched
             kernels -> S0 exact + S1 exploratory; telemetry timestamps.

Writes eval/results/gpu/factory.json (merged sections).
"""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "results", "gpu", "factory.json")
FROOT = "/home/parshu/projects/awing/data/ego10k_scale/clips"
W, H, FPS = 160, 90, 30
T = 300  # 10-s window at 30 fps


def complete_files():
    """Complete factory mp4s (size + ffprobe duration). Cached to OUT dir."""
    import glob
    cache = os.path.join(HERE, "..", "results", "gpu", "factory_files.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    cands = sorted(glob.glob(os.path.join(FROOT, "factory_*", "worker_*", "*.mp4")))
    good = []
    for p in cands:
        try:
            if os.path.getsize(p) < 500 * 10**6:
                continue
            q = subprocess.run(["ffprobe", "-hide_banner", "-v", "error",
                                "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1", p],
                               capture_output=True, text=True, timeout=30)
            if float(q.stdout.strip()) > 1100:
                good.append(p)
        except Exception:
            pass
    json.dump(good, open(cache, "w"))
    return good


def load():
    if os.path.exists(OUT):
        return json.load(open(OUT))
    return {"device": "RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU"}


def save(d):
    json.dump(d, open(OUT, "w"), indent=1)


def drain(pipe, fs, secs):
    """Read pipe for secs video worth; return (frames, wall)."""
    t0 = time.perf_counter()
    buf, n = b"", 0
    p = pipe
    while True:
        ch = p.stdout.read(1 << 20)
        if not ch:
            break
        buf += ch
        while len(buf) >= fs:
            buf = buf[fs:]
            n += 1
    p.wait()
    return n, time.perf_counter() - t0, p.returncode


def mode_sweep():
    import torch
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    d = load()
    files = complete_files()
    d["corpus"] = {"n_complete": len(files, ) if False else len(files),
                   "example": files[0] if files else None,
                   "spec": "1920x1080 HEVC 30fps ~20min"}
    if not files:
        d["error"] = "no complete factory files yet"
        save(d)
        print("no complete files")
        return
    clip = files[0]
    SEC = 30
    # (a) HEVC CPU 1-core baseline (taskset), gray 160x90 pipe
    cmd = ["taskset", "-c", "7", "ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-ss", "60", "-t", str(SEC), "-i", clip,
           "-vf", "scale=160:90,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL)
    n, wall, rc = drain(p, W * H, SEC)
    d["cpu_hevc_1core"] = {"frames": n, "wall_s": round(wall, 1), "rc": rc,
                           "frames_per_s": round(n / wall, 1),
                           "video_s_per_wall_s": round(n / FPS / wall, 2)}
    # (b) NVDEC sessions 1/2/4
    import threading
    d["nvdec_hevc"] = []
    for conc in [1, 2, 4]:
        fs = W * H * 3 // 2
        t0 = time.perf_counter()
        procs = []
        for i in range(conc):
            c = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
                 "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                 "-c:v", "hevc_cuvid",
                 "-ss", str(60 + i * 90), "-t", str(SEC), "-i", clip,
                 "-vf", f"scale_cuda={W}:{H}:format=nv12,hwdownload,format=nv12",
                 "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]
            procs.append(subprocess.Popen(c, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          stdin=subprocess.DEVNULL))
        tot, rcs = 0, []
        for pr in procs:
            b = pr.stdout.read()
            pr.wait()
            tot += len(b)
            rcs.append(pr.returncode)
        wall = time.perf_counter() - t0
        frames = tot / fs
        d["nvdec_hevc"].append({"sessions": conc, "rcs": rcs,
                                "frames": round(frames, 1),
                                "wall_s": round(wall, 1),
                                "frames_per_s": round(frames / wall, 1),
                                "video_s_per_wall_s": round(frames / FPS / wall, 2)})
    # (c) kernel batch/VRAM sweep at T=300 fp16
    from jadal.gpu.features import gpu_window_features_batch
    d["kernels_T300"] = {}
    for B in [4, 8, 16, 32]:
        torch.cuda.reset_peak_memory_stats()
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
            d["kernels_T300"][f"B{B}"] = {
                "windows_s": round(wps, 1), "hours_per_hour": round(wps * 10, 1),
                "vram_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
            del lb, rb
        except RuntimeError as e:
            d["kernels_T300"][f"B{B}"] = {"oom": str(e)[:120]}
    d["vram_cap_mib"] = 4608
    save(d)
    print(json.dumps({k: d[k] for k in ("cpu_hevc_1core", "nvdec_hevc",
                                        "kernels_T300")}, indent=1)[:2500])


def mode_sustain(minutes=12, ndec=2, B=8):
    import queue
    import threading
    import numpy as np
    import torch
    from eval.gpu.stream import S0_RULES
    from eval.gpu.telemetry import now_str
    from jadal.gpu.features import gpu_window_features_batch, FEATURE_NAMES
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    d = load()
    files = complete_files()
    assert len(files) >= 1, "no complete factory files"
    # S1 head trained on IndEgo 10fps features -> EXPLORATORY at 30fps
    from sklearn.linear_model import LogisticRegression
    feats = json.load(open(os.path.join(HERE, "..", "results", "gpu",
                                        "features.json")))["windows"]
    gold = json.load(open("/home/parshu/projects/jadal-qa-wt/spectral_data/"
                          "manifest_gold.json"))["items"]
    names = FEATURE_NAMES
    ids = [i["id"] for i in gold if i["id"] in feats and "full" in feats[i["id"]]]
    X = np.stack([[feats[i]["full"][k] for k in names] for i in ids])
    y = np.array([next(g["label"] for g in gold if g["id"] == i) for i in ids])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((X - mu) / sd, y)

    raw_q = queue.Queue(maxsize=8)
    stop_ev = threading.Event()

    def raw_reader(path, off, dur):
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
               "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
               "-c:v", "hevc_cuvid",
               "-ss", str(off), "-t", str(dur), "-i", path,
               "-vf", f"scale_cuda={W}:{H}:format=nv12,hwdownload,format=nv12",
               "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL)
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
                            raw_q.put(win, timeout=10)
                        except queue.Full:
                            pass
                        win = []
        finally:
            try:
                p.kill()
            except Exception:
                pass

    # round-robin files across readers, long takes
    per = max(int(minutes * 60 / max(ndec, 1)) + 120, 300)
    for i in range(ndec):
        threading.Thread(target=raw_reader,
                         args=(files[i % len(files)], (i * 211) % 600, per),
                         daemon=True).start()
    THR, MARGIN = 0.6, 0.25
    exits = {"S0": 0, "S1_accept": 0, "S1_reject": 0, "S2": 0}
    t_start = now_str()
    t0 = time.perf_counter()
    done = drops = padded = 0
    h2d = 0
    while time.perf_counter() - t0 < minutes * 60:
        try:
            win = raw_q.get(timeout=10)
        except queue.Empty:
            continue
        if len(win) < T:
            padded += 1
            win = win + win[-1:] * (T - len(win))
        try:
            yb = np.stack([np.frombuffer(f[:W * H], dtype=np.uint8).reshape(H, W)
                           for f in win])
            uvb = np.stack([np.frombuffer(f[W * H:], dtype=np.uint8).reshape(H // 2, W // 2, 2)
                            for f in win])
            y = torch.from_numpy(yb).cuda(non_blocking=True)
            uv = torch.from_numpy(uvb).cuda(non_blocking=True)
            h2d += y.numel() + uv.numel()
            with torch.no_grad():
                u = uv[..., 0].repeat_interleave(2, 1).repeat_interleave(2, 2).float()
                v = uv[..., 1].repeat_interleave(2, 1).repeat_interleave(2, 2).float()
                c = y.float() - 16.0
                dd, e = u - 128.0, v - 128.0
                rgb = torch.stack([1.164 * c + 1.596 * e,
                                   1.164 * c - 0.391 * dd - 0.813 * e,
                                   1.164 * c + 2.018 * dd], -1).clamp(0, 255) / 255.0
                luma = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                        + 0.114 * rgb[..., 2])
                # S0 probe on first 2 s (60 frames), GPU-side
                g2 = luma[:60]
                probe = {"s0p95": float(torch.quantile(g2, 0.95)),
                         "s0p50": float(torch.quantile(g2, 0.50)),
                         "s0var": float(g2[:10].mean(0).var()),
                         "s0bright": 1.0 if float(torch.quantile(g2, 0.95)) > 0.95 else 0.0,
                         "s0mot": float((g2[1:] - g2[:-1]).abs().mean())}
                hit = next((nm for nm, k, op, th in S0_RULES
                            if (op == "<" and probe[k] < th)
                            or (op == ">" and probe[k] > th)), None)
                if hit:
                    exits["S0"] += 1
                    done += 1
                    continue
                F = gpu_window_features_batch(luma.half(), rgb.half(),
                                              fps=30.0).float().cpu().numpy()
            P = clf.predict_proba((F - mu) / sd)[:, 1][0]
            if P >= THR + MARGIN:
                exits["S1_accept"] += 1
            elif P <= THR - MARGIN:
                exits["S1_reject"] += 1
            else:
                exits["S2"] += 1
            done += 1
        except Exception:
            drops += 1
    wall = time.perf_counter() - t0
    stop_ev.set()
    torch.cuda.synchronize()
    wps = done / wall
    d["sustain"] = {
        "minutes_wall": round(wall / 60, 2), "windows": done, "drops": drops,
        "padded": padded, "video_min": round(done * 10 / 60, 1),
        "decoders": ndec, "batch": 1, "tier": "160x90",
        "windows_s": round(wps, 2), "hours_per_hour": round(wps * 10, 1),
        "video_s_per_wall_s": round(wps * 10, 1),
        "h2d_gbps": round((h2d / 1e9) / wall, 3),
        "exits": exits, "exit_frac": {k: round(v / max(done, 1), 3)
                                      for k, v in exits.items()},
        "s1_note": "EXPLORATORY at 30fps: head trained on 10fps IndEgo features",
        "gpu_peak_vram_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "t_start": t_start, "t_end": now_str(),
    }
    rtf = wps * 10
    d["extrapolation_1080p"] = {"rtf": round(rtf, 1),
                                "gpus_for_10kh_in_2h": round(5000 / rtf, 1),
                                "gpus_for_10kh_in_3h": round(3333 / rtf, 1)}
    save(d)
    print(json.dumps({"sustain": d["sustain"],
                      "extrapolation_1080p": d["extrapolation_1080p"]}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--sweep":
        mode_sweep()
    elif sys.argv[1] == "--sustain":
        mode_sustain(float(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        raise SystemExit("usage: factory.py --sweep | --sustain [minutes]")
