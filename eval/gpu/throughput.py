"""M4 end-to-end throughput + optimisation ladder (centrepiece).

Modes (rungs build on each other; every rung re-measures):
  --r0    CPU baseline: ffmpeg CPU decode + cpu_window_features (no lock)
  --holdA GPU kernels: memcpy peaks, per-window vs batched gpu kernels,
          fp16 sweep, rgb24 NVDEC format probe (ONE lock hold, ~3 min)
  --holdB NVDEC pipeline + sustained >=5 min run + dmon + VRAM + sweep
          (ONE lock hold, ~8 min)

Writes eval/results/gpu/throughput.json and eval/results/gpu/ladder.json
(assembled incrementally; each mode merges its section).
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
THR = os.path.join(HERE, "..", "results", "gpu", "throughput.json")
LAD = os.path.join(HERE, "..", "results", "gpu", "ladder.json")
GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"

W, H, FPS = 160, 90, 10
WIN_FRAMES = 100  # 10 s @ 10 fps


def load():
    for p in (THR, LAD):
        if os.path.exists(p):
            return json.load(open(THR)), json.load(open(LAD))
    return ({"device": "RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU"}, {"rungs": []})


def save(thr, lad):
    json.dump(thr, open(THR, "w"), indent=1)
    json.dump(lad, open(LAD, "w"), indent=1)


def clips7():
    gold = json.load(open(GOLD))["items"]
    seen, out = set(), []
    for it in gold:
        if it["clip"] not in seen:
            seen.add(it["clip"])
            out.append(it)
        if len(out) == 7:
            break
    return out


def windows_per_hour(windows_s):
    return windows_s * 10.0  # 10-s windows: win/s * 10 video-s = video-s/wall-s = h/h


def mode_r0(nwin=10):
    from eval.gpu.decode import decode_window, safe_offset
    from jadal.gpu.features import cpu_window_features
    items = clips7()
    t0 = time.perf_counter()
    for i in range(nwin):
        it = items[i % len(items)]
        rgb, fps = decode_window(it["path"], safe_offset(it["path"], i, 10.0),
                                 10.0, W, H, "rgb24")
        luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        cpu_window_features(luma, rgb, fps=min(fps, 30.0))
    wall = time.perf_counter() - t0
    wps = nwin / wall
    thr, lad = load()
    thr["R0_cpu"] = {"windows_s": round(wps, 2),
                     "hours_per_hour": round(windows_per_hour(wps), 1),
                     "video_s_per_wall_s": round(wps * 10, 1),
                     "nwin": nwin, "wall_s": round(wall, 1),
                     "method": "ffmpeg CPU decode 160x90 rgb + cpu_window_features, 1 proc"}
    lad["rungs"].append({"rung": "R0_cpu_baseline",
                         "hours_per_hour": thr["R0_cpu"]["hours_per_hour"],
                         "why": "starting point: everything on CPU, one window at a time"})
    save(thr, lad)
    print(json.dumps(thr["R0_cpu"], indent=1))


def gpu_query():
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=utilization.gpu,utilization.decoder,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip()
    except Exception:
        return ""


def mode_holdA():
    import torch
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    from jadal.gpu.features import gpu_window_features_batch
    thr, lad = load()
    # (a) memcpy peaks (self-contained; microbench.json corroborates)
    n = 256 * 1024 * 1024 // 4
    a = torch.empty(n, dtype=torch.float32, device="cuda:0")
    b = torch.empty(n, dtype=torch.float32, device="cuda:0")
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20):
        b.copy_(a)
    e1.record()
    torch.cuda.synchronize()
    d2d = (256e6 / 1e9) / (e0.elapsed_time(e1) / 20 / 1e3)
    hp = torch.empty(n, dtype=torch.float32).pin_memory()
    e0.record()
    for _ in range(20):
        b.copy_(hp, non_blocking=True)
    e1.record()
    torch.cuda.synchronize()
    h2d = (256e6 / 1e9) / (e0.elapsed_time(e1) / 20 / 1e3)
    thr["memcpy"] = {"d2d_gbps": round(d2d, 2), "h2d_pinned_gbps": round(h2d, 2),
                     "theoretical_gb_s": 256, "pcie_theoretical_gb_s": 16,
                     "note": "vs 256 GB/s GDDR6 theory and ~16 GB/s PCIe4x8 theory"}
    # (b) kernel sweep: batch 1 vs 8 vs 32, fp16 input
    T = WIN_FRAMES
    sw = {}
    for B in [1, 8, 32]:
        lb = torch.randn(B, T, H, W, dtype=torch.float16, device="cuda:0")
        rb = torch.randn(B, T, H, W, 3, dtype=torch.float16, device="cuda:0")
        gpu_window_features_batch(lb, rb, fps=10.0)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 5
        with torch.no_grad():
            for _ in range(reps):
                gpu_window_features_batch(lb, rb, fps=10.0)
        torch.cuda.synchronize()
        wps = B * reps / (time.perf_counter() - t0)
        sw[f"B{B}"] = {"windows_s": round(wps, 1),
                       "hours_per_hour": round(windows_per_hour(wps), 1)}
    thr["kernels_only"] = sw
    thr["gpu_peak_vram_mib_holdA"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    # (c) NVDEC rgb24 format probe on one clip
    it = clips7()[0]
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-ss", "60", "-t", "5", "-i", it["path"],
           "-vf", f"scale_cuda={W}:{H},hwdownload,format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=90)
    expect = 5 * FPS * W * H * 3
    thr["nvdec_rgb24_probe"] = {"rc": p.returncode, "bytes": len(p.stdout),
                                "expected": expect,
                                "ok": p.returncode == 0 and len(p.stdout) == expect,
                                "stderr_tail": p.stderr[-300:].decode("utf-8", "replace")}
    lad["rungs"].append({"rung": "R1_R2_kernels",
                         "hours_per_hour": {k: v["hours_per_hour"] for k, v in sw.items()},
                         "why": "batched GPU kernels remove per-launch overhead; "
                                "kernels-only ceiling before decode is added"})
    save(thr, lad)
    print(json.dumps({k: thr[k] for k in ("memcpy", "kernels_only",
                                          "nvdec_rgb24_probe",
                                          "gpu_peak_vram_mib_holdA")}, indent=1)[:2500])


class Dmon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.rows = []
        self._run = True

    def run(self):
        while self._run:
            q = gpu_query()
            if q:
                self.rows.append((time.time(), q))
            time.sleep(1.0)

    def stop(self):
        self._run = False


def nvdec_reader(path, off, q, stop_ev, scale="160:90"):
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-ss", str(off), "-i", path,
           "-vf", f"scale_cuda={scale}:format=nv12,hwdownload,format=nv12",
           "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]
    import numpy as np
    from eval.gpu.decode import nv12_to_rgb
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL)
    fs = W * H * 3 // 2
    buf = b""
    win = []
    try:
        while not stop_ev.is_set():
            ch = p.stdout.read(1 << 20)
            if not ch:
                break
            buf += ch
            while len(buf) >= fs and not stop_ev.is_set():
                f = np.frombuffer(buf[:fs], dtype=np.uint8)
                buf = buf[fs:]
                win.append(nv12_to_rgb(f[:W * H].reshape(H, W),
                                       f[W * H:].reshape(H // 2, W // 2, 2)))
                if len(win) == WIN_FRAMES:
                    try:
                        q.put(np.stack(win).astype(np.float32), timeout=5)
                    except queue.Full:
                        pass
                    win = []
    finally:
        try:
            p.kill()
        except Exception:
            pass


def mode_holdB(sustain_s=360, ndec=4):
    import numpy as np
    import torch
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    from jadal.gpu.features import gpu_window_features_batch
    items = clips7()
    q = queue.Queue(maxsize=16)
    stop_ev = threading.Event()
    readers = []
    from eval.gpu.decode import safe_offset
    for i in range(ndec):
        it = items[i % len(items)]
        t = threading.Thread(target=nvdec_reader,
                             args=(it["path"], safe_offset(it["path"], i, 60.0, step=97), q, stop_ev),
                             daemon=True)
        t.start()
        readers.append(t)
    dmon = Dmon()
    dmon.start()
    batch = []
    B = 8
    done = drops = 0
    h2d_bytes = 0
    t0 = time.perf_counter()
    lb_rb = None
    while time.perf_counter() - t0 < sustain_s:
        try:
            win = q.get(timeout=5)
        except queue.Empty:
            continue
        batch.append(win)
        if len(batch) == B:
            t = np.stack(batch)  # (B,T,H,W,3)
            batch = []
            try:
                ten = torch.from_numpy(t).half().cuda(non_blocking=True)
                h2d_bytes += ten.numel() * 2
                luma = (0.299 * ten[..., 0] + 0.587 * ten[..., 1]
                        + 0.114 * ten[..., 2])
                with torch.no_grad():
                    gpu_window_features_batch(luma, ten, fps=10.0)
                done += B
            except Exception:
                drops += B
    wall = time.perf_counter() - t0
    stop_ev.set()
    dmon.stop()
    torch.cuda.synchronize()
    wps = done / wall
    gpus = [r[1] for r in dmon.rows]
    sm = []
    for g in gpus:
        try:
            sm.append(float(g.split(",")[0]))
        except Exception:
            pass
    thr, lad = load()
    thr["R3_pipeline"] = {
        "sustain_s": round(wall, 1), "windows": done, "drops": drops,
        "decoders": ndec, "batch": B,
        "windows_s": round(wps, 2), "hours_per_hour": round(windows_per_hour(wps), 1),
        "video_s_per_wall_s": round(wps * 10, 1),
        "gpu_util_mean": round(float(np.mean(sm)), 1) if sm else None,
        "gpu_util_max": round(float(np.max(sm)), 1) if sm else None,
        "h2d_gbps": round((h2d_bytes / 1e9) / wall, 3),
        "gpu_peak_vram_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "vram_budget_mib": 7168,
        "corpus": "replicas of 7 IndEgo 480p clips, round-robin offsets (content repeated; decode cost realistic)",
    }
    rtf = wps * 10
    thr["extrapolation"] = {
        "rtf": round(rtf, 1),
        "gpus_for_10kh_in_2h": round(5000 / rtf, 1),
        "gpus_for_10kh_in_3h": round(3333 / rtf, 1),
        "note": ("LAPTOP GPU ceiling; linear scaling only to the NVDEC/PCIe/CPU "
                 "bottleneck stated in limits; NOT a datacenter claim."),
    }
    lad["rungs"].append({"rung": "R3_nvdec_pipeline",
                         "hours_per_hour": thr["R3_pipeline"]["hours_per_hour"],
                         "why": "NVDEC decode + scale_cuda at source + pinned H2D + "
                                "batched kernels: decode and compute overlap"})
    save(thr, lad)
    print(json.dumps({"R3_pipeline": thr["R3_pipeline"],
                      "extrapolation": thr["extrapolation"]}, indent=1))


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "--r0":
        mode_r0()
    elif m == "--holdA":
        mode_holdA()
    elif m == "--holdB":
        s = int(sys.argv[2]) if len(sys.argv) > 2 else 360
        mode_holdB(sustain_s=s)
    else:
        raise SystemExit("usage: throughput.py --r0|--holdA|--holdB [sustain_s]")
