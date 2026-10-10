"""R3c rung: NVDEC chain-B -> raw NV12 bytes -> GPU-side BT.601 -> fp16 kernels.

Orchestrator feeder fix: conversion moves off hot CPUs onto the GPU.
Run under lock (small: 48 windows ~ 8 video-min):
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock \
    python3 eval/gpu/holdc.py
Writes R3c_gpu_convert into eval/results/gpu/throughput.json + ladder rung.
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
GOLD = "/workspace/jadal_pod_bundle/labels/manifest_gold.json"
W, H, T = 160, 90, 100


def main(nwin=48, ndec=3):
    import numpy as np
    import torch
    from eval.gpu.decode import safe_offset
    from eval.gpu.telemetry import now_str
    from jadal.gpu.features import gpu_window_features_batch
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    gold = json.load(open(GOLD))["items"]
    seen, clips = set(), []
    for it in gold:
        if it["clip"] not in seen:
            seen.add(it["clip"])
            clips.append(it)

    raw_q = queue.Queue(maxsize=16)
    stop_ev = threading.Event()

    def raw_reader(path, off):
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
               "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
               "-ss", str(off), "-i", path,
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
        it = clips[i % len(clips)]
        threading.Thread(target=raw_reader,
                         args=(it["path"], safe_offset(it["path"], i, 120.0, step=97)),
                         daemon=True).start()
    t_start = now_str()
    t0 = time.perf_counter()
    h2d = done = 0
    while done < nwin:
        try:
            win = raw_q.get(timeout=10)
        except queue.Empty:
            break
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
            d, e = u - 128.0, v - 128.0
            r = 1.164 * c + 1.596 * e
            g = 1.164 * c - 0.391 * d - 0.813 * e
            b = 1.164 * c + 2.018 * d
            rgb = torch.stack([r, g, b], -1).clamp(0, 255) / 255.0
            luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            gpu_window_features_batch(luma.half(), rgb.half(), fps=10.0)
        done += 1
    wall = time.perf_counter() - t0
    stop_ev.set()
    torch.cuda.synchronize()
    wps = done / wall
    thr, lad = json.load(open(THR)), json.load(open(LAD))
    thr["R3c_gpu_convert"] = {
        "nwin": done, "video_min": round(done * 10 / 60, 1), "decoders": ndec,
        "windows_s": round(wps, 2), "hours_per_hour": round(wps * 10, 1),
        "video_s_per_wall_s": round(wps * 10, 1),
        "h2d_gbps": round((h2d / 1e9) / wall, 3), "wall_s": round(wall, 1),
        "gpu_peak_vram_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "t_start": t_start, "t_end": now_str(),
        "method": "NVDEC chain-B raw NV12 -> GPU BT.601 -> fp16 kernels, 1 win/batch",
    }
    lad["rungs"].append({"rung": "R3c_gpu_side_convert",
                         "hours_per_hour": thr["R3c_gpu_convert"]["hours_per_hour"],
                         "why": "feeder fix: NV12->RGB off hot CPUs onto GPU (R3a was "
                                "starved by float64-numpy conversion)"})
    json.dump(thr, open(THR, "w"), indent=1)
    json.dump(lad, open(LAD, "w"), indent=1)
    print(json.dumps(thr["R3c_gpu_convert"], indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 48)
