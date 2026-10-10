"""M1 microbenchmarks: GPU memcpy bandwidth, NVDEC decode throughput, kernel roofline.

Regenerates eval/results/gpu/microbench.json. Run:
  flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock \
    /home/parshu/envs/jadal/bin/python eval/gpu/microbench.py
Hardware label: RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU.
"""
import json
import os
import subprocess
import sys
import threading
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "gpu", "microbench.json")
GPU = "cuda:0"

CLIPS = [
    "/home/parshu/projects/awing/data/indego/1_Assembly/mp4_480/User_15_410_0411_2_480.mp4",
]

DEVICE = "RTX 4070 Laptop"


def cuda_sync():
    torch.cuda.synchronize()


def bench_memcpy():
    out = {}
    nbytes = 256 * 1024 * 1024  # 256 MiB fp32
    n = nbytes // 4
    reps = 20
    # device-to-device
    a = torch.empty(n, dtype=torch.float32, device=GPU)
    b = torch.empty(n, dtype=torch.float32, device=GPU)
    cuda_sync()
    # warmup
    for _ in range(5):
        b.copy_(a)
    cuda_sync()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(reps):
        b.copy_(a)
    ev1.record()
    cuda_sync()
    ms = ev0.elapsed_time(ev1) / reps
    out["d2d_gbps"] = round((nbytes / 1e9) / (ms / 1e3), 2)
    out["d2d_ms_per_256mib"] = round(ms, 3)
    # host-to-device pinned
    hp = torch.empty(n, dtype=torch.float32).pin_memory()
    hp.normal_()
    cuda_sync()
    ev0.record()
    for _ in range(reps):
        b.copy_(hp, non_blocking=True)
    ev1.record()
    cuda_sync()
    ms = ev0.elapsed_time(ev1) / reps
    out["h2d_pinned_gbps"] = round((nbytes / 1e9) / (ms / 1e3), 2)
    # host-to-device pageable
    h = torch.empty(n, dtype=torch.float32)
    h.normal_()
    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        b.copy_(h)
    cuda_sync()
    s = (time.perf_counter() - t0) / reps
    out["h2d_pageable_gbps"] = round((nbytes / 1e9) / s, 2)
    # device-to-host pinned
    t0 = time.perf_counter()
    for _ in range(reps):
        hp.copy_(b, non_blocking=True)
    cuda_sync()
    s = (time.perf_counter() - t0) / reps
    out["d2h_pinned_gbps"] = round((nbytes / 1e9) / s, 2)
    out["nbytes"] = nbytes
    out["reps"] = reps
    return out


def gpu_query(args):
    try:
        r = subprocess.run(
            ["nvidia-smi"] + args, capture_output=True, text=True, timeout=15
        )
        return r.stdout.strip()
    except Exception:
        return ""


class DecSampler(threading.Thread):
    """Poll decoder utilisation during a decode run."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._run = True

    def run(self):
        while self._run:
            q = gpu_query(
                ["--query-gpu=utilization.gpu,utilization.decoder,utilization.memory",
                 "--format=csv,noheader,nounits"]
            )
            if q:
                self.samples.append(q)
            time.sleep(0.5)

    def stop(self):
        self._run = False


def run_decode(cmd, expect_bytes, timeout_s=180):
    """Run one ffmpeg, drain pipe, return wall seconds and bytes read."""
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL)
    total = 0
    try:
        while True:
            if time.perf_counter() - t0 > timeout_s:
                p.kill()
                return None, total, "timeout"
            chunk = p.stdout.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
        p.wait(timeout=30)
        rc = p.returncode
        err = p.stderr.read()[-2000:].decode("utf-8", "replace") if rc != 0 else ""
    except Exception as e:
        try:
            p.kill()
        except Exception:
            pass
        return None, total, str(e)
    wall = time.perf_counter() - t0
    return wall, total, (rc if rc == 0 else f"rc={rc} {err[-300:]}")


def bench_decode():
    out = {"configs": []}
    clips = [c for c in CLIPS if os.path.exists(c)]
    if not clips:
        out["error"] = "no clips found"
        return out
    dur = 15  # seconds of video per process (keeps one lock hold short)
    for scale in ["160:90", "320:180"]:
        w, h = (int(x) for x in scale.split(":"))
        frame_bytes = w * h  # gray
        # GPU path
        for conc in [1, 2, 4, 8]:
            sampler = DecSampler()
            sampler.start()
            t0 = time.perf_counter()
            procs = []
            for i in range(conc):
                off = (i * 60) % 600
                cmd = [
                    "ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
                    "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                    "-ss", str(off), "-t", str(dur), "-i", clips[i % len(clips)],
                    "-vf", f"scale_cuda={scale},hwdownload,format=gray",
                    "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
                ]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
                procs.append(p)
            totals = []
            rcs = []
            for p in procs:
                tot = 0
                ok = True
                while True:
                    if time.perf_counter() - t0 > 240:
                        p.kill()
                        ok = False
                        break
                    ch = p.stdout.read(1 << 20)
                    if not ch:
                        break
                    tot += len(ch)
                p.wait()
                totals.append(tot)
                rcs.append(p.returncode if ok else "timeout")
            wall = time.perf_counter() - t0
            sampler.stop()
            sampler.join(timeout=5)
            frames = sum(totals) / frame_bytes
            fps = frames / wall
            rtf = (dur * conc) / wall
            # parse dec util samples
            decvals = []
            for s in sampler.samples:
                try:
                    parts = [x.strip() for x in s.split(",")]
                    decvals.append(float(parts[1].split()[0]))
                except Exception:
                    pass
            out["configs"].append({
                "decoder": "nvdec", "scale": scale, "concurrency": conc,
                "wall_s": round(wall, 2), "bytes_per_proc": totals,
                "returncodes": rcs, "frames_total": round(frames, 1),
                "frames_per_s": round(fps, 1),
                "video_s_per_wall_s": round(rtf, 2),
                "dec_util_max": max(decvals) if decvals else None,
                "dec_util_mean": round(sum(decvals) / len(decvals), 1) if decvals else None,
            })
        # CPU path (conc 1 and 4 only, to save time)
        for conc in [1, 4]:
            t0 = time.perf_counter()
            procs = []
            for i in range(conc):
                off = (i * 60) % 600
                cmd = [
                    "ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
                    "-ss", str(off), "-t", str(dur), "-i", clips[i % len(clips)],
                    "-vf", f"scale={scale},format=gray",
                    "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
                ]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
                procs.append(p)
            totals = []
            for p in procs:
                tot = 0
                while True:
                    if time.perf_counter() - t0 > 240:
                        p.kill()
                        break
                    ch = p.stdout.read(1 << 20)
                    if not ch:
                        break
                    tot += len(ch)
                p.wait()
                totals.append(tot)
            wall = time.perf_counter() - t0
            frames = sum(totals) / frame_bytes
            out["configs"].append({
                "decoder": "cpu", "scale": scale, "concurrency": conc,
                "wall_s": round(wall, 2),
                "frames_total": round(frames, 1),
                "frames_per_s": round(frames / wall, 1),
                "video_s_per_wall_s": round((dur * conc) / wall, 2),
            })
    return out


def bench_kernels():
    """Per-op roofline on random tensors: windows/s at 160x90x100 fp16."""
    out = {}
    H, W, T = 90, 160, 100
    torch.manual_seed(0)
    for B in [8, 32]:
        res = {}
        x = torch.randn(B, T, H, W, dtype=torch.float16, device=GPU)  # luma windows
        cuda_sync()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        reps = 10

        def timed(fn):
            fn()  # warmup
            cuda_sync()
            ev0.record()
            for _ in range(reps):
                fn()
            ev1.record()
            cuda_sync()
            return ev0.elapsed_time(ev1) / reps / B  # ms per window

        # temporal FFT + band energies
        def f_temp():
            F = torch.fft.rfft(x.float(), dim=1)  # (B, T/2+1, H, W)
            P = F.abs().pow(2)
            b1 = P[:, 0:4].sum(dim=1)   # ~0-0.3Hz @10fps (df=0.1Hz)
            b2 = P[:, 3:21].sum(dim=1)  # 0.3-2Hz
            b3 = P[:, 20:41].sum(dim=1)  # 2-4Hz (nyquist 5Hz)
            return b1.mean() + b2.mean() + b3.mean()
        res["temporal_fft_ms_per_window"] = round(timed(f_temp), 3)

        # spatial 2D spectrum per frame (mean over T first to cut cost, + full variant)
        xm = x.float().mean(dim=1)  # (B,H,W)
        def f_spat():
            F = torch.fft.rfft2(xm)
            P = F.abs().pow(2)
            return P.sum()
        res["spatial_fft_ms_per_window"] = round(timed(f_spat), 3)

        # RGB PCA + inverse residual: (B, N, 3)
        rgb = torch.randn(B, H * W * 10, 3, dtype=torch.float16, device=GPU)
        def f_pca():
            mu = rgb.float().mean(dim=1, keepdim=True)
            c = rgb.float() - mu
            cov = (c.transpose(1, 2) @ c) / c.shape[1]
            ev, V = torch.linalg.eigh(cov)
            proj = c @ V[:, :, 1:]  # top-2
            rec = proj @ V[:, :, 1:].transpose(1, 2)
            resid = ((c - rec).pow(2).mean(dim=(1, 2)))
            return resid.sum()
        res["pca_residual_ms_per_window"] = round(timed(f_pca), 3)

        # luma/chroma stats
        def f_stats():
            m = x.float().mean(dim=(1, 2, 3))
            q = torch.quantile(x.float().reshape(B, -1), torch.tensor([0.05, 0.5, 0.95], device=GPU), dim=1)
            sat = (x.float() > 0.98).float().mean(dim=(1, 2, 3))
            return m.sum() + q.sum() + sat.sum()
        res["stats_ms_per_window"] = round(timed(f_stats), 3)

        # frame differences + phase correlation global shift
        def f_motion():
            d = (x.float()[:, 1:] - x.float()[:, :-1]).abs().mean(dim=(2, 3))  # (B,T-1)
            F0 = torch.fft.rfft2(x.float()[:, :-1:10])
            F1 = torch.fft.rfft2(x.float()[:, 1::10])
            R = F0 * F1.conj()
            R = R / (R.abs() + 1e-6)
            cc = torch.fft.irfft2(R)
            shift = cc.reshape(B, -1).argmax(dim=1)
            return d.sum() + shift.sum()
        res["motion_ms_per_window"] = round(timed(f_motion), 3)

        # YCrCb skin mask (from RGB windows)
        rgbw = torch.randn(B, H, W, 3, dtype=torch.float16, device=GPU)
        def f_skin():
            r, g, b = rgbw[..., 0].float(), rgbw[..., 1].float(), rgbw[..., 2].float()
            cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
            cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
            mask = (cr > 0.55) & (cr < 0.72) & (cb > 0.38) & (cb < 0.52)
            return mask.float().mean(dim=(1, 2)).sum()
        res["skin_ms_per_window"] = round(timed(f_skin), 3)

        # 8x8 block DCT energy via precomputed matrix
        N = 8
        dctm = torch.zeros(N, N, device=GPU)
        for k in range(N):
            for nn in range(N):
                dctm[k, nn] = (2.0 / N) ** 0.5 * (0.70710678 if k == 0 else 1.0) * \
                    torch.cos(torch.tensor(3.14159265 * k * (2 * nn + 1) / (2 * N))).item()
        xb = x.float()[:, :1]
        Hc, Wc = (H // N) * N, (W // N) * N
        xb = xb[:, :, :Hc, :Wc].reshape(B, Hc // N, N, Wc // N, N).permute(0, 1, 3, 2, 4).reshape(-1, N, N)
        def f_dct():
            C = dctm @ xb @ dctm.T
            e = (C.pow(2).sum(dim=(1, 2)))
            return e.sum()
        res["dct_ms_per_window"] = round(timed(f_dct), 3)
        res["windows_per_s"] = round(1000.0 / sum(v for k, v in res.items() if k.endswith("_ms_per_window")), 1)
        out[f"batch_{B}"] = res
    out["window"] = {"H": H, "W": W, "T": T, "fps": 10, "dtype": "fp16-in/fp32-math"}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="comma list of: memcpy,decode,kernels (default all)")
    only = set(ap.parse_args().only.split(","))
    want = lambda k: "all" in only or k in only
    assert torch.cuda.is_available(), "no GPU"
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT))
        except Exception:
            res = {}
    res.update({
        "device": DEVICE,
        "torch": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "regenerate": "flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock "
                      "/home/parshu/envs/jadal/bin/python eval/gpu/microbench.py [--only memcpy,kernels|decode]",
    })
    t0 = time.perf_counter()
    if want("memcpy"):
        res["memcpy"] = bench_memcpy()
    if want("decode"):
        res["decode"] = bench_decode()
    if want("kernels"):
        res["kernels"] = bench_kernels()
    res["peak_vram_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1)[:3000])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
