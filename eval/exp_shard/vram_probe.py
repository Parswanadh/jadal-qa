#!/usr/bin/env python3
"""S1.01 — VRAM as a first-class parallel resource.

Three measurements on the pod's single A40 (46 GB):

1. **residency**  — idle -> EG2 model loaded -> per-worker VRAM footprint.
2. **embed-batch** — EG2 embedding peak VRAM / throughput vs ``batch_size``
   (the GPU *batch* axis B). Frames/window at the adopted config is ~4, so this
   probes whether a bigger batch can buy anything.
3. **embed-concurrent** — B independent embed processes (the GPU *streams*
   axis); aggregate throughput + total VRAM -> the VRAM saturation point and the
   per-worker VRAM budget for a multi-pod fleet.

All numbers are VRAM deltas sampled with nvidia-smi around the work, because the
GPU is shared with other pod jobs (absolute ``memory.used`` includes them).

    bash tools/pod/gpu_run.sh python3 eval/exp_shard/vram_probe.py \
        --out eval/results/exp_shard/vram_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_shard"))

import common  # noqa: E402
import run_e8 as e8  # noqa: E402

EGW, EGH = 512, 288


def vram_used_mib():
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        m, u = [x.strip() for x in p.stdout.strip().split(",")]
        return float(m), float(u)
    except Exception:  # noqa: BLE001
        return None, None


def _frames(n):
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, size=(n, EGH, EGW, 3), dtype=np.uint8)


def _embed(model, frames, batch):
    from PIL import Image
    pils = [Image.fromarray(f) for f in frames]
    t = time.perf_counter()
    E = model.encode(pils, batch_size=batch, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=True)
    return (time.perf_counter() - t) * 1000.0, int(np.asarray(E).shape[0])


def worker(args):
    """One embed worker: load model, embed N frames, print VRAM/throughput."""
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    e8.init_worker()
    if os.path.isdir(run_e8_pod_eg2()):
        e8._CFG["views"]["eg2"]["embed_model_dir"] = run_e8_pod_eg2()
    base_mib, _ = vram_used_mib()
    torch.cuda.reset_peak_memory_stats()
    _, model = e8._get_model(e8._CFG)
    after_load_mib, _ = vram_used_mib()
    frames = _frames(args.n_images)
    ms, n = _embed(model, frames, args.batch)
    peak_torch = float(torch.cuda.max_memory_allocated() / (1 << 20))
    peak_mib, _ = vram_used_mib()
    print("WORKER " + json.dumps({
        "pid": os.getpid(), "n_images": n, "batch": args.batch, "ms": round(ms, 2),
        "images_per_s": round(n / max(ms / 1000.0, 1e-9), 2),
        "torch_peak_mib": round(peak_torch, 1),
        "smi_delta_load_mib": None if None in (base_mib, after_load_mib)
        else round(after_load_mib - base_mib, 1),
        "smi_delta_peak_mib": None if None in (base_mib, peak_mib)
        else round(peak_mib - base_mib, 1),
    }), flush=True)


def run_e8_pod_eg2():
    return (os.path.join(os.environ.get("JADAL_POD_ROOT", ""), '.cache/huggingface/hub/models--google--embeddinggemma-2')
            "/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")


def run_worker_subproc(n_images, batch, timeout=300):
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--mode", "worker",
                        "--n-images", str(n_images), "--batch", str(batch)],
                       capture_output=True, text=True, timeout=timeout)
    for ln in p.stdout.splitlines():
        if ln.startswith("WORKER "):
            return json.loads(ln[len("WORKER "):])
    return {"error": (p.stderr or p.stdout)[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sweep", "worker"), default="sweep")
    ap.add_argument("--n-images", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--batches", default="4,8,16,32,64,128")
    ap.add_argument("--out", default=os.path.join(common.SHARD_ROOT, "vram_probe.json"))
    args = ap.parse_args()

    if args.mode == "worker":
        return worker(args)

    import multiprocessing as mp
    out = {"device": "NVIDIA A40", "total_vram_mib": 46068,
           "note": "VRAM deltas sampled via nvidia-smi (GPU shared with other pod jobs); "
                   "torch_peak is per-process allocator peak",
           "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    base_mib, base_util = vram_used_mib()
    out["baseline"] = {"vram_used_mib": base_mib, "gpu_util_pct": base_util}

    # 1) residency: one worker, small batch
    w0 = run_worker_subproc(args.n_images, 32)
    out["residency"] = w0

    # 2) embed-batch sweep (single process, sequential)
    batch_rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        r = run_worker_subproc(args.n_images, b)
        batch_rows.append({"batch": b, **{k: r.get(k) for k in
                           ("ms", "images_per_s", "torch_peak_mib",
                            "smi_delta_peak_mib")}})
    out["embed_batch_sweep"] = batch_rows

    # 3) concurrent embed streams B -> aggregate throughput + VRAM
    conc_rows = []
    for B in [int(x) for x in args.concurrency.split(",")]:
        before, _ = vram_used_mib()
        procs = [subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--mode", "worker",
             "--n-images", str(args.n_images), "--batch", "32"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for _ in range(B)]
        peak = before or 0.0
        utils = []
        t0 = time.perf_counter()
        while any(p.poll() is None for p in procs):
            m, u = vram_used_mib()
            if m:
                peak = max(peak, m)
            if u is not None:
                utils.append(u)
            time.sleep(0.5)
        wall = time.perf_counter() - t0
        results = []
        for p in procs:
            for ln in p.stdout.read().splitlines():
                if ln.startswith("WORKER "):
                    results.append(json.loads(ln[len("WORKER "):]))
        total_img = sum(r["n_images"] for r in results)
        conc_rows.append({
            "B": B, "wall_s": round(wall, 2),
            "images_per_s": round(total_img / max(wall, 1e-9), 2),
            "per_worker_images_per_s": round(
                total_img / max(wall, 1e-9) / B, 2),
            "vram_peak_used_mib": peak,
            "vram_delta_vs_base_mib": None if before is None
            else round(peak - before, 1),
            "torch_peak_mib_max": max((r.get("torch_peak_mib") or 0)
                                      for r in results) if results else None,
            "gpu_util_pct_mean": round(float(np.mean(utils)), 1) if utils else None,
            "gpu_util_pct_max": max(utils) if utils else None,
            "n_ok": len(results), "errors": [r for r in results if "error" in r][:3],
        })
    out["embed_concurrent_sweep"] = conc_rows

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
