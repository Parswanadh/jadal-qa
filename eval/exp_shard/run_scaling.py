#!/usr/bin/env python3
"""S1.01 — parallel scaling sweep: K isolated shard workers on this pod.

For each K in ``--ks`` it builds the deterministic shard plan, launches K
independent ``shard.py`` processes concurrently (each <=1 core), samples CPU /
GPU / NVDEC / VRAM, aggregates the shards, and records the shard-invariance
digest. The whole sweep holds ONE GPU slot (all shards share the single A40):

    bash tools/pod/gpu_run.sh python3 eval/exp_shard/run_scaling.py \
        --ks 1,2,4,7 --subset smallest6

Ceiling attribution:
  * CPU-quota-bound  -> cpu_cores saturates ~7.65 and win/s keeps rising with K
  * NVDEC-engine-bound -> gpu_dec saturates ~100% and win/s plateaus well below
                          the CPU ceiling
  * IO / memcpy-bound -> low cpu_cores AND low gpu_dec
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_shard"))

import common  # noqa: E402

SHARD_PY = os.path.join(HERE, "shard.py")
AGG_PY = os.path.join(HERE, "aggregate.py")
CGROUP_QUOTA = 7.65


def _cpu_s():
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (s.ru_utime + s.ru_stime) + (c.ru_utime + c.ru_stime)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.s = []

    def _smi(self):
        try:
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            mem, sm = [x.strip() for x in p.stdout.strip().split(",")]
            return float(mem), float(sm), None
        except Exception:  # noqa: BLE001
            return None, None, None

    def _dmon(self):
        try:
            p = subprocess.run(["nvidia-smi", "dmon", "-s", "u", "-c", "1"],
                               capture_output=True, text=True, timeout=15)
            for ln in p.stdout.splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) >= 7:
                    # gpu sm mem enc dec jpg ofa
                    return float(parts[4])  # dec %
        except Exception:  # noqa: BLE001
            pass
        return None

    def run(self):
        while not self.stop.is_set():
            mem, sm, _ = self._smi()
            dec = self._dmon()
            if mem is not None or dec is not None:
                self.s.append({"t": time.time(), "vram_mib": mem, "gpu_sm": sm,
                               "gpu_dec": dec})
            time.sleep(0.2)


def _stat(samples, key):
    v = [s[key] for s in samples if s.get(key) is not None]
    if not v:
        return None, None
    return round(float(np.mean(v)), 2), round(float(np.max(v)), 2)


def run_one(k, items, args, root):
    plan = common.make_plan(items, k)
    planf = common.write_plan(plan, root)
    kdir = os.path.join(root, f"K{k}")
    if os.path.isdir(kdir):
        shutil.rmtree(kdir)
    os.makedirs(kdir, exist_ok=True)
    procs, metas = [], []
    for i in range(k):
        out = os.path.join(kdir, f"shard_{i}.jsonl")
        sd = os.path.join(kdir, "state", f"shard_{i}")
        cmd = [sys.executable, SHARD_PY, "--shard-index", str(i), "--shards", str(k),
               "--subset", args.subset, "--plan", planf, "--out", out,
               "--state-dir", sd]
        if args.manifest != common.MANIFEST:
            cmd += ["--manifest", args.manifest]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE, text=True))
    samp = Sampler()
    samp.start()
    cpu0 = _cpu_s()
    t0 = time.perf_counter()
    rc = [p.wait() for p in procs]
    wall = time.perf_counter() - t0
    cpu_s = _cpu_s() - cpu0
    samp.stop.set()
    samp.join(timeout=5)
    errs = [p.stderr.read()[-300:] for p in procs if p.returncode != 0]
    if any(rc):
        print(f"K={k} child rc={rc} errs={errs}", flush=True)

    for i in range(k):
        mp = os.path.join(kdir, "state", f"shard_{i}", f"shard_{i}.meta.json")
        if os.path.exists(mp):
            metas.append(json.load(open(mp)))

    # aggregate + digest
    full = os.path.join(kdir, f"combined_K{k}.jsonl")
    dg = os.path.join(kdir, f"combined_K{k}.digest.json")
    subprocess.run([sys.executable, AGG_PY, "--shard-dir", kdir, "--shards", str(k),
                    "--out", full, "--digest-out", dg, "--subset", args.subset,
                    "--manifest", args.manifest], check=True,
                   stdout=subprocess.DEVNULL)
    doc = json.load(open(dg))
    n = doc["n_windows"]
    row = {
        "K": k,
        "n_windows": n,
        "wall_s": round(wall, 2),
        "windows_per_s": round(n / max(wall, 1e-9), 4),
        "footage_h_per_wall_h": round((n * 10.0 / 3600.0) / (wall / 3600.0), 3),
        "cpu_s": round(cpu_s, 2),
        "cpu_cores_used": round(cpu_s / max(wall, 1e-9), 3),
        "cpu_quota": CGROUP_QUOTA,
        "gpu_sm_pct_mean": _stat(samp.s, "gpu_sm")[0],
        "gpu_sm_pct_max": _stat(samp.s, "gpu_sm")[1],
        "gpu_dec_pct_mean": _stat(samp.s, "gpu_dec")[0],
        "gpu_dec_pct_max": _stat(samp.s, "gpu_dec")[1],
        "vram_used_mib_mean": _stat(samp.s, "vram_mib")[0],
        "vram_used_mib_max": _stat(samp.s, "vram_mib")[1],
        "per_shard_cpu_cores_mean": round(
            float(np.mean([m["cpu_cores"] for m in metas])), 3) if metas else None,
        "per_shard_rss_mb_max": max((m["max_rss_mb"] for m in metas), default=None),
        "shard_digest": doc["combined_digest"],
        "per_shard_integrity_ok": doc.get("per_shard_integrity_ok"),
        "n_errors": sum(m.get("n_errors", 0) for m in metas),
        "plan_loads_windows": plan["loads_windows"],
        "n_samples": len(samp.s),
    }
    print(f"K={k}: {row['windows_per_s']} win/s  "
          f"{row['footage_h_per_wall_h']} h/h  cpu={row['cpu_cores_used']} cores  "
          f"dec={row['gpu_dec_pct_mean']}% sm={row['gpu_sm_pct_mean']}%  "
          f"vram={row['vram_used_mib_max']}MiB  digest={row['shard_digest'][:12]}",
          flush=True)
    return row, kdir, doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,4,7")
    ap.add_argument("--subset", default="smallest6")
    ap.add_argument("--manifest", default=common.MANIFEST)
    ap.add_argument("--root", default=common.SHARD_DIR)
    ap.add_argument("--out", default=os.path.join(common.SHARD_ROOT, "S1.01.json"))
    ap.add_argument("--vram-json", default=os.path.join(common.SHARD_ROOT,
                                                        "vram_probe.json"))
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    _, items = common.load_items(args.manifest)
    items = common.select_items(items, args.subset)
    n_clips = len({it["clip"] for it in items})

    rows, digests, keep_dirs = [], {}, {}
    for k in [int(x) for x in args.ks.split(",")]:
        row, kdir, doc = run_one(k, items, args, args.root)
        rows.append(row)
        digests[str(k)] = doc["combined_digest"]
        keep_dirs[k] = kdir

    # shard-invariance: every K must yield the SAME combined digest
    inv = len(set(digests.values())) == 1
    # committed split = the largest K that still gives every shard >=1 clip
    cand = [k for k in keep_dirs if k <= n_clips]
    base_k = max(cand) if cand else max(keep_dirs)
    # expose the chosen split as shards/shard_<i>.jsonl (the deliverable layout)
    for fn in os.listdir(keep_dirs[base_k]):
        if fn.startswith("shard_") and fn.endswith(".jsonl"):
            shutil.copy(os.path.join(keep_dirs[base_k], fn),
                        os.path.join(args.root, fn))
    shutil.copy(os.path.join(keep_dirs[base_k], f"combined_K{base_k}.digest.json"),
                os.path.join(args.root, "combined_digest.json"))

    vram = json.load(open(args.vram_json)) if os.path.exists(args.vram_json) else None
    doc = {
        "id": "S1.01",
        "title": "Sharded isolated workers: parallel scaling ceiling + RAM/VRAM config",
        "device": "NVIDIA A40 (1x NVDEC engine)",
        "cgroup_cpu_quota_cores": CGROUP_QUOTA,
        "nproc": os.cpu_count(),
        "subset": args.subset, "n_clips": n_clips, "n_windows": len(items),
        "footage_h": round(len(items) * 10.0 / 3600.0, 3),
        "optimizations": {"pixel_fps": 3.33, "eg2_fps": 1, "native_fps": 0.8,
                          "early_exit_s": 4, "skip_mosaic": True,
                          "omp_num_threads": 1},
        "ks": [int(x) for x in args.ks.split(",")],
        "scaling": rows,
        "shard_invariance": {"per_K_combined_digest": digests, "all_equal": inv,
                             "reference_digest": digests[str(sorted(keep_dirs)[0])]},
        "committed_split_K": base_k,
        "vram": vram,
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(doc, open(args.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({"subset": args.subset, "n_windows": len(items),
                      "scaling": [{k: r[k] for k in
                                   ("K", "windows_per_s", "footage_h_per_wall_h",
                                    "cpu_cores_used", "gpu_dec_pct_mean",
                                    "gpu_sm_pct_mean", "vram_used_mib_max")}
                                  for r in rows],
                      "shard_invariance_all_equal": inv,
                      "committed_split_K": base_k}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
