"""L1.01 atomic pipeline latency: perf_counter around every stage, one JSONL row/window.

Stages: io_probe | decode_cpu | decode_nvdec | features_pixel26(@10fps,@5fps)
       | eg2_embed(2fps) | mosaic | decision | write
Env: export OMP_NUM_THREADS=1; GPU use via tools/pod/gpu_run.sh wrapper.
Writes eval/results/exp_latency/windows.jsonl, clips_decode.jsonl, run_meta.json
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
OUTDIR = os.path.join(REPO, "eval", "results", "exp_latency")
os.makedirs(OUTDIR, exist_ok=True)

from eval.gpu.decode import clip_duration, decode_window, decode_window_nvdec  # noqa: E402
from jadal.gpu.features import FEATURE_NAMES, cpu_window_features  # noqa: E402

MANIFEST = "/workspace/jadal_pod_bundle/queues/manifest_scale.json"
EG2_DIR = ("/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2"
           "/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")
ENSEMBLE = "/workspace/jadal_pod_bundle/caches/ensemble/ensemble_v1.json"
W, H, WIN_S = 160, 90, 10.0
N_CLIPS_ATOMIC, WINS_PER_CLIP = 12, 2

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
        except Exception:
            pass
        time.sleep(0.5)


def now_ms(t0):
    return (time.perf_counter() - t0) * 1000.0


def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def subsample_idx(n, fps_native, fps_target, dur_s=WIN_S):
    want = int(round(fps_target * dur_s))
    if n <= 0:
        return np.zeros(0, dtype=int)
    return np.linspace(0, n - 1, min(want, n)).astype(int)


def build_mosaic(frames_rgb, times_s, wid):
    """6-tile evidence mosaic with per-tile timestamps (PIL)."""
    from PIL import Image, ImageDraw
    idx = np.linspace(0, len(frames_rgb) - 1, 6).astype(int)
    tiles = []
    for k, j in enumerate(idx):
        im = Image.fromarray((np.clip(frames_rgb[j], 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((im.width - 52, 2), f"{times_s[j]:06.2f}s", fill=(255, 255, 0))
        tiles.append(im)
    tw, th = tiles[0].size
    mos = Image.new("RGB", (tw * 3, th * 2))
    for k, t in enumerate(tiles):
        mos.paste(t, ((k % 3) * tw, (k // 3) * th))
    d = ImageDraw.Draw(mos)
    d.text((4, 4), wid, fill=(255, 255, 255))
    return mos


def load_head():
    try:
        e = json.load(open(ENSEMBLE))
        w = np.asarray(e.get("weights", e.get("coef", [0.0] * 26)), dtype=float)
        b = float(e.get("bias", e.get("intercept", 0.0)))
        if w.shape != (26,):
            raise ValueError("shape")
        return w, b, "ensemble_v1.json"
    except Exception as ex:
        return np.zeros(26), 0.0, f"zeros-fallback({str(ex)[:60]})"


def main():
    t_run0 = time.perf_counter()
    wall0 = time.time()
    man = json.load(open(MANIFEST))
    items = man["items"]
    print(f"manifest: n_windows={man['n_windows']} n_clips={man['n_clips']} "
          f"hours={man['total_hours']:.3f} window_s={man['window_s']}", flush=True)

    # ---- subset: 12 clips spread across factories, 2 windows each ----
    by_clip = {}
    for it in items:
        by_clip.setdefault(it["clip"], []).append(it)
    clips_sorted = sorted(by_clip)
    step = max(1, len(clips_sorted) // N_CLIPS_ATOMIC)
    atomic_clips = clips_sorted[::step][:N_CLIPS_ATOMIC]
    atomic = []
    for c in atomic_clips:
        atomic.extend(sorted(by_clip[c], key=lambda x: x["start_s"])[:WINS_PER_CLIP])
    facs = sorted({a["factory"] for a in atomic})
    print(f"atomic subset: {len(atomic)} windows x {len(atomic_clips)} clips x "
          f"{len(facs)} factories", flush=True)

    th = threading.Thread(target=smi_sampler, daemon=True)
    th.start()

    # ---- Phase A: per-clip cold probe + NVDEC decode (all 58 clips) ----
    clip_rows = []
    for c in clips_sorted:
        it = sorted(by_clip[c], key=lambda x: x["start_s"])[0]
        row = {"clip": c, "factory": it["factory"], "path": it["path"],
               "start_s": it["start_s"]}
        t0 = time.perf_counter()
        try:
            dur = clip_duration(it["path"])
            row["io_probe_cold_ms"] = now_ms(t0)
            row["clip_dur_s"] = dur
        except Exception as ex:
            row["io_probe_cold_ms"] = now_ms(t0)
            row["io_error"] = str(ex)[:120]
            clip_rows.append(row)
            continue
        t0 = time.perf_counter()
        try:
            rgb_n, fps_n = decode_window_nvdec(it["path"], it["start_s"], WIN_S, W, H)
            row["dec_nvdec_ms"] = now_ms(t0)
            row["dec_nvdec_frames"] = int(rgb_n.shape[0])
            row["dec_nvdec_fps"] = float(fps_n)
        except Exception as ex:
            row["dec_nvdec_ms"] = now_ms(t0)
            row["dec_nvdec_error"] = str(ex)[:200]
        clip_rows.append(row)
        if len(clip_rows) % 10 == 0:
            print(f"  phaseA {len(clip_rows)}/{len(clips_sorted)}", flush=True)

    # ---- EG2 model load (timed setup, amortized separately) ----
    t0 = time.perf_counter()
    import torch
    from sentence_transformers import SentenceTransformer
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    model = SentenceTransformer(EG2_DIR, device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16})
    model_load_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    print(f"EG2 model load: {model_load_s:.1f}s", flush=True)

    w_head, b_head, head_src = load_head()
    print(f"head: {head_src}", flush=True)

    # ---- Phase B: atomic per-window full pipeline ----
    from PIL import Image
    win_path = os.path.join(OUTDIR, "windows.jsonl")
    res_path = os.path.join(OUTDIR, "window_results.jsonl")
    fw = open(win_path, "w")
    fr = open(res_path, "w")
    for k, it in enumerate(atomic):
        row = {"id": it["id"], "clip": it["clip"], "factory": it["factory"],
               "path": it["path"], "start_s": it["start_s"]}
        # 1. io_probe (cached dict lookup after Phase A colds)
        t0 = time.perf_counter()
        try:
            dur = clip_duration(it["path"])
            row["io_probe_ms"] = now_ms(t0)
        except Exception as ex:
            row["io_probe_ms"] = now_ms(t0)
            row["io_error"] = str(ex)[:120]
        # 2a. decode CPU full window native fps
        t0 = time.perf_counter()
        try:
            rgb_c, fps_c = decode_window(it["path"], it["start_s"], WIN_S, W, H, "rgb24")
            row["decode_cpu_ms"] = now_ms(t0)
            row["decode_cpu_frames"] = int(rgb_c.shape[0])
        except Exception as ex:
            row["decode_cpu_ms"] = now_ms(t0)
            row["decode_cpu_error"] = str(ex)[:200]
            rgb_c, fps_c = None, 0.0
        # 2b. decode NVDEC full window
        t0 = time.perf_counter()
        try:
            rgb_n, fps_n = decode_window_nvdec(it["path"], it["start_s"], WIN_S, W, H)
            row["decode_nvdec_ms"] = now_ms(t0)
            row["decode_nvdec_frames"] = int(rgb_n.shape[0])
        except Exception as ex:
            row["decode_nvdec_ms"] = now_ms(t0)
            row["decode_nvdec_error"] = str(ex)[:200]
            rgb_n = None
        base = rgb_c if rgb_c is not None else rgb_n
        if base is None:
            fw.write(json.dumps(row) + "\n")
            fw.flush()
            continue
        luma = luma_of(base)
        fps_nat = float(fps_c) if fps_c else 30.0
        # 3. features @10fps (default) and @5fps (factory_1080p30)
        for tag, fpf in (("10", 10), ("5", 5)):
            ix = subsample_idx(base.shape[0], fps_nat, fpf)
            t0 = time.perf_counter()
            feats = cpu_window_features(luma[ix], base[ix], fps=min(fpf, 30.0))
            row[f"features_pixel26_fps{fpf}_ms"] = now_ms(t0)
            row[f"features_pixel26_fps{fpf}_n"] = int(len(ix))
            if tag == "10":
                feats10 = feats
        row["feature_names"] = FEATURE_NAMES
        # 4. EG2 @2fps batched (batch=20 = one window)
        ix2 = subsample_idx(base.shape[0], fps_nat, 2)
        pils = [Image.fromarray((np.clip(base[j], 0, 1) * 255).astype(np.uint8))
                for j in ix2]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        emb = model.encode(pils, batch_size=20, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
        torch.cuda.synchronize()
        row["eg2_embed_ms"] = now_ms(t0)
        row["eg2_frames"] = int(len(pils))
        row["eg2_emb_shape"] = list(emb.shape)
        row["eg2_vram_peak_mib"] = float(torch.cuda.max_memory_allocated() / 2**20)
        # 5. mosaic
        times = np.linspace(it["start_s"], it["start_s"] + WIN_S, base.shape[0])
        t0 = time.perf_counter()
        mos = build_mosaic(base, times, it["id"])
        row["mosaic_ms"] = now_ms(t0)
        row["mosaic_size"] = list(mos.size)
        # 6. decision: score + verdict + receipt hash
        t0 = time.perf_counter()
        x = np.array([float(feats10[nm]) for nm in FEATURE_NAMES])
        z = float(x @ w_head + b_head)
        p = float(1.0 / (1.0 + np.exp(-z)))
        verdict = "accept" if p >= 0.8 else ("reject" if p <= 0.2 else "review")
        canon = json.dumps({"id": it["id"], "p": round(p, 6), "verdict": verdict,
                            "profile": "default"},
                           sort_keys=True, separators=(",", ":"))
        receipt = hashlib.sha256(canon.encode()).hexdigest()
        row["decision_ms"] = now_ms(t0)
        row["decision_p"] = p
        row["decision_verdict"] = verdict
        # 7. write: result JSON append
        t0 = time.perf_counter()
        fr.write(json.dumps({"id": it["id"], "verdict": verdict, "p_bad": p,
                             "receipt_sha256": receipt}) + "\n")
        fr.flush()
        row["write_ms"] = now_ms(t0)
        fw.write(json.dumps(row) + "\n")
        fw.flush()
        print(f"  [{k+1}/{len(atomic)}] {it['id']} cpu={row.get('decode_cpu_ms',-1):.0f}ms "
              f"nv={row.get('decode_nvdec_ms',-1):.0f}ms f10={row['features_pixel26_fps10_ms']:.0f}ms "
              f"eg2={row['eg2_embed_ms']:.0f}ms -> {verdict}", flush=True)
    fw.close()
    fr.close()

    # ---- Phase C: kernels-only floor (features+mosaic on cached arrays) ----
    arr = np.random.RandomState(0).rand(100, H, W, 3)
    lum = luma_of(arr)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        cpu_window_features(lum, arr, fps=10.0)
        ts.append(now_ms(t0))
    mos_ts = []
    base_t = np.linspace(0, WIN_S, 100)
    for _ in range(3):
        t0 = time.perf_counter()
        build_mosaic(arr, base_t, "floor")
        mos_ts.append(now_ms(t0))
    floor = {"features_kernel_only_ms": ts, "mosaic_cached_ms": mos_ts}

    smi_stop.set()
    th.join(timeout=5)
    import resource
    meta = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": time.perf_counter() - t_run0,
        "manifest": {"n_windows": man["n_windows"], "n_clips": man["n_clips"],
                     "total_hours": man["total_hours"]},
        "atomic_windows": len(atomic), "atomic_clips": atomic_clips,
        "atomic_factories": facs,
        "eg2_model_load_s": model_load_s,
        "head_src": head_src,
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "floor": floor,
        "smi_samples": len(smi_log),
        "smi": smi_log,
        "device": "A40",
    }
    json.dump(clip_rows, open(os.path.join(OUTDIR, "clips_decode.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(OUTDIR, "run_meta.json"), "w"), indent=1)
    print(f"DONE wall={meta['wall_s']:.0f}s rss={meta['max_rss_gb']:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
