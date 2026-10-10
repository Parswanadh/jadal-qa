"""LB1-eg2-repro: re-embed a sample of IndEgo gold windows from the SOURCE VIDEO
and compare against the cached `segments.npz` (frames10k/human_segments).

Recipe is copied EXACTLY from eval/frames10k/embed_segments.py (the cache producer,
as recorded in configs/ensemble_v1.json["embed_command"], which passes NO
--gpu-decode flag -> the CPU path cv2 + PIL BILINEAR thumbnail is the documented
producer):

  times = np.arange(start_s, min(end_s, dur), 0.5)          # 2 fps, 20-21 frames/10 s
  per t: cap.set(cv2.CAP_PROP_POS_MSEC, t*1000); cap.read()  # skipped if read fails
         cv2.cvtColor(BGR2RGB) -> Image.fromarray().convert("RGB")
         img.thumbnail((512, 512), Image.BILINEAR)
  model.encode(pils, batch_size=32, normalize_embeddings=True,
               processing_kwargs={"image": {"max_soft_tokens": 140}})
  frames stored fp16; window vector = mean over frames of the L2-normalised frame
  embeddings, then L2-normalised (pooling from eval/exp_fps/run_eg2.py window_mean).

NO prompt/prefix is used anywhere in the cache-producing path: EG2 is used as a
pure image encoder via SentenceTransformer.encode(list_of_PIL).

Two phases so the GPU slot is only held for the encode:
  phase decode (CPU, no GPU lock) -> scratch .npz of uint8 RGB thumbs
  phase encode (call under flock) -> per-window cosine vs cache + results JSON

Run:
  OMP_NUM_THREADS=2 python3 eval/frames10k/verify_cache_reproduction.py --phase decode
  flock -w 3600 /workspace/state/gpu.slot2.lock \
    env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 python3 \
    eval/frames10k/verify_cache_reproduction.py --phase encode
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CACHE = "/workspace/jadal_pod_bundle/caches/frames10k/segments.npz"
MANIFEST = "/workspace/jadal_pod_bundle/labels/manifest_gold.json"
EG2_DIR = ("/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2/"
           "snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")
PREPROCESS_MAX_SIDE = 512
STEP_S = 0.5
MAX_SOFT_TOKENS = 140
BATCH_SIZE = 32
N_SAMPLE = 10
SCRATCH = Path("/workspace/scratch_b_verify_eg2")
WT = Path(__file__).resolve().parents[2]
OUT_JSON = WT / "eval" / "results" / "pod_verify" / "eg2_embedding_reproduction.json"

SAMPLING_RULE = (
    "deterministic clip-stratified: all 7 clips get the median window of their "
    "contiguous block in manifest_gold order; the 3 largest blocks (by window count) "
    "get one extra window at the first quartile of their block; 10 windows total"
)


def select_windows(clips: list[str]) -> list[int]:
    order: list[str] = []
    blocks: dict[str, list[int]] = {}
    for i, c in enumerate(clips):
        if c not in blocks:
            blocks[c] = []
            order.append(c)
        blocks[c].append(i)
    idx = [blocks[c][(len(blocks[c]) - 1) // 2] for c in order]
    biggest = sorted(order, key=lambda c: -len(blocks[c]))[:N_SAMPLE - len(order)]
    for c in biggest:
        b = blocks[c]
        q = b[len(b) // 4]
        if q not in idx:
            idx.append(q)
    return sorted(idx)


def sample_segment(path: str, start: float, end: float):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return [], {"fps": None, "dur": None, "times": []}
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps else 0.0
        times = np.arange(start, min(end, dur), STEP_S)
        out, used = [], []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).convert("RGB")
            img.thumbnail((PREPROCESS_MAX_SIDE, PREPROCESS_MAX_SIDE), Image.BILINEAR)
            out.append(np.asarray(img, dtype=np.uint8))
            used.append(float(t))
        return out, {"fps": float(fps), "dur": float(dur),
                     "times": used, "n_planned": int(len(times))}
    finally:
        cap.release()


def sha256_file(p: Path, cap_bytes: int = 1 << 26) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        n = 0
        while n < cap_bytes:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest()


def phase_decode(sel: list[int], man: list[dict]) -> None:
    import cv2
    from PIL import Image

    SCRATCH.mkdir(parents=True, exist_ok=True)
    thumbs, counts, meta = [], [], []
    for k in sel:
        it = man[k]
        t0 = time.time()
        arrs, info = sample_segment(it["path"], it["start_s"], it["end_s"])
        thumbs.extend(arrs)
        counts.append(len(arrs))
        info.update({"k": k, "id": it["id"], "clip": it["clip"],
                     "path": it["path"], "start_s": it["start_s"], "end_s": it["end_s"],
                     "decode_s": round(time.time() - t0, 2)})
        meta.append(info)
        print(f"  g{k:03d} {it['id']} {it['clip']:<10} n={len(arrs):>3} "
              f"planned={info['n_planned']:>3} dur={info['dur']:.2f} "
              f"shape={arrs[0].shape if arrs else None} ({info['decode_s']}s)", flush=True)
    np.savez_compressed(SCRATCH / "thumbs.npz",
                        thumbs=np.stack(thumbs).astype(np.uint8),
                        counts=np.asarray(counts, dtype=np.int16),
                        sel=np.asarray(sel, dtype=np.int16))
    (SCRATCH / "decode_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"decoded {len(thumbs)} frames for {len(sel)} windows -> {SCRATCH/'thumbs.npz'}",
          flush=True)
    print("cv2", cv2.__version__, "PIL", Image.__version__, flush=True)


def phase_encode(sel: list[int], man: list[dict]) -> dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    z = np.load(CACHE, allow_pickle=True)
    F = z["F"]
    cache_counts = z["counts"].astype(int)
    cache_ids = [str(x) for x in z["ids"]]
    t = np.load(SCRATCH / "thumbs.npz", allow_pickle=True)
    thumbs, my_counts, sel_d = t["thumbs"], t["counts"].astype(int), t["sel"].astype(int).tolist()
    assert sel_d == sel, "sample indices drifted between phases"

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), "requires CUDA"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = SentenceTransformer(EG2_DIR, model_kwargs={"torch_dtype": dtype},
                                trust_remote_code=True)
    model.eval()
    load_s = time.time() - t0
    print(f"loaded {EG2_DIR.rsplit('/',1)[-1]} in {load_s:.1f}s dtype={dtype}", flush=True)

    pils = [Image.fromarray(a, mode="RGB") for a in thumbs]
    t0 = time.time()
    with torch.inference_mode():
        E = model.encode(pils, batch_size=BATCH_SIZE, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True,
                         processing_kwargs={"image": {"max_soft_tokens": MAX_SOFT_TOKENS}})
    enc_s = time.time() - t0
    vram_peak = float(torch.cuda.max_memory_allocated() / 1e9)
    E = np.asarray(E, dtype=np.float16)
    np.save(SCRATCH / "my_frame_embeddings.fp16.npy", E)
    print(f"encoded {E.shape[0]} frames in {enc_s:.1f}s VRAM peak {vram_peak:.2f} GB", flush=True)

    rows, o = [], 0
    frame_cos_all, wcos = [], []
    for j, k in enumerate(sel):
        n = int(my_counts[j])
        mine_f = E[o:o + n].astype(np.float32)
        o += n
        nc = int(cache_counts[k])
        cache_f = F[k, :nc].astype(np.float32)
        cache_v = cache_f.mean(axis=0)
        cache_v = cache_v / np.linalg.norm(cache_v)
        mine_v = mine_f.mean(axis=0)
        mine_v = mine_v / np.linalg.norm(mine_v)
        cos = float(np.dot(mine_v, cache_v))
        maxdiff = float(np.max(np.abs(mine_v - cache_v)))
        nmin = min(n, nc)
        fc = (mine_f[:nmin] * cache_f[:nmin]).sum(axis=1)
        frame_cos_all.extend(fc.tolist())
        wcos.append(cos)
        rows.append({
            "k": int(k), "id": cache_ids[k], "clip": str(z["clips"][k]),
            "frames_mine": n, "frames_cache": nc, "frame_count_match": bool(n == nc),
            "cosine_window": cos,
            "max_abs_elementwise_diff_normalised": maxdiff,
            "min_frame_cosine": float(fc.min()),
            "median_frame_cosine": float(np.median(fc)),
            "mean_frame_cosine": float(fc.mean()),
        })
    wcos = np.asarray(wcos)
    frame_cos_all = np.asarray(frame_cos_all)
    revisions = {}
    snap = Path(EG2_DIR)
    for f in ["config.json", "config_sentence_transformers.json", "modules.json",
              "preprocessor_config.json", "processor_config.json", "1_Pooling/config.json"]:
        p = snap / f
        if p.exists():
            revisions[f] = sha256_file(p)
    w = snap / "model.safetensors"
    if w.exists():
        revisions["model.safetensors_size_bytes"] = w.stat().st_size
        revisions["model.safetensors_sha256"] = sha256_file(w)

    verdict = ("REPRODUCED: cached EG2 window embeddings regenerate from the source video "
               "via the documented recipe" if wcos.min() >= 0.9999 else
               "RECIPE MATCHES IN SUBSTANCE BUT NOT BIT-EXACTLY: see per-window metrics" if wcos.min() >= 0.99 else
               "NOT REPRODUCIBLE from source video with the documented recipe")
    res = {
        "task": "LB1-eg2-repro",
        "what": "re-embed a sample of IndEgo gold windows from the source video and compare "
                "to the cached frames10k/human_segments/segments.npz EG2 rows",
        "verdict": verdict,
        "model": {
            "id": "google/embeddinggemma-2",
            "revision": snap.name,
            "local_path": EG2_DIR,
            "device": torch.cuda.get_device_name(0),
            "dtype": str(dtype),
            "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "loaded_via": "SentenceTransformer(..., trust_remote_code=True).encode",
            "prompt_prefix": "none (pure image encode; no prompt/text anywhere in the "
                             "cache-producing path)",
            "file_sha256": revisions,
        },
        "recipe": {
            "source_script": "eval/frames10k/embed_segments.py (cache producer; "
                             "configs/ensemble_v1.json embed_command has no --gpu-decode)",
            "decode": "cv2.VideoCapture + CAP_PROP_POS_MSEC seek + read; BGR->RGB; "
                      "PIL convert RGB; thumbnail((512,512), Image.BILINEAR)",
            "step_s": STEP_S,
            "frame_index_rule": "np.arange(start_s, min(end_s, dur), 0.5) in float64; "
                                "unreadable frames skipped (no substitution)",
            "encode": f"batch_size={BATCH_SIZE}, normalize_embeddings=True, "
                      f"processing_kwargs={{'image': {{'max_soft_tokens': {MAX_SOFT_TOKENS}}}}}",
            "storage": "frame embeddings cast to fp16",
            "pooling": "per-window mean over L2-normalised frame embeddings, then L2-normalise "
                       "(eval/exp_fps/run_eg2.py window_mean)",
            "cv2": __import__("cv2").__version__, "pil": Image.__version__,
            "torch": torch.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
        },
        "sample": {
            "n_windows": len(sel), "sampling_rule": SAMPLING_RULE,
            "manifest_order_indices": [int(x) for x in sel],
            "ids": [cache_ids[k] for k in sel],
            "clips_covered": sorted({str(z["clips"][k]) for k in sel}),
        },
        "per_window": rows,
        "summary": {
            "cosine_min": float(wcos.min()),
            "cosine_median": float(np.median(wcos)),
            "cosine_max": float(wcos.max()),
            "max_abs_elementwise_diff_normalised": float(
                max(r["max_abs_elementwise_diff_normalised"] for r in rows)),
            "frame_count_agreement": f"{sum(r['frame_count_match'] for r in rows)}/{len(rows)}",
            "frame_counts": [r["frames_cache"] for r in rows],
            "frame_cosine_min": float(frame_cos_all.min()),
            "frame_cosine_median": float(np.median(frame_cos_all)),
            "frame_cosine_max": float(frame_cos_all.max()),
        },
        "timing_s": {"model_load": round(load_s, 1), "encode": round(enc_s, 1)},
        "vram_peak_gb": vram_peak,
        "cache": {"path": CACHE, "n_windows": int(F.shape[0]), "max_frames": int(F.shape[1]),
                  "dim": int(F.shape[2]), "dtype": str(F.dtype),
                  "counts_sum": int(cache_counts.sum())},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ["verdict", "summary"]}, indent=1))
    print("wrote", OUT_JSON, flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["decode", "encode"], required=True)
    args = ap.parse_args()
    man = json.load(open(MANIFEST))["items"]
    clips = [it["clip"] for it in man]
    sel = select_windows(clips)
    cache_ids = [str(x) for x in np.load(CACHE, allow_pickle=True)["ids"]]
    assert [it["id"] for it in man] == cache_ids, "manifest order != cache ids"
    print(f"n_sample={len(sel)} indices={sel} clips={sorted({clips[k] for k in sel})}",
          flush=True)
    if args.phase == "decode":
        phase_decode(sel, man)
    else:
        phase_encode(sel, man)


if __name__ == "__main__":
    main()
