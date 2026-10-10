"""EG2 embedding latency benchmark (issue #7, FIRST JOB — run ALONE on the GPU).

Measures per-frame EmbeddingGemma-2 embedding latency (median/p95, warmed)
at image token budgets 70/140/280, peak VRAM per budget, and batch-size-1
vs batch-of-5 output equality (max abs diff + bit-equality).

Usage (from repo root, GPU idle, nothing else on the GPU)::

    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/bench_eg2.py \
        --out eval/results/bench_eg2.json

Inputs are deterministic synthetic PIL frames (fixed gradients, no data
dependency) so the benchmark is self-contained. Model weights come from
``/home/parshu/projects/awing/data/eg2`` (local only, no network).
Inference is bfloat16 (NEVER fp16 — EG2 activations overflow fp16 range),
eval mode, deterministic algorithms on, cudnn.benchmark off, fixed seeds.

NOT DONE:
- Latency is measured on synthetic frames, not real head-camera segments;
  real-segment end-to-end timing (OpenCV decode + sampling + heads) is
  measured by the decider baseline instead.
- No text/video/audio modality timing (image-only for the L1 baseline).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EG2_DIR = "/home/parshu/projects/awing/data/eg2"
BUDGETS = (70, 140, 280)
SEED = 1234
WARMUP = 5
TIMED = 20
GO_NOGO_S_PER_FRAME = 1.0  # <= ~1 s/frame at 140 tokens => per-batch scoring viable


def make_frame(i: int, w: int = 640, h: int = 480) -> Image.Image:
    """Deterministic synthetic frame i (fixed gradient + bars, no randomness)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = ((xx + i * 37) % 256).astype(np.uint8)
    g = ((yy + i * 91) % 256).astype(np.uint8)
    b = (((xx + yy) // 2 + i * 13) % 256).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def load_model(model_dir: str):
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = SentenceTransformer(
        model_dir,
        model_kwargs={"torch_dtype": dtype},
        trust_remote_code=True,
    )
    model.eval()
    return model, str(dtype)


def encode_one(model, img: Image.Image, budget: int) -> np.ndarray:
    with torch.inference_mode():
        emb = model.encode(
            img,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
            processing_kwargs={"image": {"max_soft_tokens": budget}},
        )
    return np.asarray(emb, dtype=np.float64)


def encode_batch(model, imgs: list, budget: int) -> np.ndarray:
    with torch.inference_mode():
        emb = model.encode(
            imgs,
            batch_size=len(imgs),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
            processing_kwargs={"image": {"max_soft_tokens": budget}},
        )
    return np.asarray(emb, dtype=np.float64)


def bench_budget(model, budget: int) -> dict:
    frames = [make_frame(i) for i in range(3)]
    for _ in range(WARMUP):  # warm (not timed)
        encode_one(model, frames[0], budget)
    torch.cuda.synchronize()
    times: list[float] = []
    for k in range(TIMED):
        t0 = time.perf_counter()
        encode_one(model, frames[k % 3], budget)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = statistics.median(times)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    torch.cuda.reset_peak_memory_stats()
    encode_one(model, frames[1], budget)
    torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    # batch-size-1 vs batch-of-5 equality on the same frame
    e1 = encode_one(model, frames[0], budget)
    e5 = encode_batch(model, [frames[0]] * 5, budget)
    diffs = np.abs(e5 - e1[None, :])
    max_abs_diff = float(diffs.max())
    bit_equal = bool(np.array_equal(e5, e1[None, :]))
    return {
        "budget": budget,
        "n_timed": TIMED,
        "n_warmup": WARMUP,
        "median_s_per_frame": round(float(median), 4),
        "p95_s_per_frame": round(float(p95), 4),
        "peak_vram_mb_batch1": round(float(peak_vram_mb), 1),
        "batch1_vs_batch5_max_abs_diff": max_abs_diff,
        "batch1_vs_batch5_bit_equal": bit_equal,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/bench_eg2.json")
    ap.add_argument("--model-dir", default=EG2_DIR)
    args = ap.parse_args()
    model_dir = args.model_dir

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark; skipping.", file=sys.stderr)
        return 2

    model, dtype = load_model(model_dir)
    per_budget = [bench_budget(model, b) for b in BUDGETS]
    lat140 = next(r["median_s_per_frame"] for r in per_budget if r["budget"] == 140)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_dir": model_dir,
        "torch_dtype": dtype,
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "per_budget": per_budget,
        "go_nogo_threshold_s_per_frame": GO_NOGO_S_PER_FRAME,
        "median_s_per_frame_at_140": lat140,
        "go_nogo_140": "GO" if lat140 <= GO_NOGO_S_PER_FRAME else "NO-GO",
        "notes": (
            "Synthetic deterministic frames; image-only; bfloat16; batch size 1; "
            "per-call processing_kwargs image.max_soft_tokens sets the vision budget."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    raise SystemExit(main())
