"""EG2 embedding latency benchmark (issue #7, FIRST JOB — run ALONE on the GPU).

Measures per-frame EmbeddingGemma-2 embedding latency (median/p95, warmed)
at image token budgets 70/140/280, peak VRAM per budget, and batch-size-1
vs batch-of-5 output equality (max abs diff + bit-equality).

Usage (from repo root, GPU idle, nothing else on the GPU)::

    PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python eval/bench_eg2.py \
        --out eval/results/bench_eg2.json

Inputs are REAL frames from ``/home/parshu/projects/awing/data/ego10k_eval/*.parquet``
(first 2 frame_ids per file after sorting, preprocessing byte-identical to
serving via ``jadal.decision.preprocess_pil``). Frame ids are recorded in the
output JSON. No labels are used, so the gemini-labelling caveat does not
apply to this file (it applies to decider_baseline*.json).
Model weights come from ``/home/parshu/projects/awing/data/eg2`` (local
only, no network). Inference is bfloat16 (NEVER fp16 — EG2 activations
overflow fp16 range), eval mode, deterministic algorithms on,
cudnn.benchmark off, fixed seeds.

NOT DONE:
- Latency covers backbone embedding only, not OpenCV decode + sampling +
  heads (see the decider baseline for end-to-end context).
- No text/video/audio modality timing (image-only for the L1 baseline).
"""
from __future__ import annotations

import argparse
import io
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

from jadal import decision as decision_mod

EG2_DIR = decision_mod.EG2_DIR
DATA_DIR = "/home/parshu/projects/awing/data/ego10k_eval"
PARQUET_FILES = ("ego4d.parquet", "egocentric_10k.parquet", "epic_kitchens.parquet")
FRAMES_PER_FILE = 2
BUDGETS = (70, 140, 280)
SEED = 1234
WARMUP = 5
TIMED = 20
GO_NOGO_S_PER_FRAME = 1.0  # <= ~1 s/frame at 140 tokens => per-batch scoring viable


def load_real_frames() -> list[dict]:
    """First FRAMES_PER_FILE frame_ids per parquet file (sorted), preprocessed like serving."""
    import pyarrow.parquet as pq

    frames: list[dict] = []
    for f in PARQUET_FILES:
        pf = pq.ParquetFile(str(Path(DATA_DIR) / f))
        rows: list[dict] = []
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_groups([rg], columns=["frame_id", "image"])
            d = tbl.to_pydict()
            for i in range(tbl.num_rows):
                rows.append({"frame_id": d["frame_id"][i],
                             "bytes": bytes(d["image"][i]["bytes"])})
        rows.sort(key=lambda r: r["frame_id"])
        for r in rows[:FRAMES_PER_FILE]:
            img = decision_mod.preprocess_pil(Image.open(io.BytesIO(r["bytes"])))
            frames.append({"frame_id": r["frame_id"], "file": f, "img": img})
    return frames


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


def bench_budget(model, budget: int, frames: list[dict]) -> dict:
    imgs = [f["img"] for f in frames]
    for _ in range(WARMUP):  # warm (not timed)
        encode_one(model, imgs[0], budget)
    torch.cuda.synchronize()
    times: list[float] = []
    for k in range(TIMED):
        t0 = time.perf_counter()
        encode_one(model, imgs[k % len(imgs)], budget)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = statistics.median(times)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    torch.cuda.reset_peak_memory_stats()
    encode_one(model, imgs[1 % len(imgs)], budget)
    torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    # batch-size-1 vs batch-of-5 equality on the same frame
    e1 = encode_one(model, imgs[0], budget)
    e5 = encode_batch(model, [imgs[0]] * 5, budget)
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

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark; skipping.", file=sys.stderr)
        return 2

    frames = load_real_frames()
    model, dtype = load_model(args.model_dir)
    per_budget = [bench_budget(model, b, frames) for b in BUDGETS]
    lat140 = next(r["median_s_per_frame"] for r in per_budget if r["budget"] == 140)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_dir": args.model_dir,
        "torch_dtype": dtype,
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "frames": [{"frame_id": f["frame_id"], "file": f["file"],
                    "size": list(f["img"].size)} for f in frames],
        "frames_note": ("real eval-parquet frames (first 2 frame_ids per file, sorted), "
                        "preprocessing byte-identical to serving "
                        "(jadal.decision.preprocess_pil); no labels used"),
        "per_budget": per_budget,
        "go_nogo_threshold_s_per_frame": GO_NOGO_S_PER_FRAME,
        "median_s_per_frame_at_140": lat140,
        "go_nogo_140": "GO" if lat140 <= GO_NOGO_S_PER_FRAME else "NO-GO",
        "notes": (
            "Real-frame image-only embeddings; bfloat16; batch size 1; "
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
