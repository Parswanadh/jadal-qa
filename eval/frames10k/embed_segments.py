"""Embed human-labelled segment frames with EG2 (GPU, under flock; batched).

Coverage: ALL frames of each of the 150 human-labelled 10-s segments is too
costly at full fps, so: sample every 0.5 s (20 frames/segment, whole-segment
coverage, NOT 4 frames). Same backbone/preprocessing as embed_frames.py.
Output: eval/results/frames10k/human_segments/segments.npz + meta.json.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np

EG2_DIR = "/home/parshu/projects/awing/data/eg2"
PREPROCESS_MAX_SIDE = 512
STEP_S = 0.5

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "frames10k" / "human_segments"


def sample_segment(path: str, start: float, end: float, step: float = STEP_S,
                   gpu: bool = False):
    import cv2
    from PIL import Image

    if gpu:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gpu_pre import gpu_thumb_bgr

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps else 0.0
        times = np.arange(start, min(end, dur), step)
        pils = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if gpu:
                pils.append(gpu_thumb_bgr(frame))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).convert("RGB")
            img.thumbnail((PREPROCESS_MAX_SIDE, PREPROCESS_MAX_SIDE), Image.BILINEAR)
            pils.append(img)
        return pils
    finally:
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gpu-decode", action="store_true",
                    help="upload BGR frames to GPU for RGB/resize (else CPU cv2+PIL)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), "requires CUDA"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    torch.cuda.reset_peak_memory_stats()
    model = SentenceTransformer(str(EG2_DIR), model_kwargs={"torch_dtype": dtype},
                                trust_remote_code=True)
    model.eval()

    man = json.load(open(args.manifest))
    items = man["items"]
    print(f"segments={len(items)} snapshot={man['snapshot_utc']} sha={man['labels_sha256'][:12]}", flush=True)

    # decode all (CPU) first, keep PILs (150*20 small thumbs ~ fine)
    all_pils, counts, ids, clips, labels, reasons = [], [], [], [], [], []
    t_dec0 = time.time()
    for it in items:
        pils = sample_segment(it["path"], it["start_s"], it["end_s"], gpu=args.gpu_decode)
        all_pils.extend(pils)
        counts.append(len(pils))
        ids.append(it["id"])
        clips.append(it["clip"])
        labels.append(int(it["label"]))
        reasons.append(it.get("reason", ""))
    t_dec = time.time() - t_dec0
    print(f"decoded {sum(counts)} frames ({np.mean(counts):.1f}/seg) in {t_dec:.1f}s", flush=True)

    t1 = time.time()
    with torch.inference_mode():
        E = model.encode(all_pils, batch_size=args.batch_size, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True,
                         processing_kwargs={"image": {"max_soft_tokens": 140}})
    t_enc = time.time() - t1
    E = np.asarray(E, dtype=np.float16)
    vram_peak = float(torch.cuda.max_memory_allocated() / 1e9)
    n = len(all_pils)
    print(f"encoded {n} frames in {t_enc:.1f}s = {n/t_enc:.2f} fps; VRAM peak {vram_peak:.2f} GB", flush=True)

    # repack per segment (pad to max with nan)
    T = max(counts)
    F = np.full((len(items), T, E.shape[1]), np.nan, dtype=np.float16)
    o = 0
    for i, c in enumerate(counts):
        F[i, :c] = E[o:o + c]
        o += c
    np.savez_compressed(OUT / "segments.npz", F=F,
                        ids=np.asarray(ids), clips=np.asarray(clips),
                        labels=np.asarray(labels, dtype=np.int8),
                        reasons=np.asarray(reasons), counts=np.asarray(counts, dtype=np.int16))
    meta = {"n_segments": len(items), "n_frames": n, "frames_per_seg_mean": float(np.mean(counts)),
            "step_s": STEP_S, "decode_s": t_dec, "encode_s": t_enc,
            "frames_per_s_encode": float(n / t_enc),
            "frames_per_s_wall": float(n / (t_dec + t_enc)),
            "vram_peak_gb": vram_peak, "batch_size": args.batch_size,
            "gpu_decode": bool(args.gpu_decode),
            "snapshot_utc": man["snapshot_utc"], "labels_sha256": man["labels_sha256"],
            "n_good": man["n_good"], "n_bad": man["n_bad"],
            "label_source": "human (single labeller; read-only snapshot)"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
