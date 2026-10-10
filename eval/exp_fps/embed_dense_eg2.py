"""E1.02b: dense EG2 re-embed of the 150 IndEgo human windows with ONE decode pass.

Recipe identical to eval/frames10k/embed_segments.py (the source of the 2 fps
cache segments.npz), only the sampling grid is denser:

  * seek to start_s once, then read consecutively -> the native 10 fps frames
    of the 10 s window (100 frames/window);
  * frame -> RGB -> PIL -> thumbnail(512, 512, BILINEAR)   (same preprocessing)
  * EmbeddingGemma-2, bf16, processing_kwargs image max_soft_tokens=140,
    normalize_embeddings=True                             (same backbone)
  * per-frame L2-normalised rows are what we store; the window mean / L2-norm
    pooling happens in run_e1_02b.py                        (same pooling)

5 fps and 2 fps views are exact row subsamples (::2, ::5) of the 10 fps rows,
so the three densities share one decode and one embedding pass and the
comparison across densities is paired by construction.

Output (NOT committed; too large for git):
  <out>/dense10.npz  F (150, <=100, 768) fp16 + ids/clips/labels/counts/start_s
  <out>/meta.json    device, revision, timings, VRAM peak
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

EG2_DIR = ("/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2/"
           "snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")
PREPROCESS_MAX_SIDE = 512
NATIVE_FPS = 10.0
WIN_FRAMES = 100          # 10 s window at native 10 fps
DEFAULT_MANIFEST = "/workspace/jadal_pod_bundle/labels/manifest_gold.json"
DEFAULT_OUT = "/workspace/cache_e1_02b"


def decode_window(path: str, start_s: float, n_frames: int = WIN_FRAMES):
    """Seek once to start_s, then read n_frames consecutive native frames."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)
        frames = []
        while len(frames) < n_frames:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            frames.append(fr)
        return frames
    finally:
        cap.release()


def to_pil(frame_bgr):
    """Same preprocessing as embed_segments.py: BGR->RGB, PIL RGB, thumbnail 512."""
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).convert("RGB")
    img.thumbnail((PREPROCESS_MAX_SIDE, PREPROCESS_MAX_SIDE), Image.BILINEAR)
    return img


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-windows", type=int, default=8,
                    help="windows decoded+encoded per chunk (RAM bound)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N windows only")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), "requires CUDA"
    dev_name = torch.cuda.get_device_name(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    torch.cuda.reset_peak_memory_stats()
    model = SentenceTransformer(str(EG2_DIR), model_kwargs={"torch_dtype": dtype},
                                trust_remote_code=True)
    model.eval()

    man = json.load(open(args.manifest))
    items = man["items"]
    if args.limit:
        items = items[:args.limit]
    print(f"windows={len(items)} snapshot={man['snapshot_utc']} "
          f"labels_sha={man['labels_sha256'][:12]} device={dev_name}", flush=True)

    ids = [it["id"] for it in items]
    clips = [it["clip"] for it in items]
    labels = [int(it["label"]) for it in items]
    starts = [float(it["start_s"]) for it in items]

    F = np.full((len(items), WIN_FRAMES, 768), np.nan, dtype=np.float16)
    counts = np.zeros(len(items), dtype=np.int16)
    t_enc = 0.0
    t_dec = 0.0

    for i0 in range(0, len(items), args.chunk_windows):
        chunk = items[i0:i0 + args.chunk_windows]
        pils = []
        cnts = []
        t0 = time.time()
        for it in chunk:
            frames = decode_window(it["path"], float(it["start_s"]))
            cnts.append(len(frames))
            for f in frames:
                pils.append(to_pil(f))
        t_dec += time.time() - t0
        if not pils:
            continue
        t1 = time.time()
        with torch.inference_mode():
            E = model.encode(pils, batch_size=args.batch_size, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True,
                             processing_kwargs={"image": {"max_soft_tokens": 140}})
        t_enc += time.time() - t1
        E = np.asarray(E, dtype=np.float16)
        o = 0
        for j, c in enumerate(cnts):
            F[i0 + j, :c] = E[o:o + c]
            counts[i0 + j] = c
            o += c
        del pils, E
        done = min(i0 + args.chunk_windows, len(items))
        print(f"  {done}/{len(items)} windows  frames={int(counts[:done].sum())}  "
              f"dec={t_dec:.0f}s enc={t_enc:.0f}s", flush=True)

    n_frames = int(counts.sum())
    vram_peak = float(torch.cuda.max_memory_allocated() / 1e9)
    np.savez_compressed(out / "dense10.npz", F=F, ids=np.asarray(ids),
                        clips=np.asarray(clips), labels=np.asarray(labels, dtype=np.int8),
                        counts=counts, start_s=np.asarray(starts, dtype=np.float64))
    wt = EG2_DIR
    meta = {
        "experiment": "E1.02b",
        "n_windows": len(items), "n_frames": n_frames,
        "frames_per_window_mean": float(counts.mean()),
        "empty_windows": int((counts == 0).sum()),
        "sampling": "10 fps native (seek to start_s then read 100 consecutive frames)",
        "preprocess": f"BGR->RGB, PIL RGB, thumbnail({PREPROCESS_MAX_SIDE},{PREPROCESS_MAX_SIDE}) BILINEAR",
        "encode": "SentenceTransformer.encode(normalize_embeddings=True, "
                  "processing_kwargs={'image': {'max_soft_tokens': 140}})",
        "eg2_snapshot_dir": wt, "eg2_revision": Path(wt).name,
        "eg2_model_sha256": sha256_file(Path(wt) / "model.safetensors"),
        "device": dev_name, "dtype": str(dtype), "batch_size": args.batch_size,
        "chunk_windows": args.chunk_windows,
        "decode_s": t_dec, "encode_s": t_enc,
        "frames_per_s_encode": n_frames / t_enc if t_enc else None,
        "vram_peak_gb": vram_peak,
        "snapshot_utc": man["snapshot_utc"], "labels_sha256": man["labels_sha256"],
        "label_source": "human (single labeller; read-only snapshot)",
        "cache_path": str(out / "dense10.npz"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
