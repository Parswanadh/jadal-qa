"""Pixel-sensitivity control: how large a *pixel-level* difference explains the
residual between my re-embed and the cache?

Norm-invariant analysis (renormalise every frame before the dot product) gives, for the
10 sampled windows:
    mine vs mine, different batch composition : frame cosine >= 0.999967 (i.e. bf16 is
                                                directionally deterministic here)
    mine vs cache                            : frame cosine median 0.99534  (||delta||~0.10)
So the cache residual is a real per-frame directional difference, not fp16/kernel noise,
but far too small for a wrong frame index (median 0.78) or the gpu-decode preprocessing
variant (median 0.94).

This control asks: is it consistent with a few-LSB pixel difference (e.g. a different
FFmpeg/cv2 build decoding the same H.264 slightly differently)? Perturb the SAME thumbs by
uniform integer noise of +-1 / +-2 / +-4 LSB and re-encode; report the frame cosine vs the
unperturbed cache comparison baseline.

  flock -w 3600 /workspace/state/gpu.slot2.lock env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
    python3 eval/frames10k/verify_pixel_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_cache_reproduction import CACHE, EG2_DIR, MAX_SOFT_TOKENS, SCRATCH, select_windows, MANIFEST

OUT = SCRATCH / "pixel_sensitivity.json"


def rn(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def main() -> None:
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    z = np.load(CACHE, allow_pickle=True)
    F, ccnt = z["F"], z["counts"].astype(int)
    man = json.load(open(MANIFEST))["items"]
    sel = select_windows([it["clip"] for it in man])
    t = np.load(SCRATCH / "thumbs.npz", allow_pickle=True)
    thumbs, counts = t["thumbs"], t["counts"].astype(int)
    assert t["sel"].astype(int).tolist() == sel

    model = SentenceTransformer(EG2_DIR, model_kwargs={"torch_dtype": torch.bfloat16},
                                trust_remote_code=True).eval()

    def enc(arrays):
        with torch.inference_mode():
            return np.asarray(model.encode(
                [Image.fromarray(a, mode="RGB") for a in arrays], batch_size=32,
                show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
                processing_kwargs={"image": {"max_soft_tokens": MAX_SOFT_TOKENS}}),
                dtype=np.float16).astype(np.float32)

    def vs_cache(E):
        o, fc, wc = 0, [], []
        for j, k in enumerate(sel):
            n = int(counts[j])
            a = rn(E[o:o + n])
            o += n
            b = rn(F[k, :int(ccnt[k])].astype(np.float32))
            fc.extend((a * b).sum(1).tolist())
            wc.append(float(rn(a.mean(0)[None])[0] @ rn(b.mean(0)[None])[0]))
        fc, wc = np.asarray(fc), np.asarray(wc)
        return {"frame_cosine_min": float(fc.min()), "frame_cosine_median": float(np.median(fc)),
                "frame_cosine_p10": float(np.percentile(fc, 10)),
                "frame_cosine_max": float(fc.max()),
                "window_cosine_min": float(wc.min()),
                "window_cosine_median": float(np.median(wc)),
                "window_cosine_max": float(wc.max())}

    res = {"note": "identical pixels, uniform integer LSB perturbation before encode"}
    res["unperturbed"] = vs_cache(enc(thumbs))
    print("unperturbed", json.dumps(res["unperturbed"]), flush=True)
    rng = np.random.RandomState(7)
    for amp in (1, 2, 4):
        noise = rng.randint(-amp, amp + 1, size=thumbs.shape, dtype=np.int16)
        pert = np.clip(thumbs.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        res[f"noise_pm{amp}_lsb"] = vs_cache(enc(pert))
        print(f"noise +-{amp} LSB", json.dumps(res[f"noise_pm{amp}_lsb"]), flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
