"""Diagnose the residual gap in the EG2 cache reproduction.

Two competing explanations for the observed frame-level cosine ~0.996 / window ~0.9996:
  H1 bf16 batch-composition numerics (cuBLAS reduction order differs from the
     cache run, which encoded all 3005 frames in one encode() call);
  H2 the cache was produced with --gpu-decode (eval/frames10k/gpu_pre.py), whose
     thumbnail path UPSCALES 480x480 -> 512x512 via cuda bilinear + byte truncation,
     whereas the CPU path's PIL thumbnail((512,512)) leaves 480x480 untouched.

Tests (all bf16, same 200 sample frames, under the GPU lock):
  A baseline  : CPU thumbs (480x480), batch 32            [same as main run]
  B batchvar  : CPU thumbs (480x480), batch 8             [batching sensitivity]
  C shuffled  : CPU thumbs (480x480), batch 32, permuted  [batch composition]
  D gpupre    : GPU-decode preprocessing (512x512), batch 32
Report window cosine vs the cache for each variant and per-frame cosine stats.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_cache_reproduction import CACHE, EG2_DIR, MAX_SOFT_TOKENS, SCRATCH, select_windows, MANIFEST

OUT = SCRATCH / "diagnosis.json"


def gpu_preprocess(rgb480: np.ndarray) -> Image.Image:
    """Exact copy of eval/frames10k/gpu_pre.py gpu_thumb_bgr, fed from an RGB array."""
    bgr = np.ascontiguousarray(rgb480[:, :, ::-1])
    t = torch.from_numpy(bgr).to("cuda").permute(2, 0, 1).contiguous()
    _, h, w = t.shape
    s = 512 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = Fn.interpolate(t[None].float(), size=(nh, nw), mode="bilinear",
                           align_corners=False)[0].clamp(0, 255).byte()
    return Image.fromarray(small.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def main() -> None:
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    z = np.load(CACHE, allow_pickle=True)
    Fc = z["F"]
    ccnt = z["counts"].astype(int)
    clips = [str(c) for c in z["clips"]]
    man = json.load(open(MANIFEST))["items"]
    sel = select_windows([it["clip"] for it in man])
    t = np.load(SCRATCH / "thumbs.npz", allow_pickle=True)
    thumbs, my_counts = t["thumbs"], t["counts"].astype(int)
    assert t["sel"].astype(int).tolist() == sel

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EG2_DIR, model_kwargs={"torch_dtype": torch.bfloat16},
                                trust_remote_code=True).eval()

    def enc(pils, bs=32):
        with torch.inference_mode():
            return np.asarray(model.encode(
                pils, batch_size=bs, show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=True,
                processing_kwargs={"image": {"max_soft_tokens": MAX_SOFT_TOKENS}}),
                dtype=np.float16)

    pils_cpu = [Image.fromarray(a, mode="RGB") for a in thumbs]
    orig = np.asarray([Image.fromarray(a).size for a in thumbs][0])
    print("cpu-thumb size", orig, flush=True)

    variants = {}
    E_A = enc(pils_cpu, 32)
    variants["A_cpu_thumbs_bs32"] = E_A
    variants["B_cpu_thumbs_bs8"] = enc(pils_cpu, 8)
    perm = np.random.RandomState(0).permutation(len(pils_cpu))
    E_C = enc([pils_cpu[i] for i in perm], 32)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    variants["C_cpu_thumbs_bs32_shuffled"] = E_C[inv]
    pils_gpu = [gpu_preprocess(a) for a in thumbs]
    print("gpu-decode-pre thumb size", pils_gpu[0].size, flush=True)
    variants["D_gpu_preproc_512_bs32"] = enc(pils_gpu, 32)

    res = {"sizes": {"cpu_thumb": [int(x) for x in orig],
                     "gpu_preproc": [int(x) for x in pils_gpu[0].size]},
           "variants": {}}
    for name, E in variants.items():
        o = 0
        wcos, fcos, wmax = [], [], []
        for j, k in enumerate(sel):
            n = int(my_counts[j])
            mf = E[o:o + n].astype(np.float32)
            o += n
            cf = Fc[k, :int(ccnt[k])].astype(np.float32)
            cv = cf.mean(0)
            cv /= np.linalg.norm(cv)
            mv = mf.mean(0)
            mv /= np.linalg.norm(mv)
            wcos.append(float(mv @ cv))
            wmax.append(float(np.max(np.abs(mv - cv))))
            nm = min(n, len(cf))
            fcos.extend((mf[:nm] * cf[:nm]).sum(1).tolist())
        wcos = np.asarray(wcos)
        fcos = np.asarray(fcos)
        res["variants"][name] = {
            "window_cosine_min": float(wcos.min()),
            "window_cosine_median": float(np.median(wcos)),
            "window_cosine_max": float(wcos.max()),
            "window_max_abs_diff_max": float(max(wmax)),
            "frame_cosine_min": float(fcos.min()),
            "frame_cosine_median": float(np.median(fcos)),
            "frame_cosine_max": float(fcos.max()),
        }
        print(name, json.dumps(res["variants"][name]), flush=True)

    # cross-variant frame agreement (identical pixels -> pure encoder/batch effect)
    for a, b in [("A_cpu_thumbs_bs32", "B_cpu_thumbs_bs8"),
                 ("A_cpu_thumbs_bs32", "C_cpu_thumbs_bs32_shuffled"),
                 ("A_cpu_thumbs_bs32", "D_gpu_preproc_512_bs32")]:
        A = variants[a].astype(np.float32)
        B = variants[b].astype(np.float32)
        c = (A * B).sum(1)
        res.setdefault("cross_variant_frame_cosine", {})[f"{a}_vs_{b}"] = {
            "min": float(c.min()), "median": float(np.median(c)), "max": float(c.max())}
        print(f"{a} vs {b} frame-cos min/med/max "
              f"{c.min():.5f}/{np.median(c):.5f}/{c.max():.5f}", flush=True)

    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
