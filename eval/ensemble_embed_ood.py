"""Embed every 0.5 s frame of the 60 OOD windows with EG2 (GPU, under lock).

Reuses jadal.ensemble.scorer sampling/encode path (same preprocessing as the
frozen model). Writes eval/results/ensemble/ood_frames.npz
{F (60,T,768) fp16 nan-padded, ids, counts} + ood_windows.json passthrough ids.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUEUE = "/home/parshu/projects/jadal-qa-wt/scale-data/eval/results/scale/label_queue_random.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "ensemble", "ood_frames.npz")


def main():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from jadal.ensemble.scorer import load_config, sample_frames_eg2
    cfg, _ = load_config()
    v = cfg["views"]["eg2"]
    segs = json.load(open(QUEUE))["segments"]
    print(f"windows={len(segs)}", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = SentenceTransformer(v["embed_model_dir"],
                                model_kwargs={"torch_dtype": dtype},
                                trust_remote_code=True).eval()
    all_pils, counts, ids = [], [], []
    t0 = time.time()
    for s in segs:
        pils = sample_frames_eg2(s["path"], s["start_s"], s["end_s"], v["frame_step_s"])
        all_pils.extend(pils)
        counts.append(len(pils))
        ids.append(s["segment_id"])
    print(f"decoded {sum(counts)} frames in {time.time()-t0:.1f}s", flush=True)
    t1 = time.time()
    with torch.inference_mode():
        E = model.encode(all_pils, batch_size=32, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True,
                         processing_kwargs={"image": {"max_soft_tokens": v["max_soft_tokens"]}})
    print(f"encoded {len(all_pils)} in {time.time()-t1:.1f}s", flush=True)
    E = np.asarray(E, dtype=np.float16)
    T = max(counts)
    F = np.full((len(ids), T, E.shape[1]), np.nan, dtype=np.float16)
    o = 0
    for i, c in enumerate(counts):
        F[i, :c] = E[o:o + c]
        o += c
    np.savez_compressed(OUT, F=F, ids=np.asarray(ids),
                        counts=np.asarray(counts, dtype=np.int16))
    print(f"wrote {OUT} F={F.shape}")


if __name__ == "__main__":
    main()
