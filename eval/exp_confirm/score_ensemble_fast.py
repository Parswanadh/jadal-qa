#!/usr/bin/env python3
"""Fast frozen-ensemble scoring for the 108 hold-out (batched EG2 encode).

Identical math to jadal.ensemble.scorer.score(): the frozen pixel view
(pixel_features_for_window), the frozen EG2 frame sampling
(sample_frames_eg2), and the frozen aggregation (mean over frames then L2
normalise) + frozen heads/fusion (ensemble_prob). The ONLY difference is that
EG2 encodings of many windows are computed in one batched model.encode call
instead of one call per window; per-item normalisation + per-window mean makes
this mathematically identical (validated against the frozen q2 scores with
--validate). No fitting, no tuning.

Run under the GPU lock:
  bash tools/pod/gpu_run.sh env OMP_NUM_THREADS=2 python3 eval/exp_confirm/score_ensemble_fast.py [--validate]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
CFG_BUNDLE = os.path.join(BUNDLE, "caches", "ensemble", "ensemble_v1.json")
MODEL_DIR = os.environ.get(
    "JADAL_EG2_DIR",
    "/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2/"
    "snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")
INPUTS = os.path.join(REPO, "eval", "results", "exp_confirm", "inputs")
CLIPS = os.environ.get("JADAL_CONFIRM_CLIPS", "/workspace/scratch/c1/q3_clips")
CHUNK = 12


def _sample_one(args):
    clip, wid = args
    from jadal.ensemble.scorer import pixel_features_for_window, sample_frames_eg2
    cfg = json.load(open(CFG_BUNDLE))
    cfg["views"]["eg2"]["embed_model_dir"] = MODEL_DIR
    feats, pmeta = pixel_features_for_window(clip, 0.0, 10.0, cfg)
    pils = sample_frames_eg2(clip, 0.0, 10.0, cfg["views"]["eg2"]["frame_step_s"])
    return wid, feats, pmeta, pils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=os.path.join(INPUTS, "confirm108_ensemble.json"))
    args = ap.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from jadal.ensemble.scorer import ensemble_prob

    cfg = json.load(open(CFG_BUNDLE))
    cfg_sha = hashlib.sha256(open(CFG_BUNDLE, "rb").read()).hexdigest()
    v = cfg["views"]["eg2"]
    v["embed_model_dir"] = MODEL_DIR
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = SentenceTransformer(v["embed_model_dir"], model_kwargs={"torch_dtype": dtype},
                                trust_remote_code=True).eval()

    def encode(pils):
        with torch.inference_mode():
            E = model.encode(pils, batch_size=32, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True,
                             processing_kwargs={"image": {"max_soft_tokens": v["max_soft_tokens"]}})
        return np.asarray(E, dtype=np.float64)

    def eg2_mean(E):
        m = E.mean(axis=0)
        return m / max(np.linalg.norm(m), 1e-12)

    if args.validate:
        # compare batched vs the frozen per-window path on 2 frozen q2 windows
        from jadal.ensemble.scorer import eg2_mean_for_window
        q2 = json.load(open(os.path.join(BUNDLE, "queues", "label_queue_random2.json")))["segments"]
        ref = {}
        for s in json.load(open(os.path.join(BUNDLE, "caches", "ensemble", "ood_scores2.json")))["scores"]:
            ref[s["id"]] = s
        picks = [s for s in q2 if s["segment_id"] in ref][:2]
        batch = [(os.path.join(BUNDLE, "data/ego10k_scale/label_windows2",
                               s["segment_id"] + ".mp4"), s["segment_id"]) for s in picks]
        with mp.Pool(2) as pool:
            rows = pool.map(_sample_one, batch)
        allpils = [p for _, _, _, ps in rows for p in ps]
        Eall = encode(allpils)
        off = 0
        for wid, feats, pmeta, ps in rows:
            E = Eall[off:off + len(ps)]; off += len(ps)
            p, pe, pg = ensemble_prob(eg2_mean(E), feats, cfg)
            eg2m_direct, _ = eg2_mean_for_window(batch[[b[1] for b in batch].index(wid)][0], 0, 10, cfg)
            p_d, pe_d, pg_d = ensemble_prob(eg2m_direct, feats, cfg)
            r = ref[wid]
            print(f"VAL {wid}: batch p={p:.4f} pe={pe:.4f} pg={pg:.4f} | per-window p={p_d:.4f} pe={pe_d:.4f} | "
                  f"frozen p={r['usable_prob']} pe={r['p_eg2']} pg={r['p_pixel']}")
        return

    items = json.load(open(os.path.join(INPUTS, "windows108.json")))
    jobs = [(os.path.join(CLIPS, it["id"] + ".mp4"), it["id"]) for it in items]
    out = {"config_sha256": cfg_sha, "model_dir": MODEL_DIR,
           "method": "frozen two-view, batched EG2 encode (validated identical)", "scores": []}
    with mp.Pool(2) as pool:
        for ci in range(0, len(jobs), CHUNK):
            chunk = jobs[ci:ci + CHUNK]
            rows = pool.map(_sample_one, chunk)
            allpils = [p for _, _, _, ps in rows for p in ps]
            Eall = encode(allpils)
            off = 0
            for wid, feats, pmeta, ps in rows:
                E = Eall[off:off + len(ps)]; off += len(ps)
                p, pe, pg = ensemble_prob(eg2_mean(E), feats, cfg)
                out["scores"].append({"id": wid, "usable_prob": round(p, 4),
                                      "p_eg2": round(pe, 4), "p_pixel": round(pg, 4),
                                      "n_frames_pixel": pmeta["n_frames"],
                                      "n_frames_eg2": int(len(ps))})
            print(f"chunk {ci//CHUNK+1} done ({min(ci+CHUNK,len(jobs))}/{len(jobs)})", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out, "n", len(out["scores"]))


if __name__ == "__main__":
    main()
