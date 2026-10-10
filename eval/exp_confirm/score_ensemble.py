#!/usr/bin/env python3
"""Score the 108 round-3 hold-out windows with the FROZEN two-view ensemble_v1.

Uses the frozen scorer functions verbatim (jadal.ensemble.scorer) and the frozen
heads from the bundle config (raw sha256 4da1068a..., the sha recorded in the
frozen ood_scores.json). Only the EG2 model path is repointed to the pod's local
HF snapshot (the heads/coefficients are byte-identical; only the path string
differs). No fitting, no tuning.

Run under the GPU lock:
  bash tools/pod/gpu_run.sh env OMP_NUM_THREADS=2 python3 eval/exp_confirm/score_ensemble.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

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


def main():
    from jadal.ensemble.scorer import (ensemble_prob, pixel_features_for_window,
                                       eg2_mean_for_window, features_digest)
    cfg = json.load(open(CFG_BUNDLE))
    cfg_sha = hashlib.sha256(open(CFG_BUNDLE, "rb").read()).hexdigest()
    # repoint only the model path (heads untouched)
    cfg["views"]["eg2"]["embed_model_dir"] = MODEL_DIR
    items = json.load(open(os.path.join(INPUTS, "windows108.json")))
    out = {"config_sha256": cfg_sha, "model_dir": MODEL_DIR, "scores": []}
    for it in items:
        t0 = time.time()
        clip = os.path.join(CLIPS, it["id"] + ".mp4")
        rec = {"id": it["id"]}
        try:
            feats, pmeta = pixel_features_for_window(clip, 0.0, 10.0, cfg)
            eg2m, emeta = eg2_mean_for_window(clip, 0.0, 10.0, cfg)
            p, p_e, p_g = ensemble_prob(eg2m, feats, cfg)
            rec.update({"usable_prob": round(p, 4), "p_eg2": round(p_e, 4),
                        "p_pixel": round(p_g, 4),
                        "n_frames_pixel": pmeta["n_frames"],
                        "n_frames_eg2": emeta["n_frames"],
                        "features_digest": features_digest(
                            feats, eg2m, {"n_frames_pixel": pmeta["n_frames"],
                                          "n_frames_eg2": emeta["n_frames"]})})
        except Exception as e:
            rec["error"] = str(e)[:300]
        rec["latency_ms"] = int((time.time() - t0) * 1000)
        out["scores"].append(rec)
    path = os.path.join(INPUTS, "confirm108_ensemble.json")
    json.dump(out, open(path, "w"), indent=1)
    ok = sum(1 for s in out["scores"] if "error" not in s)
    print(f"scored {ok}/{len(out['scores'])} -> {path}")


if __name__ == "__main__":
    main()
