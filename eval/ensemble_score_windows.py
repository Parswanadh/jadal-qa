"""Batch window scoring for the frozen ensemble (decode+EG2+heads, GPU for EG2).

Single process, model loaded once (use under the gpu lock for EG2 parts):
  OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python eval/ensemble_score_windows.py \
      --windows <json:[{id,path,start_s,end_s}]> --out <scores.json> [--no-eg2]

Writes {sha_config, model, scores: [{id, usable_prob, verdict, p_eg2, p_pixel,
margin, features_digest, reasons, versions, latency_ms, n_frames_pixel,
n_frames_eg2}]}. Deterministic on same machine+device (see scorer docstring).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-eg2", action="store_true",
                    help="pixel view only (p_eg2 null; usable_prob = pixel head)")
    args = ap.parse_args()

    from jadal.ensemble.scorer import (load_config, ensemble_prob, pixel_features_for_window,
                                       eg2_mean_for_window, features_digest, file_sha256,
                                       verdict_for)
    cfg, used = load_config()
    cfg_sha = file_sha256(used)
    items = json.load(open(args.windows))
    out = {"config_sha256": cfg_sha, "scores": []}
    for it in items:
        t0 = time.time()
        try:
            feats, pmeta = pixel_features_for_window(it["path"], it["start_s"], it["end_s"], cfg)
            if args.no_eg2:
                from jadal.ensemble.scorer import head_prob
                import numpy as np
                order = cfg["views"]["pixel"]["feature_names"]
                x = np.array([float(feats[k]) for k in order])
                p_g = head_prob(x, cfg["views"]["pixel"])
                p, p_e = p_g, None
                eg2m, emeta = np.zeros(768), {"n_frames": 0}
            else:
                eg2m, emeta = eg2_mean_for_window(it["path"], it["start_s"], it["end_s"], cfg)
                p, p_e, p_g = ensemble_prob(eg2m, feats, cfg)
            prob = round(p, 4)
            lo, hi, thr = cfg["abstain_band"][0], cfg["abstain_band"][1], cfg["threshold"]
            verdict = verdict_for(prob, lo, hi, thr)
            meta = {"n_frames_pixel": pmeta["n_frames"], "n_frames_eg2": emeta["n_frames"]}
            out["scores"].append({"id": it.get("id"), "usable_prob": prob, "verdict": verdict,
                                  "p_eg2": None if p_e is None else round(p_e, 4),
                                  "p_pixel": round(p_g, 4), "margin": round(p - 0.5, 4),
                                  "features_digest": features_digest(feats, eg2m, meta),
                                  "n_frames_pixel": pmeta["n_frames"],
                                  "n_frames_eg2": emeta["n_frames"],
                                  "latency_ms": int((time.time() - t0) * 1000)})
        except Exception as e:
            out["scores"].append({"id": it.get("id"), "error": str(e)[:300]})
    json.dump(out, open(args.out, "w"), indent=1)
    ok = sum(1 for s in out["scores"] if "error" not in s)
    print(f"scored {ok}/{len(out['scores'])} -> {args.out}")


if __name__ == "__main__":
    main()
