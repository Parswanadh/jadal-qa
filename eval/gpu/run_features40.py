"""Extract the 8 FRESH per-pixel features of the 26->40 expansion.

The 6 Z3 features (ct_fsize_std, mo_hand_motion_excess, inv_clip_frac,
inv_laplacian_var, inv_hist_entropy, inv_dct_hf_lf) are REUSED from the frozen
caches (z3_160.npz, confirm108_features.npz) -- "reuse caches; compute only the
new ones". This script decodes each window once and computes the 8 remaining
new features:

  spec_b1, spec_b2, spec_b3      finer spectral radial bands (FFT energy shells)
  col_rg_ratio, col_bg_ratio     colour-channel ratios (mean R/G, mean B/G)
  temp_lag2, temp_lag3           temporal-lag differences (|frame_t - frame_{t-k}|)
  noise_floor                    per-pixel noise floor (MAD of the Laplacian)

Sets: gold 150 (IndEgo), ood160 (Egocentric-10K, human labels),
holdout108 (round-3 hold-out, evaluate only).

CPU-only (no GPU lock). Run:
  OMP_NUM_THREADS=2 python3 eval/gpu/run_features40.py
Writes eval/results/exp_feat40/feat40_new8_<set>.npz
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from eval.gpu.decode import decode_window  # noqa: E402
from jadal.gpu.features import cpu_new_features  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(HERE, "..", "results", "exp_feat40")
W, H = 160, 90
NEW8 = ["spec_b1", "spec_b2", "spec_b3", "col_rg_ratio", "col_bg_ratio",
        "temp_lag2", "temp_lag3", "noise_floor"]


def _resolve_holdout(path):
    for sub in ("clips", "clips_more"):
        p = os.path.join(BUNDLE, "data", "ego10k_scale", sub, path)
        if os.path.exists(p):
            return p
    return os.path.join(BUNDLE, "data", "ego10k_scale", "clips", path)


def build_items():
    sets = {}
    g = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))["items"]
    sets["gold"] = [{"id": it["id"],
                     "path": it["path"].replace("/home/parshu/projects/awing/data",
                                                os.path.join(BUNDLE, "data")),
                     "start_s": it["start_s"], "dur": round(it["end_s"] - it["start_s"], 2)}
                    for it in g]
    o = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))["windows"]
    sets["ood"] = [{"id": x["segment_id"],
                    "path": os.path.join(BUNDLE, x["video"]),
                    "start_s": 0.0, "dur": 10.0} for x in o]
    h = json.load(open(os.path.join(BUNDLE, "labels", "round3",
                                    "ood_labels3_108_with_factory.json")))["windows"]
    sets["holdout"] = [{"id": x["segment_id"],
                        "path": _resolve_holdout(x["source_clip"]),
                        "start_s": float(x["source_start_s"]), "dur": 10.0}
                       for x in h]
    return sets


def process(item):
    try:
        rgb, fps = decode_window(item["path"], item["start_s"], item["dur"], W, H, "rgb24")
    except Exception as e:
        return item["id"], {"error": str(e)[:200]}
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    f = cpu_new_features(luma, rgb, fps=min(fps, 30.0))
    return item["id"], f


def main(jobs=8):
    sets = build_items()
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for name, items in sets.items():
        t0 = time.perf_counter()
        with mp.Pool(min(jobs, len(items))) as pool:
            rows = pool.map(process, items, chunksize=4)
        ids = [r[0] for r in rows]
        X = np.asarray([[float(r[1].get(k, np.nan)) for k in NEW8] for r in rows], float)
        errs = sum(1 for _, r in rows if isinstance(r, dict) and "error" in r)
        wall = time.perf_counter() - t0
        np.savez_compressed(os.path.join(OUT, f"feat40_new8_{name}.npz"),
                            ids=np.asarray(ids), names=np.asarray(NEW8), X=X)
        summary[name] = {"n": len(ids), "errors": errs,
                         "wall_s": round(wall, 1),
                         "ms_per_window": round(1000 * wall / max(len(ids), 1), 1)}
        print(f"{name}: n={len(ids)} errors={errs} wall={wall:.1f}s "
              f"({summary[name]['ms_per_window']} ms/window)", flush=True)
    json.dump(summary, open(os.path.join(OUT, "extract_cost.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
