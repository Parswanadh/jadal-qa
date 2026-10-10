"""Extract whole-window + streaming-prefix + S0-probe features for gold + synth.

CPU-only (no GPU lock). Run:
  /home/parshu/envs/jadal/bin/python eval/gpu/run_features.py
Writes eval/results/gpu/features.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from eval.gpu.decode import decode_window  # noqa: E402
from jadal.gpu.features import cpu_window_features  # noqa: E402

SPEC = os.path.join(HERE, "..", "results", "gpu", "features.json")
GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
SYNTH = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_synth.json"

W, H = 160, 90


def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def process(item):
    wid = item["id"]
    try:
        rgb, fps = decode_window(item["path"], item.get("start_s", 0.0),
                                 item.get("dur_s", 10.0), W, H, "rgb24")
    except Exception as e:
        return wid, {"error": str(e)[:200]}
    luma = luma_of(rgb)
    T = rgb.shape[0]
    out = {"frames": T, "fps": fps, "clip": item.get("clip"),
           "label": item.get("label"), "degradation": item.get("degradation")}
    out["full"] = cpu_window_features(luma, rgb, fps=min(fps, 30.0))
    # streaming prefixes: cumulative 2-s chunks
    chunk = max(int(round(2.0 * fps)), 1)
    prefixes = []
    for end in range(chunk, T + 1, chunk):
        prefixes.append(cpu_window_features(luma[:end], rgb[:end], fps=min(fps, 30.0)))
    out["prefix"] = prefixes
    # S0 probe: first 2 s at tiny res (gray 80x45) — cheap early signals
    try:
        g2, _ = decode_window(item["path"], item.get("start_s", 0.0), 2.0,
                              80, 45, "gray")
        out["s0"] = {
            "mean": float(g2.mean()), "p5": float(__import__("numpy").percentile(g2, 5)),
            "p95": float(__import__("numpy").percentile(g2, 95)),
            "spatial_var": float(g2[:10].mean(axis=0).var()),
            "temp_e": float(abs(__import__("numpy").diff(g2, axis=0)).mean()),
        }
    except Exception as e:
        out["s0"] = {"error": str(e)[:200]}
    return wid, out


def main(jobs=8, limit=None):
    gold = json.load(open(GOLD))["items"]
    synth = json.load(open(SYNTH))["items"]
    items = []
    for it in gold:
        items.append({"id": it["id"], "path": it["path"],
                      "start_s": it["start_s"],
                      "dur_s": round(it["end_s"] - it["start_s"], 2),
                      "clip": it["clip"], "label": it["label"]})
    for it in synth:
        items.append({"id": it["id"], "path": it["path"], "start_s": 0.0,
                      "dur_s": 10.0, "clip": "synth",
                      "label": it["label"], "degradation": it.get("degradation")})
    if limit:
        items = items[:limit]
    t0 = time.perf_counter()
    with mp.Pool(min(jobs, len(items))) as pool:
        rows = pool.map(process, items)
    res = {"windows": dict(rows), "wall_s": round(time.perf_counter() - t0, 1),
           "n": len(rows), "size": [W, H]}
    os.makedirs(os.path.dirname(SPEC), exist_ok=True)
    json.dump(res, open(SPEC, "w"))
    errs = sum(1 for _, r in rows if "error" in r)
    print(f"wrote {SPEC}: n={len(rows)} errors={errs} wall={res['wall_s']}s")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
