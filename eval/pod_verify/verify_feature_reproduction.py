"""LEAD B / LB1 — close the "GPU feature-extraction half is unverified" gap.

The cached evidence chain (caches/gpu/features.json) was produced on the source machine.
Everything downstream (baseline 0.810, the 26-feature view, the fused PoC) reads that cache.
This script RE-DERIVES the features from the video on THIS pod and compares them, feature by
feature, to the cached values. If the decode+feature path reproduces, the model/eval half and
the feature-extraction half are both verified; if it drifts, we say by how much and on which
features.

Protocol
  * windows taken in manifest order (labels/manifest_gold.json, READ-ONLY), first N or --ids
  * decode: eval/gpu/decode.py :: decode_window(path, start_s, dur_s, 160, 90, "rgb24")  (CPU ffmpeg)
  * features: jadal/gpu/features.py :: cpu_window_features(luma, rgb, fps=min(fps, 30.0))
  * comparison: per-feature max |recomputed - cached| and relative error, plus exact-match counts

Determinism: the CPU feature path is documented as bit-stable; this run reports the observed
tolerance rather than assuming it.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from eval.gpu.decode import decode_window          # noqa: E402
from jadal.gpu.features import cpu_window_features  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
CACHE = os.path.join(BUNDLE, "caches", "gpu", "features.json")
MANIFEST = os.path.join(BUNDLE, "labels", "manifest_gold.json")
OUT = os.path.join(ROOT, "eval", "results", "pod_verify", "feature_reproduction.json")
W, H = 160, 90


def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="how many windows to re-derive")
    ap.add_argument("--ids", default="", help="comma-separated window ids (overrides --n)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cache = json.load(open(CACHE))["windows"]
    items = json.load(open(MANIFEST))["items"]
    if args.ids:
        want = [i.strip() for i in args.ids.split(",") if i.strip()]
        items = [it for it in items if it["id"] in want]
    else:
        # spread across clips so a single bad clip cannot explain the result
        by_clip = {}
        for it in items:
            by_clip.setdefault(it["clip"], []).append(it)
        picked, i = [], 0
        while len(picked) < args.n and any(len(v) > i for v in by_clip.values()):
            for c in sorted(by_clip):
                if len(by_clip[c]) > i and len(picked) < args.n:
                    picked.append(by_clip[c][i])
            i += 1
        items = picked

    rows, per_feat, t0 = [], {}, time.time()
    for it in items:
        wid = it["id"]
        if wid not in cache:
            rows.append({"id": wid, "status": "not_in_cache"})
            continue
        cached = cache[wid].get("full") or {}
        try:
            rgb, fps = decode_window(it["path"], it.get("start_s", 0.0),
                                     round(it["end_s"] - it["start_s"], 2), W, H, "rgb24")
            luma = luma_of(rgb)
            fresh = cpu_window_features(luma, rgb, fps=min(fps, 30.0))
        except Exception as e:  # keep negative results
            rows.append({"id": wid, "clip": it["clip"], "status": "error", "error": str(e)[:200]})
            continue
        diffs, rels = {}, {}
        for k, v in fresh.items():
            if k not in cached:
                continue
            d = abs(float(v) - float(cached[k]))
            diffs[k] = d
            rels[k] = d / max(abs(float(cached[k])), 1e-12)
            per_feat.setdefault(k, []).append(d)
        maxd = max(diffs.values()) if diffs else float("nan")
        worst = max(diffs, key=diffs.get) if diffs else None
        rows.append({
            "id": wid, "clip": it["clip"], "label": it["label"], "status": "ok",
            "frames": int(rgb.shape[0]), "fps": round(float(fps), 4),
            "cache_frames": cache[wid].get("frames"),
            "n_features_compared": len(diffs),
            "max_abs_diff": maxd, "worst_feature": worst,
            "worst_abs_diff": diffs.get(worst) if worst else None,
            "exact_matches": int(sum(1 for d in diffs.values() if d == 0.0)),
            "per_feature_abs_diff": {k: diffs[k] for k in sorted(diffs)},
        })
        print(f"{wid:>6s} {it['clip']:<12s} frames {rgb.shape[0]:>3d} (cache {cache[wid].get('frames')}) "
              f"max|d|={maxd:.3e} worst={worst}", flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {}
    for k, v in per_feat.items():
        summary[k] = {"max_abs_diff": float(np.max(v)), "mean_abs_diff": float(np.mean(v)),
                      "n": len(v)}
    out = {
        "experiment": "LB1-feature-reproduction",
        "verified_by": "lead-b",
        "claim_under_test": ("caches/gpu/features.json reproduces from the source video on this pod, "
                            "which would close the 'GPU feature-extraction half is unverified' gap"),
        "n_windows": len(rows), "n_ok": len(ok), "n_error": len(rows) - len(ok),
        "label_source": "human gold IndEgo (labels/manifest_gold.json, READ-ONLY)",
        "protocol": ("decode_window(160x90 rgb24, CPU ffmpeg) -> cpu_window_features(fps=min(fps,30)) "
                     "vs cached caches/gpu/features.json['windows'][id]['full']; per-feature abs diff"),
        "device": "RunPod A40 pod, CPU feature path (CUDA visible but not used by cpu_window_features)",
        "warmup_note": "windows picked one per clip round-robin so a single clip cannot explain the result",
        "per_feature": summary,
        "windows": rows,
        "global": {
            "max_abs_diff_all": float(np.max([r["max_abs_diff"] for r in ok])) if ok else None,
            "mean_max_abs_diff": float(np.mean([r["max_abs_diff"] for r in ok])) if ok else None,
            "all_exact": bool(ok) and all(r["exact_matches"] == r["n_features_compared"] for r in ok),
            "elapsed_s": round(time.time() - t0, 2),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    g = out["global"]
    print(f"\nwindows ok={len(ok)}/{len(rows)}  max|d| across all = {g['max_abs_diff_all']}  "
          f"mean per-window max|d| = {g['mean_max_abs_diff']}  all_exact={g['all_exact']}")
    print("wrote", os.path.realpath(args.out))


if __name__ == "__main__":
    main()
