#!/usr/bin/env python3
"""GRID.04 — compact post-kill-gate control ladder (frame count, layout, tile px).

The GRID.02 kill-gate failed, so this is the SMALL control the brief permits, not
the full sweep.  It answers the customer's headline configuration question
(12 frames in a 4x3 grid) and reports the AUROC-vs-canvas-pixels Pareto.

Every arm uses the identical feature definition (spatial+intertile+diff), the
identical head recipe and the identical folds; arms are compared with paired
clip-bootstrap intervals on the same 150 windows.

Run: JADAL_BUNDLE=<bundle> python3 eval/exp_grid/run_grid04.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))

import common as C  # noqa: E402
from eval.exp_grid.run_grid import matrix, load_features  # noqa: E402

RESULTS = os.path.join(ROOT, "eval", "results", "exp_grid")
SEED, N_BOOT = 0, 2000
REFERENCE = "m3x3_t224"          # the GRID.01/02 arm
ARMS = ["m2x2_t224", "m3x3_t224", "m4x3_t224", "m4x4_t224", "m4x3_t160", "m4x3_t112"]


def canvas_px(rows, cols, tile_px):
    return (cols * tile_px + (cols + 1) * 2) * (rows * tile_px + (rows + 1) * 2)


def main():
    gold = C.load_gold()
    ids, y, clip = gold["ids"], gold["y"], gold["clip"]
    base_p = C.loco_proba(gold["X"], y, clip, C=0.3)
    base_auc = C.fast_auc(y, base_p)

    rows = []
    probs = {}
    for tag in ARMS:
        try:
            doc = load_features(tag)
        except FileNotFoundError:
            print(f"skip {tag}: not extracted")
            continue
        X = matrix(doc, ids, doc["feature_names"])
        p = C.loco_proba(X, y, clip, C=0.3)
        probs[tag] = p
        lo, hi, _ = C.group_boot_ci(y, p, clip, n_boot=N_BOOT, seed=SEED)
        cfg = doc["config"]
        r, c = cfg["grid"]
        rows.append({
            "tag": tag, "n_tiles": cfg["n"], "grid": [r, c], "tile_px": cfg["tile_px"],
            "timestamps": cfg["timestamps"], "n_features": len(doc["feature_names"]),
            "canvas_px": canvas_px(r, c, cfg["tile_px"]),
            "auroc": round(C.fast_auc(y, p), 4), "ci95": [round(lo, 3), round(hi, 3)],
            "ms_per_window": doc.get("ms_per_window_per_config"),
            "feature_digest_sha256": doc["feature_digest_sha256"],
        })
    for r in rows:
        tag = r["tag"]
        d_ref = C.paired_group_boot_diff(y, probs[tag], probs[REFERENCE], clip, n_boot=N_BOOT, seed=SEED)
        d_base = C.paired_group_boot_diff(y, probs[tag], base_p, clip, n_boot=N_BOOT, seed=SEED)
        r["vs_n9_paired"] = {"delta": round(d_ref["delta"], 4),
                             "ci95": [round(d_ref["lo"], 3), round(d_ref["hi"], 3)],
                             "excludes_zero": d_ref["excludes_zero"]}
        r["vs_baseline26_paired"] = {"delta": round(d_base["delta"], 4),
                                     "ci95": [round(d_base["lo"], 3), round(d_base["hi"], 3)],
                                     "excludes_zero": d_base["excludes_zero"]}
        print(f"{tag:12s} N={r['n_tiles']:2d} {r['grid']} t{r['tile_px']:3d} "
              f"AUROC {r['auroc']:.4f} {r['ci95']}  vs N9 {d_ref['delta']:+.3f} "
              f"[{d_ref['lo']:+.3f},{d_ref['hi']:+.3f}]  vs base26 {d_base['delta']:+.3f}")

    ok = [r for r in rows if r["tag"] in probs]
    headline = next((r for r in ok if r["tag"] == "m4x3_t224"), None)
    best = max(ok, key=lambda r: r["auroc"]) if ok else None
    tail_frac = 0.20
    pareto = []
    for r in sorted(ok, key=lambda r: r["canvas_px"]):
        dominated = any((o["canvas_px"] <= r["canvas_px"] and o["auroc"] >= r["auroc"]
                         and o is not r) for o in ok)
        if not dominated:
            pareto.append(r["tag"])

    cfg = {"arms": ARMS, "reference": REFERENCE, "groups": "spatial+intertile+diff",
           "model": "StandardScaler+LogisticRegression(C=0.3,class_weight=balanced)",
           "folds": "leave-one-clip-out over 7 clips", "seed": SEED,
           "why": "post-kill-gate small control permitted by the brief; the full sweep is not run"}
    C.save_results(os.path.join(RESULTS, "GRID.04.json"), {
        "id": "GRID.04", "n": int(len(y)), "n_bad": int((y == 0).sum()),
        "label_source": gold["label_source"],
        "protocol": "LOCO over 7 clips, identical feature definition and head recipe per arm; "
                    "paired clip-bootstrap comparisons on the same windows",
        "arms": rows, "pareto_tags": pareto,
        "headline_4x3_12_frames": headline, "best_arm": best,
        "baseline_26_on_same_folds": {"auroc": round(base_auc, 4)},
        "gamma": {"note": "the full GRID.04 sweep (N in {4,6,9,12,16} x tile in {112,160,224} x "
                          "uniform vs motion-adaptive selection x timestamps on/off) was NOT run "
                          "because the GRID.02 kill-gate failed", "tail_frac": tail_frac},
        "ci95": headline["ci95"] if headline else None,
        "ci95_type": f"clip-level bootstrap ({N_BOOT} draws, seed {SEED})",
        "device": C.DEVICE_CPU, "cost": "mosaic build+features ~195 ms/window/arm (CPU), 0 GPU",
        "command": "JADAL_BUNDLE=<bundle> python3 eval/exp_grid/run_grid04.py",
        "git_sha": C.git_sha(), "seed": SEED, "config_sha256": C.config_sha256(cfg), "config": cfg,
    })
    print("wrote", os.path.join(RESULTS, "GRID.04.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
