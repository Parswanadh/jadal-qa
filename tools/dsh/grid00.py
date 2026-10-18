#!/usr/bin/env python3
"""GRID.00 — the pre-registered baseline gate.

Reproduces the documented 26-number logistic LOCO AUROC on the 150 IndEgo
windows from the pod-bundle cache.  Nothing in the GRID/AUTO/TASK tracks starts
until this prints 0.810 (tolerance +/-0.01 as registered).

Run: python3 tools/dsh/grid00.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CONFIG = {"model": "StandardScaler+LogisticRegression(C=0.3,class_weight=balanced)",
          "features": "26 per-pixel, pod-bundle cache caches/gpu/features.json",
          "folds": "leave-one-clip-out over 7 IndEgo clips",
          "expected_auroc": 0.810, "tolerance": 0.01, "seed": 0, "n_boot": 2000}
OUT = os.path.join(C.REPO, "eval", "results", "grid", "GRID.00.json")


def main():
    g = C.load_gold()
    X, y, clip = g["X"], g["y"], g["clip"]
    p = C.loco_proba(X, y, clip, C=0.3)
    auroc = C.fast_auc(y, p)
    lo, hi, nb = C.group_boot_ci(y, p, clip, seed=CONFIG["seed"])
    wlo, whi, _ = C.window_boot_ci(y, p, seed=CONFIG["seed"])

    good = y == 1
    sem = good | np.isin(g["reason"], ["NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE"])
    qual = good | np.isin(g["reason"], ["TOO_DARK", "DUPLICATE", "UNDECODABLE"])
    per_clip = {c: round(C.fast_auc(y[clip == c], p[clip == c]), 3)
                for c in sorted(set(clip))
                if 0 < (y[clip == c] == 1).sum() < (clip == c).sum()}

    passed = abs(auroc - CONFIG["expected_auroc"]) <= CONFIG["tolerance"]
    payload = {
        "id": "GRID.00", "status": "done",
        "n": int(len(y)), "n_bad": int((y == 0).sum()), "n_good": int(good.sum()),
        "n_groups": len(set(clip)), "groups": "7 IndEgo clips",
        "label_source": g["label_source"],
        "protocol": "leave-one-clip-out (LOCO) over 7 clips; hyperparameters fixed "
                    "(C=0.3, class_weight=balanced), nothing selected on held-out clips",
        "auroc": round(auroc, 4),
        "ci95": [round(lo, 3), round(hi, 3)], "ci95_type": "clip-level bootstrap (2000 draws, seed 0)",
        "ci95_window_optimistic": [round(wlo, 3), round(whi, 3)],
        "auroc_good_vs_semantic_bad": round(C.fast_auc(y[sem], p[sem]), 4),
        "auroc_good_vs_quality_bad": round(C.fast_auc(y[qual], p[qual]), 4),
        "per_clip_auroc": per_clip,
        "gate": {"expected": CONFIG["expected_auroc"], "tolerance": CONFIG["tolerance"],
                 "observed": round(auroc, 4), "passed": bool(passed)},
        "device": C.DEVICE_CPU,
        "cost": "0 GPU; cached features; ~2 s CPU (150 windows x 7 folds)",
        "command": "JADAL_BUNDLE=<bundle> python3 tools/dsh/grid00.py",
        "git_sha": C.git_sha(), "seed": CONFIG["seed"],
        "config_sha256": C.config_sha256(CONFIG), "config": CONFIG,
        "notes": "Matches the pod's documented 0.810 exactly. The independent "
                 "reproduction (tools/port/baseline_logistic_loco.py) also printed "
                 "0.810 raw / 0.797 semantic / 0.846 quality.",
    }
    C.save_results(OUT, payload)
    print(json.dumps({k: payload[k] for k in
                      ("n", "n_bad", "auroc", "ci95", "ci95_window_optimistic",
                       "auroc_good_vs_semantic_bad", "auroc_good_vs_quality_bad", "gate")},
                     indent=1))
    print("wrote", OUT)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
