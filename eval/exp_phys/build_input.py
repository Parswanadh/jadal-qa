"""Build the frozen, committed input set for the Q3 determinism proof.

Sources (all read-only, real data):
  gold (150 windows): /workspace/jadal_pod_bundle/caches/perpixel/
      - physics_feats.npz  -> hist_ent, ten_norm
      - theory_feats.npz   -> blur_norm (from the per-window float block)
      - gold.npz           -> y (human label), clips (LOCO group)
  ood  (30 windows):   /tmp/opencode/ood_feats.json, produced ONCE by
      eval/exp_phys/extract_ood_feats.py on branch b/ood-main
      (jadal.perpixel.{series,physics,theory}) over the local
      Egocentric-10K label_windows mp4s in the pod bundle.

Output: eval/exp_phys/input_windows.json  (features only; no frames, no video).

The rule scorer NEVER reads the labels; they are carried for context and for
the run summary (how many gold/OOD windows the rule gate rejects), never for
tuning (the rule table is frozen from configs/perpixel_v1.json).
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
B = "/workspace/jadal_pod_bundle"
CACHE = os.path.join(B, "caches", "perpixel")
OOD_FEATS = "/tmp/opencode/ood_feats.json"
OUT = os.path.join(HERE, "input_windows.json")

FEATURES = ("hist_ent", "ten_norm", "blur_norm")


def _round4(x: float) -> float:
    return round(float(x), 4)


def load_gold() -> list[dict]:
    ph = np.load(os.path.join(CACHE, "physics_feats.npz"), allow_pickle=True)
    th = np.load(os.path.join(CACHE, "theory_feats.npz"), allow_pickle=True)
    gd = np.load(os.path.join(CACHE, "gold.npz"), allow_pickle=True)
    ph_ids = [str(i) for i in ph["ids"]]
    ph_names = [str(n) for n in ph["names"]]
    ph_ix = {n: k for k, n in enumerate(ph_names)}
    ph_row = {str(i): k for k, i in enumerate(ph_ids)}
    th_ids = [str(i) for i in th["ids"]]
    th_row = {str(i): k for k, i in enumerate(th_ids)}
    th_float = {}
    for meta in th["meta"]:
        d = json.loads(str(meta))
        th_float[d["id"]] = d["float"]
    gd_ids = [str(i) for i in gd["ids"]]
    y = {str(i): int(v) for i, v in zip(gd_ids, gd["y"])}
    clips = {str(i): str(c) for i, c in zip(gd_ids, gd["clips"])}
    out = []
    for wid in ph_ids:
        row = ph["X"][ph_row[wid]]
        out.append({
            "window_id": wid,
            "split": "gold",
            "label": y[wid],
            "clip": clips[wid],
            "features": {
                "hist_ent": _round4(row[ph_ix["hist_ent"]]),
                "ten_norm": _round4(row[ph_ix["ten_norm"]]),
                "blur_norm": _round4(th_float[wid]["blur_norm"]),
            },
        })
    return out


def load_ood() -> list[dict]:
    d = json.load(open(OOD_FEATS))
    out = []
    for w in d["windows"]:
        out.append({
            "window_id": w["window_id"],
            "split": "ood",
            "label": w["label"],
            "clip": w.get("factory") or "ood",
            "features": {k: _round4(w["features"][k]) for k in FEATURES},
        })
    return out


def main() -> None:
    gold = load_gold()
    ood = load_ood()
    doc = {
        "description": ("Fixed input set for the Q3 deterministic-rule-path "
                        "proof: real per-window features, no frames/video."),
        "feature_units": ("hist_ent bits; ten_norm ratio (Tenengrad/noise^2); "
                          "blur_norm dimensionless (laplacian var / sigma^2)"),
        "sources": {
            "gold": "jadal_pod_bundle/caches/perpixel/{physics,theory}_feats.npz"
                    " + gold.npz (150 IndEgo windows, human labels)",
            "ood": ("30 Egocentric-10K label_windows mp4s, features extracted "
                    "once by eval/exp_phys/extract_ood_feats.py on b/ood-main"),
        },
        "n_gold": len(gold),
        "n_ood": len(ood),
        "n_windows": len(gold) + len(ood),
        "windows": gold + ood,
    }
    json.dump(doc, open(OUT, "w"), indent=1, sort_keys=False)
    print(f"wrote {OUT}: {len(gold)} gold + {len(ood)} ood = "
          f"{len(gold)+len(ood)} windows")


if __name__ == "__main__":
    main()
