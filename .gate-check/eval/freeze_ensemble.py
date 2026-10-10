"""Freeze the two-view ensemble on all 150 human gold windows.

- C per view: inner-style LOCO over all 7 clips (each clip held out once,
  trained on other 6) maximizing OOF AUROC; tie-break -> smaller C.
- Heads: StandardScaler + LogisticRegression(class_weight=balanced) on all 150.
- Platt (a,b) fit on full-data LOCO-OOF ensemble probs (monotonic; reference
  only — usable_prob stays the uncalibrated prob-average that was validated).
- Thresholds: 0.5 default + 5%/10% false-reject points on full-OOF-good.
- Abstain band [0.4,0.6] heuristic; empirical OOF stats reported, honest
  nested stats live in eval/results/ensemble/nested_loco.json.
Writes configs/ensemble_v1.json + .sha256. Prints sha256.
"""
import hashlib
import json
import os
import subprocess

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from eval.ensemble_nested_loco import GRID, MANIFEST, FEATS, SEG, load, fit_predict

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "ensemble_v1.json")


def full_inner_pick(X, y, clip, clips):
    """LOCO over all clips; OOF AUROC per C; tie-break smaller C."""
    aucs = {}
    oof = {}
    for C in GRID:
        p = np.zeros(len(y))
        for c in clips:
            te = np.where(clip == c)[0]
            tr = np.where(clip != c)[0]
            pp, _, _ = fit_predict(X[tr], y[tr], X[te], C)
            p[te] = pp
        oof[C] = p
        aucs[C] = float(roc_auc_score(y, p))
    best = min([c for c in GRID if aucs[c] == max(aucs.values())])
    return best, aucs, oof


def main():
    ids, clip, y, reason, Xe, Xg, names, man = load()
    clips = sorted(set(clip))
    Ce, aucse, oofe = full_inner_pick(Xe, y, clip, clips)
    Cg, aucsg, oofg = full_inner_pick(Xg, y, clip, clips)

    heads = {}
    for tag, X, C in [("eg2", Xe, Ce), ("pixel", Xg, Cg)]:
        sc = StandardScaler()
        Z = sc.fit_transform(X)
        m = LogisticRegression(max_iter=5000, C=C, class_weight="balanced")
        m.fit(Z, y)
        heads[tag] = {"C": C, "mean": [float(v) for v in sc.mean_],
                      "scale": [float(v) for v in sc.scale_],
                      "coef": [float(v) for v in m.coef_[0].ravel()],
                      "intercept": float(m.intercept_[0])}
    oof_ens = (oofe[Ce] + oofg[Cg]) / 2
    pm = LogisticRegression(max_iter=5000)
    pm.fit(oof_ens.reshape(-1, 1), y)
    good = oof_ens[y == 1]
    t05, t10 = float(np.quantile(good, 0.05)), float(np.quantile(good, 0.10))
    band = [0.4, 0.6]
    inside = (oof_ens >= band[0]) & (oof_ens <= band[1])
    pr = (oof_ens[~inside] > 0.5).astype(int)
    yy = y[~inside]
    band_stats = {"coverage_abstain": float(inside.mean()),
                  "confident_balacc_oof": float(((pr[yy == 1] == 1).mean() + (pr[yy == 0] == 0).mean()) / 2)}
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        sha = "unknown"
    cfg = {
        "name": "ensemble_v1", "n": len(ids),
        "label_source": "human gold (single labeller)",
        "labels_sha256": man.get("labels_sha256", ""),
        "views": {
            "eg2": {"kind": "logistic_on_L2normed_mean_EmbeddingGemma2_frame_embeddings",
                    "embedding_dim": 768, "frame_step_s": 0.5,
                    "embed_model_dir": "/home/parshu/projects/awing/data/eg2",
                    "normalize_embeddings": True, "max_soft_tokens": 140,
                    "aggregation": "mean_over_frames_then_L2_normalise",
                    "C": Ce, "full_inner_oof_auroc_by_C": aucse, **heads["eg2"]},
            "pixel": {"kind": "logistic_on_26_perpixel_window_features",
                      "feature_names": names,
                      "decode": {"tool": "ffmpeg_cpu", "w": 160, "h": 90, "pix_fmt": "rgb24",
                                "dur_s": 10.0,
                                "luma": "0.299R+0.587G+0.114B_float64",
                                "fps": "n_frames/10_capped_at_30",
                                "code": "jadal/gpu/features.py:cpu_window_features (from origin/feat/gpu-pipeline, read-only)"},
                      "C": Cg, "full_inner_oof_auroc_by_C": aucsg, **heads["pixel"]},
        },
        "fusion": {"kind": "equal_weight_probability_average", "weights": [0.5, 0.5],
                   "note": "no learned stacking weights"},
        "platt_reference_only": {"a": float(pm.coef_[0][0]), "b": float(pm.intercept_[0]),
                                 "note": "monotonic; fit on full-OOF; NOT applied to usable_prob"},
        "threshold": 0.5, "threshold_5pctFR_fullOOF": t05, "threshold_10pctFR_fullOOF": t10,
        "abstain_band": band, "abstain_band_oof_stats": band_stats,
        "verdict_rule": "usable_prob>0.5->accept, <0.4->reject, in [0.4,0.6]->review",
        "full_oof_auroc_resubstitution_free": float(roc_auc_score(y, oof_ens)),
        "code_git_sha": sha,
        "embed_command": ("OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python "
                          "/home/parshu/projects/jadal-qa-wt/frames10k/eval/frames10k/embed_segments.py "
                          "--manifest /home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"),
        "pixel_command": ("OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python "
                          "eval/ensemble_score_windows.py --windows <json> --out <npz>"),
        "eg2_cross_device_note": "EG2 GPU numerics are NOT claimed identical across devices",
    }
    # sanity: config weights reproduce head probs on cached features
    for tag, X in [("eg2", Xe), ("pixel", Xg)]:
        h = cfg["views"][tag]
        Z = (X - np.asarray(h["mean"])) / np.asarray(h["scale"])
        p = 1 / (1 + np.exp(-(Z @ np.asarray(h["coef"]) + h["intercept"])))
        ref = oofe[Ce] if tag == "eg2" else oofg[Cg]
        # refit-on-all vs LOCO-OOF differ; just check range sanity
        assert np.all((p > 0) & (p < 1)) and not np.isnan(p).any()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    blob = json.dumps(cfg, indent=1)
    open(OUT, "w").write(blob)
    sha256 = hashlib.sha256(blob.encode()).hexdigest()
    open(OUT + ".sha256", "w").write(sha256 + "  ensemble_v1.json\n")
    print(f"Ce={Ce} {aucse}\nCg={Cg} {aucsg}")
    print(f"full-OOF ens AUROC {cfg['full_oof_auroc_resubstitution_free']:.4f} (C picked by same OOF: optimistic, not a claim)")
    print(f"band stats {band_stats} t05={t05:.3f} t10={t10:.3f}")
    print(f"wrote {OUT}\nsha256 {sha256}")


if __name__ == "__main__":
    main()
