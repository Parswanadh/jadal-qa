#!/usr/bin/env python3
"""GRID.01 / GRID.02 evaluation: mosaic-native features vs the 26-number baseline.

Protocol (pre-registered in experiments/GRID.01/PREREG.json and GRID.02/PREREG.json):
  * in-domain, 150 IndEgo windows (100 good / 50 bad, ONE labeller, 7 clips);
  * leave-one-clip-out (LOCO) with StandardScaler + LogisticRegression(C=0.3,
    class_weight='balanced') — the shipped head recipe, nothing selected on the
    held-out clip;
  * pooled AUROC with clip-bootstrap 95 % CI (2000 draws, seed 0);
  * paired clip-bootstrap for every A-vs-B AUROC difference;
  * mandatory negative controls that MUST print ~0.5.

Kill-gate (pre-registered): after GRID.02, if the intertile group adds less than
0.02 nested-LOCO AUROC over spatial-only, the mosaic-as-decision-input line stops.

Run: JADAL_BUNDLE=<bundle> python3 eval/exp_grid/run_grid.py
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
from jadal.mosaic import features as F  # noqa: E402

RESULTS = os.path.join(ROOT, "eval", "results", "exp_grid")
SEED, N_BOOT = 0, 2000
KILL_GATE_DELTA = 0.02


def load_features(tag):
    p = os.path.join(RESULTS, f"features_{tag}.json")
    d = json.load(open(p))
    return d


def matrix(feats_doc, ids, names, fill=np.nan):
    W = feats_doc["windows"]
    X = np.full((len(ids), len(names)), fill, float)
    for i, wid in enumerate(ids):
        w = W.get(wid)
        if w is None:
            continue
        for j, n in enumerate(names):
            v = w.get(n)
            X[i, j] = np.nan if v is None else float(v)
    return X


def group_names(doc, prefix):
    return [n for n in doc["feature_names"] if n.startswith(prefix)]


def evaluate(X, y, clip, label, extra=None):
    p = C.loco_proba(X, y, clip, C=0.3)
    a = C.fast_auc(y, p)
    lo, hi, nb = C.group_boot_ci(y, p, clip, n_boot=N_BOOT, seed=SEED)
    wlo, whi, _ = C.window_boot_ci(y, p, n_boot=N_BOOT, seed=SEED)
    thr, ba = C.youden_threshold(y, p)
    out = {"label": label, "n_features": X.shape[1], "auroc": round(a, 4),
           "ci95": [round(lo, 3), round(hi, 3)],
           "ci95_type": f"clip-level bootstrap ({N_BOOT} draws, seed {SEED})",
           "ci95_window_optimistic": [round(wlo, 3), round(whi, 3)],
           "bal_acc_at_0.5": round(C.balanced_accuracy(y, (p >= 0.5).astype(int)), 4),
           "youden_threshold_in_sample": round(thr, 4),
           "bal_acc_youden_in_sample": round(ba, 4)}
    if extra:
        out.update(extra)
    out["_p"] = p
    return out


def main():
    gold = C.load_gold()
    ids, y, clip = gold["ids"], gold["y"], gold["clip"]
    doc = load_features("m3x3_t224")
    names_all = doc["feature_names"]
    n_sp = len(group_names(doc, "sp_"))
    n_it = len(group_names(doc, "it_"))
    n_dm = len(group_names(doc, "dm_"))
    print(f"features: spatial {n_sp}, intertile {n_it}, diff {n_dm}, total {len(names_all)}")

    # ---------------- baseline (same folds, recomputed here) ----------------
    base = evaluate(gold["X"], y, clip, "baseline_26_shipped_features")
    print(f"baseline 26 features           AUROC {base['auroc']} {base['ci95']}")

    # ---------------- GRID.01: spatial only ----------------
    sp_names = group_names(doc, "sp_")
    sp_mean_names = [n for n in sp_names if n.endswith("_mean")]
    X_sp = matrix(doc, ids, sp_names)
    X_spm = matrix(doc, ids, sp_mean_names)
    r_sp = evaluate(X_sp, y, clip, "spatial_pooled_meanminmaxstd")
    r_spm = evaluate(X_spm, y, clip, "spatial_mean_only_capacity_control")
    p_sp, p_spm, p_base = r_sp["_p"], r_spm["_p"], base["_p"]
    d_spm = C.paired_group_boot_diff(y, p_sp, p_spm, clip, n_boot=N_BOOT, seed=SEED)
    d_base = C.paired_group_boot_diff(y, p_sp, p_base, clip, n_boot=N_BOOT, seed=SEED)
    print(f"GRID.01 spatial pooled         AUROC {r_sp['auroc']} {r_sp['ci95']}")
    print(f"GRID.01 spatial mean-only      AUROC {r_spm['auroc']} {r_spm['ci95']}  "
          f"delta {d_spm['delta']:+.3f} [{d_spm['lo']:+.3f},{d_spm['hi']:+.3f}]")
    print(f"GRID.01 vs baseline 26         delta {d_base['delta']:+.3f} "
          f"[{d_base['lo']:+.3f},{d_base['hi']:+.3f}] excludes0={d_base['excludes_zero']}")

    # ---------------- negative controls (MUST be ~0.5) ----------------
    rng = np.random.RandomState(SEED)
    shuff = []
    for s in range(20):
        r = np.random.RandomState(1000 + s)
        ys = y[r.permutation(len(y))]
        shuff.append(C.fast_auc(ys, C.loco_proba(X_sp, ys, clip, C=0.3)))
    rand = []
    for s in range(20):
        Xr = rng.normal(size=X_sp.shape)
        rand.append(C.fast_auc(y, C.loco_proba(Xr, y, clip, C=0.3)))
    null_X = np.zeros((len(ids), 4))          # flat-grey null mosaic: identical rows
    null_auc = C.fast_auc(y, C.loco_proba(null_X, y, clip, C=0.3))
    controls = {"shuffled_labels_mean_auroc": round(float(np.mean(shuff)), 4),
                "shuffled_labels_max_auroc": round(float(np.max(shuff)), 4),
                "random_gaussian_mean_auroc": round(float(np.mean(rand)), 4),
                "random_gaussian_max_auroc": round(float(np.max(rand)), 4),
                "null_grey_mosaic_auroc": round(float(null_auc), 4),
                "passes": bool(np.mean(shuff) < 0.60 and np.mean(rand) < 0.60
                               and abs(null_auc - 0.5) < 0.05)}
    print("controls:", json.dumps(controls))

    # ---------------- GRID.01 results ----------------
    cfg01 = {"grid": [3, 3], "n_tiles": 9, "tile_px": 224, "timestamps": True,
             "order": "row-major time", "groups": ["spatial"], "pools": list(F.POOLS),
             "model": "StandardScaler+LogisticRegression(C=0.3,class_weight=balanced)",
             "folds": "leave-one-clip-out over 7 clips", "seed": SEED,
             "primary": "spatial_pooled_meanminmaxstd", "secondary": "spatial_mean_only",
             "kill_gate_delta": KILL_GATE_DELTA}
    for r in (r_sp, r_spm, base):
        r.pop("_p", None)
    C.save_results(os.path.join(RESULTS, "GRID.01.json"), {
        "id": "GRID.01", "n": int(len(y)), "n_bad": int((y == 0).sum()),
        "label_source": gold["label_source"],
        "protocol": "LOCO over 7 IndEgo clips, shipped head recipe, nothing selected on held-out clips",
        "primary": {**r_sp, "delta_vs_baseline26": d_base},
        "secondary_mean_pooling": {**r_spm, "delta_vs_primary": d_spm},
        "baseline_26_on_same_folds": base,
        "negative_controls": controls,
        "ci95": r_sp["ci95"], "ci95_type": r_sp["ci95_type"],
        "device": C.DEVICE_CPU,
        "cost": f"mosaic build+features {doc['ms_per_window_per_config']} ms/window (CPU)",
        "command": "JADAL_BUNDLE=<bundle> python3 eval/exp_grid/run_grid.py",
        "git_sha": C.git_sha(), "seed": SEED,
        "config_sha256": C.config_sha256(cfg01), "config": cfg01,
        "feature_digest_sha256": doc["feature_digest_sha256"],
        "notes": "EmbeddingGemma-2 mosaic reference on this layout was 0.785 (E2.01, embedding "
                 "model, optional arm); the 26-number baseline on the same folds is in this file.",
    })

    # ---------------- GRID.02: add intertile, leave-one-group-out ----------------
    it_names = group_names(doc, "it_")
    dm_names = group_names(doc, "dm_")
    X_spit = matrix(doc, ids, sp_names + it_names)
    r_spit = evaluate(X_spit, y, clip, "spatial+intertile")
    p_spit = r_spit["_p"]
    d_it = C.paired_group_boot_diff(y, p_spit, p_sp, clip, n_boot=N_BOOT, seed=SEED)
    gate = bool(r_spit["auroc"] - r_sp["auroc"] >= KILL_GATE_DELTA)
    print(f"GRID.02 spatial+intertile      AUROC {r_spit['auroc']} {r_spit['ci95']}  "
          f"delta {d_it['delta']:+.3f} [{d_it['lo']:+.3f},{d_it['hi']:+.3f}] "
          f"excludes0={d_it['excludes_zero']}")
    print(f"KILL-GATE (delta >= {KILL_GATE_DELTA}): {'PASS -> GRID.03/04 may run' if gate else 'FAIL -> stop mosaic-as-decision-input'}")

    # per-group ablation
    abl = {}
    for tag, nm in (("spatial_only", sp_names), ("spatial+intertile", sp_names + it_names)):
        Xa = matrix(doc, ids, nm)
        pa = C.loco_proba(Xa, y, clip, C=0.3)
        abl[tag] = {"auroc": round(C.fast_auc(y, pa), 4), "n_features": len(nm)}
    # single intertile features (orientation-free pooled AUC)
    Xit = matrix(doc, ids, it_names)
    singles = sorted(((round(max(C.fast_auc(y, Xit[:, j]), 1 - C.fast_auc(y, Xit[:, j])), 3), n)
                      for j, n in enumerate(it_names)), reverse=True)
    for r in (r_spit, r_sp, base, r_spm):
        r.pop("_p", None)
    cfg02 = {**cfg01, "groups": ["spatial", "intertile"], "kill_gate_delta": KILL_GATE_DELTA}
    C.save_results(os.path.join(RESULTS, "GRID.02.json"), {
        "id": "GRID.02", "n": int(len(y)), "n_bad": int((y == 0).sum()),
        "label_source": gold["label_source"],
        "protocol": "LOCO over 7 clips; leave-one-group-out ablation of the intertile group",
        "primary": {**r_spit, "delta_vs_spatial_only": d_it},
        "spatial_only": r_sp,
        "baseline_26_on_same_folds": base,
        "per_group_ablation": abl,
        "top_single_intertile_features_pooled_auc": singles[:6],
        "kill_gate": {"delta_required": KILL_GATE_DELTA, "observed": round(d_it["delta"], 4),
                      "passed": gate},
        "negative_controls": controls,
        "ci95": r_spit["ci95"], "ci95_type": r_spit["ci95_type"],
        "device": C.DEVICE_CPU,
        "cost": f"mosaic build+features {doc['ms_per_window_per_config']} ms/window (CPU)",
        "command": "JADAL_BUNDLE=<bundle> python3 eval/exp_grid/run_grid.py",
        "git_sha": C.git_sha(), "seed": SEED,
        "config_sha256": C.config_sha256(cfg02), "config": cfg02,
        "feature_digest_sha256": doc["feature_digest_sha256"],
    })
    print("wrote GRID.01.json and GRID.02.json")

    # ---------------- GRID.03 as the post-kill-gate small control ----------------
    X_all = matrix(doc, ids, sp_names + it_names + dm_names)
    r_all = evaluate(X_all, y, clip, "spatial+intertile+diff")
    d_dm = C.paired_group_boot_diff(y, r_all["_p"], p_spit, clip, n_boot=N_BOOT, seed=SEED)
    print(f"GRID.03 +diff group (control)   AUROC {r_all['auroc']} {r_all['ci95']}  "
          f"delta {d_dm['delta']:+.3f} [{d_dm['lo']:+.3f},{d_dm['hi']:+.3f}] "
          f"excludes0={d_dm['excludes_zero']}")
    p_all = r_all["_p"]

    # ---------------- canvas controls: mask cost, text leak, arrangement -------
    ctrl = {}
    arms = {"content_reference_no_ts_no_mask": "m3x3_t224_nots",
            "text_visible_not_masked": "m3x3_t224_leak",
            "tile_order_shuffled": "m3x3_t224_shuf"}
    ps = {}
    for label, tag in arms.items():
        try:
            d2 = load_features(tag)
        except FileNotFoundError:
            continue
        nm = [n for n in d2["feature_names"] if n in set(names_all)]
        X2 = matrix(d2, ids, nm)
        p2 = C.loco_proba(X2, y, clip, C=0.3)
        ps[label] = p2
        ctrl[label] = {"arm_tag": tag, "n_features": len(nm),
                       "auroc": round(C.fast_auc(y, p2), 4),
                       "delta_vs_primary_all_groups": round(C.fast_auc(y, p2) - r_all["auroc"], 4)}
    if "content_reference_no_ts_no_mask" in ps and "text_visible_not_masked" in ps:
        d_leak = C.paired_group_boot_diff(y, ps["text_visible_not_masked"],
                                          ps["content_reference_no_ts_no_mask"], clip,
                                          n_boot=N_BOOT, seed=SEED)
        ctrl["TEXT_LEAK_test"] = {
            "comparison": "timestamps drawn and NOT masked  vs  no timestamps and no mask",
            "delta_auroc": round(d_leak["delta"], 4), "ci95": [round(d_leak["lo"], 3), round(d_leak["hi"], 3)],
            "excludes_zero": d_leak["excludes_zero"],
            "verdict": ("text leaks into the features" if d_leak["excludes_zero"] and d_leak["delta"] > 0
                        else "no measurable text leak")}
        d_mask = C.paired_group_boot_diff(y, p_all, ps["content_reference_no_ts_no_mask"], clip,
                                          n_boot=N_BOOT, seed=SEED)
        ctrl["MASK_COST_test"] = {
            "comparison": "primary (timestamps drawn, masked)  vs  content reference (no mask)",
            "delta_auroc": round(d_mask["delta"], 4), "ci95": [round(d_mask["lo"], 3), round(d_mask["hi"], 3)],
            "excludes_zero": d_mask["excludes_zero"]}
    if "tile_order_shuffled" in ps:
        ctrl["ARRANGEMENT_test"] = {
            "comparison": "primary vs shuffled tile placement",
            "note": "the pooled feature set is arrangement-invariant by construction, so this "
                    "control is expected to be a no-op and is reported as the answer to "
                    "'does tile layout matter', not as a leak test",
            "delta_vs_primary_all_groups": ctrl["tile_order_shuffled"]["delta_vs_primary_all_groups"]}
    print("canvas controls:", json.dumps(ctrl, indent=1)[:900])

    for r in (r_all, r_spit, r_sp, r_spm, base):
        r.pop("_p", None)
    cfg03 = {**cfg01, "groups": ["spatial", "intertile", "diff"]}
    C.save_results(os.path.join(RESULTS, "GRID.03.json"), {
        "id": "GRID.03", "n": int(len(y)), "n_bad": int((y == 0).sum()),
        "label_source": gold["label_source"],
        "protocol": "LOCO over 7 clips; difference-mosaic group added on top of spatial+intertile",
        "primary": {**r_all, "delta_vs_spatial_plus_intertile": d_dm},
        "spatial_plus_intertile": r_spit,
        "baseline_26_on_same_folds": base,
        "kill_gate_upstream": {"passed": gate, "note": "GRID.03 ran only as the small "
                               "post-kill-gate control permitted by the brief"},
        "canvas_controls": ctrl,
        "ci95": r_all["ci95"], "ci95_type": r_all["ci95_type"],
        "device": C.DEVICE_CPU,
        "cost": f"mosaic build+features {doc['ms_per_window_per_config']} ms/window (CPU)",
        "command": "JADAL_BUNDLE=<bundle> python3 eval/exp_grid/run_grid.py",
        "git_sha": C.git_sha(), "seed": SEED,
        "config_sha256": C.config_sha256(cfg03), "config": cfg03,
        "feature_digest_sha256": doc["feature_digest_sha256"],
    })
    print("wrote GRID.01.json, GRID.02.json, GRID.03.json")

    with open(os.path.join(RESULTS, "controls.json"), "w") as f:
        json.dump({"negative_controls": controls, "canvas_controls": ctrl,
                   "feature_digest_sha256": doc["feature_digest_sha256"],
                   "git_sha": C.git_sha()}, f, indent=1)
    return 0 if controls["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
