#!/usr/bin/env python
"""TASK.07 -- INDEPENDENT re-run of GRID.01 / GRID.02 and their kill gate.

Written by stream S5 (task/verification) from the GRID results JSONs and the
committed feature vectors.  Deliberately independent:

  * does NOT import eval/exp_grid/run_grid.py;
  * does NOT use tools/dsh/common.py for AUROC, the LOCO probabilities, the
    clip bootstrap or the paired bootstrap -- all four are re-implemented here;
  * `common.load_gold()` is imported ONLY to assert that my own derivation from
    the bundle manifest and the cached feature file agrees with it, so a bug in
    that loader cannot hide behind this verification;
  * the mosaic artifact is integrity-checked by re-implementing
    jadal/mosaic/features.py::feature_digest and checking all 150 per-window
    digests and the top-level digest.

Claims being checked (from eval/results/exp_grid/GRID.01.json / GRID.02.json):
  GRID.01 spatial (60 features)      AUROC 0.6864, clip CI [0.522, 0.772]
         paired vs the 26-feature baseline: delta -0.1236 [-0.20164, -0.07104]
  GRID.02 spatial+intertile (72)     AUROC 0.7058, clip CI [0.514, 0.785]
         paired vs spatial-only: delta +0.0194 [-0.018027, +0.053160]
  kill gate: delta >= 0.02 -> observed 0.0194 -> FAILED (mosaic stops here)

Run: JADAL_BUNDLE=<bundle> /home/parshu/envs/jadal/bin/python tools/task/verify_grid01.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))
from dsh import common  # noqa: E402  (schema writer + loader cross-check only)

ID = "TASK.07"
OUT = os.path.join(REPO, "eval", "results", "task", f"{ID}.json")
MOSAIC = os.path.join(REPO, "eval", "results", "exp_grid", "features_m3x3_t224.json")
GRID01 = os.path.join(REPO, "eval", "results", "exp_grid", "GRID.01.json")
GRID02 = os.path.join(REPO, "eval", "results", "exp_grid", "GRID.02.json")
GRID03 = os.path.join(REPO, "eval", "results", "exp_grid", "GRID.03.json")
MOSAIC_FEATURE_VERSION = "mosaic-features-1.0.0"

SEED, N_BOOT = 0, 2000
SENS_SEED, SENS_N_BOOT = 12345, 10000   # does the interval depend on the RNG stream?
C_REG, MAX_ITER = 0.3, 3000
KILL_GATE_DELTA = 0.02


# ------------------------------------------------------- my own metrics ------
def my_auc(y, s):
    """Rank AUROC with ties averaged.  Written here; cross-checked vs sklearn."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(s), float)
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def my_loco(X, y, clip, C=C_REG, class_weight="balanced", max_iter=MAX_ITER):
    """Leave-one-clip-out out-of-fold probabilities.  My own CV loop."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, float)
    y = np.asarray(y, int)
    clip = np.asarray(clip)
    assert not np.isnan(X).any(), "feature matrix has NaN -- LOCO would be invalid"
    p = np.zeros(len(y), float)
    for g in sorted(set(clip.tolist())):
        tr, te = clip != g, clip == g
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=max_iter, C=C, class_weight=class_weight)
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def _group_index(groups):
    uniq = sorted(set(np.asarray(groups).tolist()))
    gi = np.asarray([uniq.index(g) for g in groups])
    return uniq, [np.where(gi == k)[0] for k in range(len(uniq))]


def my_group_boot_ci(y, s, groups, n_boot=N_BOOT, seed=SEED):
    """Resample GROUPS with replacement; skip draws with one class."""
    y, s = np.asarray(y), np.asarray(s, float)
    uniq, idx_by = _group_index(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[k] for k in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(my_auc(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def my_paired_group_boot(y, s_a, s_b, groups, n_boot=N_BOOT, seed=SEED):
    """Paired: the SAME resampled groups for both scores."""
    y = np.asarray(y)
    s_a, s_b = np.asarray(s_a, float), np.asarray(s_b, float)
    uniq, idx_by = _group_index(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[k] for k in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(my_auc(y[idx], s_a[idx]) - my_auc(y[idx], s_b[idx]))
    delta = my_auc(y, s_a) - my_auc(y, s_b)
    if not vals:
        return {"delta": float(delta), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"delta": float(delta), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": len(vals)}


def my_window_boot_ci(y, s, n_boot=N_BOOT, seed=SEED):
    y, s = np.asarray(y), np.asarray(s, float)
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(my_auc(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


# ------------------------------------------------- independent data loading --
def load_gold_myself(bundle):
    """Re-derive ids / y / clip / X26 from the bundle files directly.

    Two independent sources are cross-checked:
      (a) labels/manifest_gold.json  -- the human labels;
      (b) caches/gpu/features.json   -- which carries its OWN 'clip' and 'label'
          fields per window, so a mismatch would expose a manifest/cache drift.
    """
    man = json.load(open(os.path.join(bundle, "labels", "manifest_gold.json")))
    feats = json.load(open(os.path.join(bundle, "caches", "gpu", "features.json")))["windows"]
    names = sorted(feats[man["items"][0]["id"]]["full"].keys())  # read from the cache
    ids, y, clip, reasons = [], [], [], []
    for it in man["items"]:
        k = it["id"]
        if k not in feats or "full" not in feats[k]:
            continue
        ids.append(k); y.append(int(it["label"])); clip.append(it["clip"])
        reasons.append(it.get("reason"))
    cache_ids = set(feats.keys())
    n_only_cache = len([k for k in cache_ids if k not in set(ids)])
    # (b) the cache's own labels/clips must agree with the manifest
    disagree = [(k, feats[k].get("label"), int(it["label"]), feats[k].get("clip"), it["clip"])
                for it in man["items"] if it["id"] in feats and "full" in feats[it["id"]]
                for k in [it["id"]]
                if feats[k].get("label") is not None
                and (int(feats[k]["label"]) != int(it["label"]) or feats[k].get("clip") != it["clip"])]
    X = np.asarray([[feats[k]["full"][n] for n in names] for k in ids], float)
    return {"ids": ids, "y": np.asarray(y, int), "clip": np.asarray(clip, dtype=object),
            "X": np.nan_to_num(X), "names": names, "reasons": reasons,
            "n_windows_in_cache": len(cache_ids), "n_cache_only": n_only_cache,
            "cache_vs_manifest_disagreements": disagree}


def verify_mosaic_artifact(path):
    """Re-implement jadal/mosaic/features.py::feature_digest and check the artifact.

    feature_digest(feats) = sha256(json.dumps({"feature_version": V,
        "values": {k: round(float(v), 6) for k in sorted(feats)}},
        sort_keys=True, separators=(",", ":")))
    top-level digest = sha256(json.dumps(per_window_digest, sort_keys=True))  (default separators)
    """
    doc = json.load(open(path))
    v = doc["feature_version"]
    bad, n = [], 0
    for wid, feats in doc["windows"].items():
        q = {k: (None if feats[k] is None or (isinstance(feats[k], float) and np.isnan(feats[k]))
                 else round(float(feats[k]), 6)) for k in sorted(feats)}
        d = hashlib.sha256(json.dumps({"feature_version": v, "values": q},
                                      sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest()
        n += 1
        if d != doc["per_window_digest"].get(wid):
            bad.append(wid)
    top = hashlib.sha256(json.dumps(doc["per_window_digest"], sort_keys=True).encode()).hexdigest()
    return {"n_windows_checked": n, "mismatched_per_window_digests": bad,
            "declared_top_digest": doc["feature_digest_sha256"],
            "recomputed_top_digest": top,
            "top_digest_matches": top == doc["feature_digest_sha256"],
            "feature_version_in_file": v,
            "feature_version_expected": MOSAIC_FEATURE_VERSION,
            "feature_names_count": len(doc["feature_names"]),
            "n_windows_in_file": doc["n_windows"],
            "n_duplicate_feature_names": len(doc["feature_names"]) - len(set(doc["feature_names"]))}


def matrix(doc, ids, names):
    W = doc["windows"]
    X = np.full((len(ids), len(names)), np.nan, float)
    for i, wid in enumerate(ids):
        w = W.get(wid)
        if w is None:
            continue
        for j, n in enumerate(names):
            val = w.get(n)
            X[i, j] = np.nan if val is None else float(val)
    return X


def check_preregistration(eid, result_path):
    """Verify the frozen config really is the config that produced the result.

    Checks what is checkable: the PREREG config hash recomputes, and it equals
    the config hash recorded in the results JSON.  What is NOT checkable from
    the repo is ordering -- PREREG.json and the results were committed in the
    same commit, so `frozen_before_run` is a self-report plus a date.  That is
    recorded honestly rather than asserted.
    """
    import subprocess
    p_path = os.path.join(REPO, "experiments", eid, "PREREG.json")
    if not os.path.exists(p_path):
        return {"available": False}
    pre = json.load(open(p_path))
    res = json.load(open(result_path))
    cfg = pre.get("config") or {}
    recomputed = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode()).hexdigest()
    def added(path):
        try:
            return subprocess.run(["git", "log", "--diff-filter=A", "--format=%ci %h", "--", path],
                                  capture_output=True, text=True, cwd=REPO,
                                  timeout=20).stdout.strip().splitlines()[-1]
        except Exception:
            return "unknown"
    return {
        "available": True,
        "config_sha256_recomputed": recomputed,
        "config_sha256_declared_in_prereg": pre.get("config_sha256"),
        "config_sha256_matches_prereg": recomputed == pre.get("config_sha256"),
        "config_sha256_matches_result": recomputed == res.get("config_sha256"),
        "frozen_before_run_self_reported": pre.get("frozen_before_run"),
        "frozen_utc": pre.get("frozen_utc"),
        "prereg_added_commit": added(os.path.join("experiments", eid, "PREREG.json")),
        "results_added_commit": added(os.path.relpath(result_path, REPO)),
        "ordering_independently_verifiable": False,
        "why_not": ("PREREG.json and the results JSON were added in the SAME commit, so git "
                    "history cannot show the config was frozen before the run; only the date "
                    "and the self-reported flag exist"),
    }


def main():
    t0 = time.time()
    bundle = os.environ.get("JADAL_BUNDLE") or common.BUNDLE
    grid01 = json.load(open(GRID01))
    grid02 = json.load(open(GRID02))
    grid03 = json.load(open(GRID03)) if os.path.exists(GRID03) else None

    g = load_gold_myself(bundle)
    ids, y, clip = g["ids"], g["y"], g["clip"]

    # ---- assertion 1: my derivation agrees with the shared loader -----------
    shared = common.load_gold(bundle)
    loader_check = {
        "ids_equal": g["ids"] == shared["ids"],
        "y_equal": bool(np.array_equal(g["y"], shared["y"])),
        "clip_equal": bool(np.array_equal(g["clip"], shared["clip"])),
        "n_features_equal": g["X"].shape == shared["X"].shape,
        # column order may differ (alphabetical here vs canonical there); a
        # permutation of columns cannot change a StandardScaler+LR fit, and the
        # set of names must be identical either way.
        "feature_name_sets_equal": set(g["names"]) == set(common.FEATURE_NAMES),
        "X_columns_permutation_equal": bool(
            g["X"].shape == shared["X"].shape and all(
                np.allclose(g["X"][:, g["names"].index(n)], shared["X"][:, i])
                for i, n in enumerate(common.FEATURE_NAMES))),
    }
    if not all(v for k, v in loader_check.items() if isinstance(v, bool)):
        raise SystemExit(f"my derivation disagrees with common.load_gold: {loader_check}")

    # ---- assertion 2: the mosaic artifact is the one it claims to be --------
    art = verify_mosaic_artifact(MOSAIC)
    doc = json.load(open(MOSAIC))
    if art["mismatched_per_window_digests"] or not art["top_digest_matches"]:
        raise SystemExit(f"mosaic artifact integrity check FAILED: {art}")
    if art["feature_version_in_file"] != MOSAIC_FEATURE_VERSION:
        raise SystemExit(f"unexpected feature_version: {art}")

    sp_names = [n for n in doc["feature_names"] if n.startswith("sp_")]
    it_names = [n for n in doc["feature_names"] if n.startswith("it_")]
    dm_names = [n for n in doc["feature_names"] if n.startswith("dm_")]
    sp_mean = [n for n in sp_names if n.endswith("_mean")]
    assert len(sp_names) == 60 and len(it_names) == 12, (len(sp_names), len(it_names))

    # ---- my own LOCO scores -------------------------------------------------
    p_base = my_loco(g["X"], y, clip)                                   # 26 features
    p_sp = my_loco(matrix(doc, ids, sp_names), y, clip)                  # 60
    p_spm = my_loco(matrix(doc, ids, sp_mean), y, clip)                  # 15
    p_spit = my_loco(matrix(doc, ids, sp_names + it_names), y, clip)     # 72
    p_spitdm = my_loco(matrix(doc, ids, sp_names + it_names + dm_names), y, clip)  # 79

    # ---- cross-check my AUROC against sklearn ------------------------------
    from sklearn.metrics import roc_auc_score
    auc_xcheck = {"mine_base": my_auc(y, p_base), "sklearn_base": float(roc_auc_score(y, p_base)),
                  "mine_spatial": my_auc(y, p_sp), "sklearn_spatial": float(roc_auc_score(y, p_sp)),
                  "mine_spit": my_auc(y, p_spit), "sklearn_spit": float(roc_auc_score(y, p_spit))}
    auc_xcheck["max_abs_diff"] = max(abs(auc_xcheck["mine_base"] - auc_xcheck["sklearn_base"]),
                                     abs(auc_xcheck["mine_spatial"] - auc_xcheck["sklearn_spatial"]),
                                     abs(auc_xcheck["mine_spit"] - auc_xcheck["sklearn_spit"]))

    def block(p, label, n_features):
        a = my_auc(y, p)
        lo, hi, nb = my_group_boot_ci(y, p, clip)
        wlo, whi, _ = my_window_boot_ci(y, p)
        slo, shi, snb = my_group_boot_ci(y, p, clip, n_boot=SENS_N_BOOT, seed=SENS_SEED)
        return {"label": label, "n_features": n_features, "auroc": round(a, 4),
                "auroc_raw": a, "ci95": [round(lo, 3), round(hi, 3)],
                "ci95_raw": [lo, hi], "n_boot_valid": nb,
                "ci95_type": f"clip-level bootstrap ({N_BOOT} draws, seed {SEED})",
                "ci95_window_optimistic": [round(wlo, 3), round(whi, 3)],
                "ci95_sensitivity": [round(slo, 3), round(shi, 3)],
                "ci95_sensitivity_type": f"clip-level bootstrap ({SENS_N_BOOT} draws, seed {SENS_SEED})",
                "ci95_sensitivity_n_boot_valid": snb}

    mine = {"baseline_26": block(p_base, "baseline_26_shipped_features", 26),
            "spatial_60": block(p_sp, "spatial_pooled_meanminmaxstd", 60),
            "spatial_mean_only_15": block(p_spm, "spatial_mean_only_capacity_control", 15),
            "spatial_intertile_72": block(p_spit, "spatial+intertile", 72),
            "spatial_intertile_diff_79": block(p_spitdm, "spatial+intertile+diff", 79)}

    delta_sp_base = my_paired_group_boot(y, p_sp, p_base, clip)
    delta_spit_sp = my_paired_group_boot(y, p_spit, p_sp, clip)
    delta_spm_sp = my_paired_group_boot(y, p_spm, p_sp, clip)
    delta_dm = my_paired_group_boot(y, p_spitdm, p_spit, clip)
    delta_sens = my_paired_group_boot(y, p_spit, p_sp, clip,
                                      n_boot=SENS_N_BOOT, seed=SENS_SEED)

    # ---- per-clip AUROC table ----------------------------------------------
    per_clip = {}
    for c in sorted(set(clip.tolist())):
        m = clip == c
        row = {"n": int(m.sum()), "n_good": int((y[m] == 1).sum()), "n_bad": int((y[m] == 0).sum())}
        for tag, p in (("baseline_26", p_base), ("spatial_60", p_sp),
                       ("spatial_intertile_72", p_spit)):
            row[tag] = (round(my_auc(y[m], p[m]), 4)
                        if len(set(y[m].tolist())) == 2 else None)
        per_clip[c] = row

    # ---- kill gate ---------------------------------------------------------
    claim_01 = grid01["primary"]["auroc"]                       # 0.6864
    claim_02 = grid02["primary"]["auroc"]                       # 0.7058
    mine_01, mine_02 = mine["spatial_60"]["auroc"], mine["spatial_intertile_72"]["auroc"]
    raw_delta = mine["spatial_intertile_72"]["auroc_raw"] - mine["spatial_60"]["auroc_raw"]
    sign_convention = ("paired delta is AUROC(spatial+intertile) - AUROC(spatial_only); "
                       "positive means the intertile group helped")
    observed_claim = grid02["kill_gate"]["observed"]            # +0.0194
    # run_grid.py computes the gate on the ROUNDED aurocs; check both readings
    gate_on_rounded = bool(round(mine_02, 4) - round(mine_01, 4) >= KILL_GATE_DELTA)
    gate_on_raw = bool(raw_delta >= KILL_GATE_DELTA)
    gate = {
        "delta_required": KILL_GATE_DELTA,
        "my_delta_raw": raw_delta,
        "my_delta_rounded_4dp": round(mine_02, 4) - round(mine_01, 4),
        "claimed_observed_delta": observed_claim,
        "claim_passed": grid02["kill_gate"]["passed"],
        "my_gate_on_rounded_auroc": gate_on_rounded,
        "my_gate_on_raw_auroc": gate_on_raw,
        "rounding_changes_outcome": bool(gate_on_rounded != gate_on_raw),
        "margin_below_threshold_raw": KILL_GATE_DELTA - raw_delta,
        "paired_ci_excludes_zero": delta_spit_sp["excludes_zero"],
        "note": ("run_grid.py evaluates the gate on auroc values already rounded to 4 dp; "
                 "both readings are computed here"),
    }

    comparisons = {
        "grid01_spatial_vs_claim": {"mine_auroc": mine_01, "claimed_auroc": claim_01,
                                    "mine_ci95": mine["spatial_60"]["ci95"], "claimed_ci95": [0.522, 0.772],
                                    "auroc_diff": round(mine_01 - claim_01, 6)},
        "grid01_spatial_vs_baseline26": {
            "mine": delta_sp_base,
            "claim": grid01["primary"]["delta_vs_baseline26"],
            "delta_diff": round(delta_sp_base["delta"] - grid01["primary"]["delta_vs_baseline26"]["delta"], 6)},
        "grid01_mean_only_vs_primary": {
            "mine": delta_spm_sp, "claim": grid01["secondary_mean_pooling"]["delta_vs_primary"],
            "delta_diff": round(delta_spm_sp["delta"] - grid01["secondary_mean_pooling"]["delta_vs_primary"]["delta"], 6)},
        "grid02_spatial_intertile_vs_claim": {
            "mine_auroc": mine_02, "claimed_auroc": claim_02,
            "mine_ci95": mine["spatial_intertile_72"]["ci95"], "claimed_ci95": [0.514, 0.785],
            "auroc_diff": round(mine_02 - claim_02, 6)},
        "grid03_diff_vs_spatial_intertile": {
            "mine": delta_dm,
            "claim": grid03["primary"]["delta_vs_spatial_plus_intertile"] if grid03 else None,
            "claim_auroc": grid03["primary"]["auroc"] if grid03 else None,
            "mine_auroc": mine["spatial_intertile_diff_79"]["auroc"],
            "delta_diff": (round(delta_dm["delta"] - grid03["primary"]["delta_vs_spatial_plus_intertile"]["delta"], 6) if grid03 else None)},
        "grid02_intertile_vs_spatial_only": {
            "mine": delta_spit_sp, "claim": grid02["primary"]["delta_vs_spatial_only"],
            "delta_diff": round(delta_spit_sp["delta"] - grid02["primary"]["delta_vs_spatial_only"]["delta"], 6)},
        "baseline26_vs_grid00": {
            "mine_auroc": mine["baseline_26"]["auroc"], "mine_ci95": mine["baseline_26"]["ci95"],
            "grid00_auroc": 0.81, "grid00_ci95": [0.683, 0.892],
            "auroc_diff": round(mine["baseline_26"]["auroc"] - 0.81, 6)},
    }
    reproduced = {
        "grid01_auroc": abs(comparisons["grid01_spatial_vs_claim"]["auroc_diff"]) < 5e-5,
        "grid01_paired_delta": abs(comparisons["grid01_spatial_vs_baseline26"]["delta_diff"]) < 5e-5,
        "grid02_auroc": abs(comparisons["grid02_spatial_intertile_vs_claim"]["auroc_diff"]) < 5e-5,
        "grid02_paired_delta": abs(comparisons["grid02_intertile_vs_spatial_only"]["delta_diff"]) < 5e-5,
        "grid00_baseline": abs(comparisons["baseline26_vs_grid00"]["auroc_diff"]) < 5e-5,
        "grid03_auroc": (abs(mine["spatial_intertile_diff_79"]["auroc"] - grid03["primary"]["auroc"]) < 5e-5) if grid03 else None,
        "grid03_paired_delta": (abs(comparisons["grid03_diff_vs_spatial_intertile"]["delta_diff"]) < 5e-5) if grid03 else None,
    }
    ci_match = {
        "grid01_ci": mine["spatial_60"]["ci95"] == [0.522, 0.772],
        "grid02_ci": mine["spatial_intertile_72"]["ci95"] == [0.514, 0.785],
        "baseline_ci": mine["baseline_26"]["ci95"] == [0.683, 0.892],
        "grid01_paired_ci": [round(delta_sp_base["lo"], 5), round(delta_sp_base["hi"], 5)]
                            == [-0.20164, -0.07104],
        "grid02_paired_ci": [round(delta_spit_sp["lo"], 5), round(delta_spit_sp["hi"], 5)]
                            == [-0.01803, 0.05316],
    }
    all_reproduced = all(reproduced.values())
    verdict = ("reproduced exactly" if all_reproduced and all(ci_match.values())
               else "REPRODUCED WITH DISAGREEMENT")

    cfg = {"id": ID, "verified_ids": ["GRID.01", "GRID.02", "GRID.00 (baseline only)"],
           "source_of_claim": ["eval/results/exp_grid/GRID.01.json", "eval/results/exp_grid/GRID.02.json"],
           "feature_artifact": "eval/results/exp_grid/features_m3x3_t224.json",
           "my_auc_impl": "rank AUROC, ties averaged (cross-checked vs sklearn roc_auc_score)",
           "my_loco": f"StandardScaler+LogisticRegression(C={C_REG}, class_weight=balanced, max_iter={MAX_ITER}), my own leave-one-clip-out loop",
           "bootstrap": f"resample 7 clips with replacement, {N_BOOT} draws, seed {SEED}; paired draws share indices",
           "sensitivity_bootstrap": f"{SENS_N_BOOT} draws, seed {SENS_SEED}",
           "kill_gate_delta": KILL_GATE_DELTA, "seed": SEED,
           "did_not_import": ["eval/exp_grid/run_grid.py", "tools/dsh/common.py metric functions"]}

    payload = {
        "id": ID, "status": "done",
        "headline": ("independent re-run of GRID.02: spatial+intertile AUROC and its paired delta "
                     "vs spatial-only, and whether the 0.02 kill gate still fails"),
        "auroc": mine_02, "ci95": [round(delta_spit_sp["lo"], 4), round(delta_spit_sp["hi"], 4)],
        "ci95_type": f"paired clip-level bootstrap of the delta ({N_BOOT} draws, seed {SEED})",
        "n": int(len(y)), "n_bad": int((y == 0).sum()), "n_good": int((y == 1).sum()),
        "n_groups": int(len(set(clip.tolist()))),
        "label_source": "human gold, ONE labeller (150 IndEgo windows: 100 good / 50 bad, 7 clips)",
        "protocol": ("independent re-implementation of LOCO + rank AUROC + clip bootstrap + paired clip "
                     "bootstrap on the committed mosaic feature vectors; no code from the claiming "
                     "script or from the shared metric module is used"),
        "verdict": verdict,
        "reproduced": reproduced, "ci_match": ci_match, "all_reproduced": all_reproduced,
        "mine": mine, "claimed": {
            "grid01_primary_auroc": claim_01, "grid01_primary_ci95": grid01["primary"]["ci95"],
            "grid01_delta_vs_baseline26": grid01["primary"]["delta_vs_baseline26"],
            "grid01_secondary_mean_only_auroc": grid01["secondary_mean_pooling"]["auroc"],
            "grid02_primary_auroc": claim_02, "grid02_primary_ci95": grid02["primary"]["ci95"],
            "grid02_delta_vs_spatial_only": grid02["primary"]["delta_vs_spatial_only"],
            "baseline26_auroc": grid01["baseline_26_on_same_folds"]["auroc"],
            "baseline26_ci95": grid01["baseline_26_on_same_folds"]["ci95"],
        },
        "comparisons_mine_first": comparisons,
        "per_clip_auroc": per_clip,
        "kill_gate": gate,
        "kill_gate_conclusion_survives": bool(not gate_on_rounded and not gate_on_raw),
        "preregistration": {"GRID.01": check_preregistration("GRID.01", GRID01),
                            "GRID.02": check_preregistration("GRID.02", GRID02)},
        "presentation_findings": [
            {
                "where": "GRID.01.json -> secondary_mean_pooling.delta_vs_primary",
                "what": ("the recorded value -0.0258 is AUROC(primary 60-feature) MINUS "
                         "AUROC(secondary 15-feature mean-only) = 0.6864 - 0.7122. The field name "
                         "reads as if it were the secondary's delta against the primary, which "
                         "would be +0.0258. Same magnitude, ambiguous sign convention."),
                "my_value": round(delta_spm_sp["delta"], 6),
                "my_label_convention": "AUROC(secondary mean-only) - AUROC(primary pooled)",
                "grid_report_reads_it_correctly": (
                    "yes -- docs/reports/grid_pipeline.md says the control is 0.026 'better' than "
                    "the 60-feature version, the correct direction"),
                "affects_a_conclusion": False,
            },
            {
                "where": "eval/exp_grid/run_grid.py kill-gate evaluation",
                "what": ("the gate compares AUROC values already rounded to 4 dp (round(a, 4) is "
                         "stored back into the result dict before the comparison). Recomputed on "
                         "RAW values too: the outcome is the same either way."),
                "raw_delta": raw_delta,
                "gate_margin_to_threshold": KILL_GATE_DELTA - raw_delta,
                "rounding_changes_outcome": bool(gate_on_rounded != gate_on_raw),
                "affects_a_conclusion": False,
            },
        ],
        "sign_convention": sign_convention,
        "data_integrity": {"mosaic_artifact": art, "loader_cross_check": loader_check,
                           "my_gold_derivation": {"n": len(ids),
                                                  "n_cache_only_windows": g["n_cache_only"],
                                                  "n_windows_in_cache": g["n_windows_in_cache"],
                                                  "cache_vs_manifest_disagreements":
                                                      g["cache_vs_manifest_disagreements"][:5],
                                                  "n_cache_vs_manifest_disagreements":
                                                      len(g["cache_vs_manifest_disagreements"])}},
        "auc_cross_check_sklearn": auc_xcheck,
        "caveat": ("this reproduces the GRID numbers, it does not validate them: the same 150 in-domain "
                   "windows from ONE labeller, 7 clips, and a mosaic feature family the claiming stream "
                   "also designed. Exact agreement means the arithmetic and the artifact are sound, not "
                   "that the protocol is free of selection."),
        "config": cfg, "config_sha256": common.config_sha256(cfg),
        "device": common.DEVICE_CPU,
        "cost": f"0 GPU; 4 LOCO fits (7 folds each) + 4 bootstraps; {time.time() - t0:.1f} s CPU",
        "command": "JADAL_BUNDLE=<bundle> /home/parshu/envs/jadal/bin/python tools/task/verify_grid01.py",
        "git_sha": common.git_sha(), "seed": SEED,
    }
    common.save_results(OUT, payload)

    print(f"VERDICT: {verdict}")
    print("mine  first, claim second:")
    print(f"  GRID.00 baseline 26 : {mine['baseline_26']['auroc']:.4f} {mine['baseline_26']['ci95']}"
          f"  vs claim 0.8100 [0.683, 0.892]")
    print(f"  GRID.01 spatial 60  : {mine['spatial_60']['auroc']:.4f} {mine['spatial_60']['ci95']}"
          f"  vs claim {claim_01:.4f} {grid01['primary']['ci95']}")
    print(f"  GRID.02 sp+it 72    : {mine['spatial_intertile_72']['auroc']:.4f} "
          f"{mine['spatial_intertile_72']['ci95']}  vs claim {claim_02:.4f} {grid02['primary']['ci95']}")
    print(f"  GRID.01 delta vs 26 : {delta_sp_base['delta']:+.5f} "
          f"[{delta_sp_base['lo']:+.5f}, {delta_sp_base['hi']:+.5f}]")
    print(f"  GRID.02 delta vs sp : {delta_spit_sp['delta']:+.5f} "
          f"[{delta_spit_sp['lo']:+.5f}, {delta_spit_sp['hi']:+.5f}]"
          f"  (claim {observed_claim:+.4f} "
          f"[{grid02['primary']['delta_vs_spatial_only']['lo']:+.5f}, "
          f"{grid02['primary']['delta_vs_spatial_only']['hi']:+.5f}])")
    print(f"  KILL GATE >= {KILL_GATE_DELTA}: raw delta {raw_delta:.6f} -> "
          f"{'PASS' if gate_on_raw else 'FAIL'}; rounded -> {'PASS' if gate_on_rounded else 'FAIL'}; "
          f"rounding changes outcome: {gate['rounding_changes_outcome']}")
    print("  sensitivity CI (10000 draws, seed 12345):",
          mine["spatial_intertile_72"]["ci95_sensitivity"], "delta:",
          [round(delta_sens["lo"], 5), round(delta_sens["hi"], 5)])
    if grid03:
        print(f"  GRID.03 sp+it+dm 79 : {mine['spatial_intertile_diff_79']['auroc']:.4f} "
              f"{mine['spatial_intertile_diff_79']['ci95']}  vs claim {grid03['primary']['auroc']:.4f} "
              f"{grid03['primary']['ci95']}")
        print(f"  GRID.03 delta vs 72 : {delta_dm['delta']:+.5f} "
              f"[{delta_dm['lo']:+.5f}, {delta_dm['hi']:+.5f}]  (claim "
              f"{grid03['primary']['delta_vs_spatial_plus_intertile']['delta']:+.5f} "
              f"[{grid03['primary']['delta_vs_spatial_plus_intertile']['lo']:+.5f}, "
              f"{grid03['primary']['delta_vs_spatial_plus_intertile']['hi']:+.5f}])")
    print("per-clip:", json.dumps(per_clip, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
