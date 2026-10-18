#!/usr/bin/env python3
"""GRID.09 — brightness-calibrated blur rule on the shipped S0 probe.

The shipped pipeline's S0 probe (``scripts/poc_run.py::s0_from_luma``, consumed by
``jadal/poc/cascade.py::s0_decide``) measures brightness, spatial variance and
temporal energy but no sharpness, so a strongly blurred window reaches S1.  This
experiment adds ``lapvar_cal`` (Laplacian variance of the 80x45 grey probe
divided by the probe's mean luma; one definition in ``jadal.poc.s0_blur``) as a
sixth S0 rule and measures the catch / false-reject trade-off.

Protocol (pre-registered, see experiments/GRID.09/README.md)
-----------------------------------------------------------
* Positives are SYNTHETIC: a Gaussian blur applied to the decoded 480p source of
  windows whose INDEPENDENT human label is good.  No defect is claimed to be
  human-labelled and no blurred frame is committed.
* Split is GROUPED and fixed before any threshold is chosen: for each dataset the
  sorted unique groups (IndEgo clip / OOD-160 factory) take even index -> CALIB,
  odd index -> EVAL.  The threshold is chosen on CALIB only; every headline
  number is reported on EVAL.
* Threshold rule (chosen on CLEAN data only, never on a blurred window): the
  largest t whose clean-only false-reject rate on CALIB is <= 2%, i.e. the 2nd
  percentile of the clean CALIB distribution.  The frozen value is asserted
  against ``jadal/poc/s0_blur.BLUR_LAPVAR_CAL_MIN_CALIBRATED`` before evaluation.
  The SHIPPED default in ``BLUR_LAPVAR_CAL_MIN`` is 0.0 (rule disabled), so this
  branch changes no shipped verdict until that constant is flipped; every number
  here is measured at the calibrated operating point and the JSON records both.
* Second, reported arm: Youden J on CALIB (clean vs sigma=3) — for reference only.
* Intervals: group bootstrap (resample clips/factories), 2000 draws, seed 0.
  The IndEgo-only eval split has 3 groups; its interval is printed but flagged.

Phases
------
  --phase probe      decode + all variants -> eval/results/exp_grid/probe_blur_windows.json
  --phase calibrate  read the cache, choose the threshold -> grid09_calibration.json
  --phase head       shipped S1 head (26 features) on a subset, full 10 s path
  --phase evaluate   read the cache + frozen threshold -> GRID.09.json
  --phase all        probe, calibrate, head, evaluate

Nothing here writes pixels; the only artifacts are small numeric JSONs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import common as C  # noqa: E402  (tools/dsh/common.py — shared, not forked)
import cv2  # noqa: E402
import poc_run  # noqa: E402  (production probe: s0_from_luma)
from jadal.mosaic import s0_parity as P  # noqa: E402
from jadal.poc import cascade as CASC  # noqa: E402
from jadal.poc import s0_blur  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results", "exp_grid")
CACHE = os.path.join(OUT, "probe_blur_windows.json")
CALIB = os.path.join(OUT, "grid09_calibration.json")
RESULTS = os.path.join(OUT, "GRID.09.json")

WORK = (160, 90)          # production working resolution (C.WORK_W x C.WORK_H)
PROBE_FRAMES = 60         # 2 s at the production 30 fps normalisation
PROBE_SOURCE_S = 2.0      # seconds of source content the probe sees
SEED = 0
N_BOOT = 2000
FRR_BUDGET = 0.02         # clean-only false-reject budget for threshold selection

#: variant name -> (gaussian sigma or None, multiplicative gain applied AFTER blur)
VARIANTS = {
    "clean": (None, 1.0),
    "blur15": (1.5, 1.0),
    "blur30": (3.0, 1.0),
    "blur80": (8.0, 1.0),
    "blur30_dark": (3.0, 0.5),
    "blur30_bright": (3.0, 1.4),
    "dark": (None, 0.5),
    "bright": (None, 1.4),
}

#: statistics recorded per variant
STATS = ("mean", "p5", "p95", "spatial_var", "temp_e", "lapvar_cal",
         "lapvar_cal_45", "lapvar_cal_tmean", "lapvar_raw", "lapvar_raw_160")

#: the sharpness statistics the ablation table compares.  ``lapvar_cal`` is the
#: production rule statistic (160x90 decoded luma); ``lapvar_cal_45`` is the
#: rejected 80x45 candidate that the shipped probe buffer would have given.
SHARPNESS_STATS = ("lapvar_cal", "lapvar_cal_45", "lapvar_cal_tmean",
                   "lapvar_raw", "lapvar_raw_160")

#: thresholds examined in the operating-point table, as clean-CALIB quantiles
OP_QUANTILES = (0.005, 0.01, 0.02, 0.05, 0.10)

SPLITS = "even/odd index over the sorted unique groups of each dataset (even -> calib, odd -> eval)"


def ksize_for(sigma: float) -> int:
    return 2 * int(np.ceil(3.0 * sigma)) + 1


# ------------------------------------------------------------------ decode ----
def decode_grey_source(path: str, start_s: float, seconds: float):
    """Decode ``seconds`` of source GREY as float32 [0,1] (n, h, w) + native fps.

    Grey is derived with the production weights on RGB in [0,1]
    (``poc_run.luma_of``), so the probe sees the same luma definition; the
    downscale to 160x90 is INTER_AREA (production uses ffmpeg ``scale=160:90``,
    bicubic -- recorded as a confound in the results, not hidden).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)
    want = max(1, int(round(seconds * fps)))
    out = []
    while len(out) < want:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        rgb = fr[..., ::-1].astype(np.float32)
        out.append((0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                    + 0.114 * rgb[..., 2]) / 255.0)
    cap.release()
    if len(out) < 2:
        raise RuntimeError(f"no frames {path}@{start_s}")
    return np.stack(out), float(fps)


def apply_variant(stack: np.ndarray, name: str) -> np.ndarray:
    sigma, gain = VARIANTS[name]
    g = stack
    if sigma is not None:
        k = ksize_for(sigma)
        g = np.stack([cv2.GaussianBlur(f, (k, k), sigma) for f in g])
    if gain != 1.0:
        g = np.clip(g * gain, 0.0, 1.0)
    return g.astype(np.float32)


def apply_variant_rgb(rgb_u8: np.ndarray, name: str) -> np.ndarray:
    """Same defect on uint8 RGB at SOURCE resolution (for the shipped-head arm).

    The defect is injected on the source pixels, exactly as it would be in a
    blurred upload; only then does the pipeline downscale to 160x90.  uint8
    blur/gain instead of float32 costs a rounding step (recorded as a confound).
    """
    sigma, gain = VARIANTS[name]
    g = rgb_u8
    if sigma is not None:
        k = ksize_for(sigma)
        g = cv2.GaussianBlur(g, (k, k), sigma)
    if gain != 1.0:
        g = np.clip(g.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)
    return g


def normalise_to_probe(stack: np.ndarray, native_fps: float) -> np.ndarray:
    """60 probe frames: frame-repeat to 30 fps (production ``_normalise_fps``).

    IndEgo is 10 fps, so 2 s of source becomes 20 native frames repeated 3x = 60
    probe frames -- exactly what ``fps=30`` resampling does to the production
    stream.  A 30 fps source is used as-is.
    """
    rep = max(1, int(round(30.0 / native_fps))) if native_fps < 29.5 else 1
    idx = np.repeat(np.arange(len(stack)), rep)
    if len(idx) < PROBE_FRAMES:
        idx = np.concatenate([idx, np.full(PROBE_FRAMES - len(idx), len(stack) - 1)])
    return stack[idx[:PROBE_FRAMES]]


def probe_of(stack30: np.ndarray) -> dict:
    """Production probe (``poc_run.s0_from_luma``) + the recorded ablations."""
    small = np.stack([cv2.resize(f, (WORK[0], WORK[1]),
                                 interpolation=cv2.INTER_AREA) for f in stack30])
    s0 = poc_run.s0_from_luma(small)
    g80 = small.reshape(PROBE_FRAMES, 45, 2, 80, 2).mean(axis=(2, 4))
    out = {k: float(s0[k]) for k in ("mean", "p5", "p95", "spatial_var", "temp_e",
                                     "lapvar_cal", "lapvar_cal_45")}
    out["lapvar_cal_tmean"] = s0_blur.probe_lapvar_cal_of_mean(g80)
    out["lapvar_raw"] = s0_blur.probe_laplacian_var(g80)
    out["lapvar_raw_160"] = s0_blur.probe_laplacian_var(small)
    return out


def _init_worker():
    os.nice(10)


def process_window(task):
    """One window: decode once, apply every variant in memory, probe each."""
    wid, dataset, group, path, start_s = task
    src, fps = decode_grey_source(path, start_s, PROBE_SOURCE_S)
    rec = {"id": wid, "dataset": dataset, "group": group, "native_fps": fps,
           "n_source_frames": int(len(src)), "variants": {}}
    for name in VARIANTS:
        stack = apply_variant(src, name)
        rec["variants"][name] = probe_of(normalise_to_probe(stack, fps))
    return wid, rec


# ------------------------------------------------------------------ splits ----
def split_map(recs):
    """(dataset, group) -> 'calib' | 'eval' by sorted group index parity."""
    out = {}
    for ds in sorted({r["dataset"] for r in recs}):
        groups = sorted({r["group"] for r in recs if r["dataset"] == ds})
        for i, g in enumerate(groups):
            out[(ds, g)] = "calib" if i % 2 == 0 else "eval"
    return out


def groups_for(recs, key="group"):
    return [f"{r['dataset']}:{r[key]}" for r in recs]


def group_rate_ci(hits, groups, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap CI for a rate: resample groups with replacement."""
    hits = np.asarray(hits, float)
    gs = list(groups)
    uniq = sorted(set(gs))
    idx_by = {g: np.where(np.asarray(gs) == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[uniq[k]] for k in pick])
        if len(idx) == 0:
            continue
        vals.append(float(hits[idx].mean()))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)], len(vals)


def auroc_ci(scores_pos, scores_neg, groups_pos, groups_neg, n_boot=N_BOOT, seed=SEED):
    """Group-bootstrap CI for the RULE-oriented AUROC of a less-than statistic.

    The blur rule fires when the statistic is LOW, so the score handed to
    ``fast_auc`` is ``-statistic``: 0.5 is chance and 1.0 means every blurred
    window scores below every clean window.
    """
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    s = -np.concatenate([scores_pos, scores_neg])
    gr = list(groups_pos) + list(groups_neg)
    uniq = sorted(set(gr))
    gi = np.asarray([uniq.index(g) for g in gr])
    idx_by = [np.where(gi == k)[0] for k in range(len(uniq))]
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[k] for k in pick])
        d = y[idx]
        if d.min() == d.max():
            continue
        vals.append(C.fast_auc(d, s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return C.fast_auc(y, s), [float(lo), float(hi)], len(vals)


# ------------------------------------------------------------------ phases ----
def phase_probe(args):
    g = C.load_gold()
    o = C.load_ood160()
    ood_rel = {w["window"]: w["video"] for w in
               json.load(open(C._ood160_manifest(C.BUNDLE)))["windows"]}
    tasks = []
    for i, wid in enumerate(g["ids"]):
        if int(g["y"][i]) != 1:
            continue                      # only human-GOOD windows are used
        tasks.append((wid, "indego", str(g["clip"][i]), g["path"][i],
                      float(g["start_s"][i])))
    for i, wid in enumerate(o["ids"]):
        if int(o["y"][i]) != 1:
            continue
        rel = ood_rel[wid]
        p = rel if os.path.isabs(rel) else os.path.join(C.BUNDLE, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        tasks.append((wid, "ood160", str(o["factory"][i]), p, 0.0))
    print(f"GRID.09 probe: {len(tasks)} human-good windows "
          f"({sum(1 for t in tasks if t[1]=='indego')} IndEgo, "
          f"{sum(1 for t in tasks if t[1]=='ood160')} OOD-160), "
          f"{len(VARIANTS)} variants each, {args.workers} workers", flush=True)
    t0 = time.time()
    store, done = {}, 0
    if args.workers <= 1:
        _init_worker()
        it = map(process_window, tasks)
        for wid, rec in it:
            store[wid] = rec
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker) as ex:
            for wid, rec in ex.map(process_window, tasks, chunksize=4):
                store[wid] = rec
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    payload = {
        "id": "GRID.09", "phase": "probe", "cache_version": 1,
        "n_windows": len(store), "variants": {k: list(v) for k, v in VARIANTS.items()},
        "ksize": {k: (None if v[0] is None else ksize_for(v[0]))
                  for k, v in VARIANTS.items()},
        "probe_source_seconds": PROBE_SOURCE_S, "probe_frames": PROBE_FRAMES,
        "work_res": list(WORK), "stats": list(STATS),
        "split_rule": SPLITS, "seed": SEED,
        "git_sha": C.git_sha(), "device": C.DEVICE_CPU,
        "command": "JADAL_BUNDLE=<bundle> nice -n 10 python eval/exp_grid/grid09.py --phase probe",
        "windows": store,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(payload, f, sort_keys=True)
    print(f"wrote {CACHE}  ({len(store)} windows, {time.time()-t0:.0f}s)", flush=True)


def _load_cache():
    d = json.load(open(CACHE))
    recs = list(d["windows"].values())
    sm = split_map(recs)
    for r in recs:
        r["split"] = sm[(r["dataset"], r["group"])]
    return d, recs


def phase_calibrate(args):
    d, recs = _load_cache()
    clean_calib = [r for r in recs if r["split"] == "calib"]
    vals = np.asarray([r["variants"]["clean"]["lapvar_cal"] for r in clean_calib], float)
    thr_clean = float(np.percentile(vals, 100.0 * FRR_BUDGET))
    # reference arm: Youden J on CALIB (clean vs sigma=3)
    pos = np.asarray([r["variants"]["blur30"]["lapvar_cal"] for r in clean_calib], float)
    neg = vals
    cand = np.unique(np.round(np.concatenate([pos, neg]), 8))
    best = (-1.0, None)
    for t in cand:
        tpr = float((pos < t).mean())
        tnr = float((neg >= t).mean())
        j = tpr + tnr - 1.0
        if j > best[0]:
            best = (j, float(t))
    per_ds = {}
    for ds in sorted({r["dataset"] for r in clean_calib}):
        v = np.asarray([r["variants"]["clean"]["lapvar_cal"] for r in clean_calib
                        if r["dataset"] == ds], float)
        per_ds[ds] = {"n": int(len(v)), "p02": float(np.percentile(v, 2.0)),
                      "p50": float(np.percentile(v, 50.0)),
                      "min": float(v.min()), "max": float(v.max())}
    payload = {
        "id": "GRID.09", "phase": "calibrate",
        "n_calib_windows": len(clean_calib),
        "n_calib_groups": len({(r["dataset"], r["group"]) for r in clean_calib}),
        "frr_budget": FRR_BUDGET,
        "selection_rule": ("largest t with clean-only false-reject rate on CALIB <= 2% "
                           "= 2nd percentile of the clean CALIB lapvar_cal distribution"),
        "selected_threshold": thr_clean,
        "youden_j_on_calib": {"threshold": best[1], "j": best[0],
                              "note": "reference arm only; not the frozen rule"},
        "clean_calib_lapvar_cal": {"n": int(len(vals)), "p02": thr_clean,
                                   "p50": float(np.percentile(vals, 50.0)),
                                   "min": float(vals.min()), "max": float(vals.max())},
        "per_dataset": per_ds,
        "split_rule": SPLITS,
        "calib_groups": sorted({f"{r['dataset']}:{r['group']}" for r in clean_calib}),
        "eval_groups": sorted({f"{r['dataset']}:{r['group']}" for r in recs
                               if r["split"] == "eval"}),
        "git_sha": C.git_sha(), "seed": SEED, "device": C.DEVICE_CPU,
        "command": "python eval/exp_grid/grid09.py --phase calibrate",
        "cache": os.path.relpath(CACHE, ROOT),
    }
    with open(CALIB, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print(json.dumps({k: payload[k] for k in
                      ("selected_threshold", "youden_j_on_calib", "per_dataset")},
                     indent=1, sort_keys=True))
    print(f"wrote {CALIB}")


def _arm(rows, variant, thr, label):
    """Catch / false-reject numbers for one variant at one threshold."""
    vals = np.asarray([r["variants"][variant]["lapvar_cal"] for r in rows], float)
    hit = vals < thr
    ci, nb = group_rate_ci(hit, groups_for(rows))
    return {"label": label, "n": int(len(rows)), "threshold": float(thr),
            "catch_rate": float(hit.mean()) if len(rows) else float("nan"),
            "ci95": ci, "ci95_type": f"group bootstrap ({nb} draws, seed {SEED})",
            "median_lapvar_cal": float(np.median(vals)) if len(rows) else float("nan")}


def phase_evaluate(args):
    d, recs = _load_cache()
    if not os.path.exists(CALIB):
        raise SystemExit("run --phase calibrate first")
    cal = json.load(open(CALIB))
    frozen = float(s0_blur.BLUR_LAPVAR_CAL_MIN_CALIBRATED)
    if abs(frozen - float(cal["selected_threshold"])) > 1e-12:
        raise SystemExit(
            f"BLUR_LAPVAR_CAL_MIN_CALIBRATED={frozen} does not match the calibrated "
            f"{cal['selected_threshold']}; freeze the constant before evaluating "
            "(that is the point of the two-phase protocol)")
    thr = frozen
    shipped_default = float(s0_blur.BLUR_LAPVAR_CAL_MIN)
    rule_enabled_by_default = shipped_default > 0.0

    def six_rule(probe):
        """The five shipped rules + BLUR at the CALIBRATED threshold."""
        rules = [tuple(r) for r in CASC.S0_RULES[:5]] + [("BLUR", "lapvar_cal", "<", thr)]
        return P.decide_with(probe, rules)
    ev = [r for r in recs if r["split"] == "eval"]
    hdr = {"indego": [], "ood160": [], "pooled": ev}

    def subset(name):
        return ev if name == "pooled" else [r for r in ev if r["dataset"] == name]

    # per-sigma catch + AUROC on the EVAL split
    per_sigma, aurocs = {}, {}
    for v in ("blur15", "blur30", "blur80", "blur30_dark", "blur30_bright"):
        rows = {}
        for ds in hdr:
            rs = subset(ds)
            if ds == "pooled":
                rows[ds] = _arm(rs, v, thr, f"{v} vs clean, pooled EVAL")
            else:
                rs = [r for r in rs if r["dataset"] == ds]
                rows[ds] = _arm(rs, v, thr, f"{v} vs clean, EVAL {ds}")
        per_sigma[v] = rows
        au = {}
        for ds in hdr:
            rs = subset(ds)
            clean = np.asarray([r["variants"]["clean"]["lapvar_cal"] for r in rs], float)
            pos = np.asarray([r["variants"][v]["lapvar_cal"] for r in rs], float)
            gp = groups_for(rs)
            a, ci, nb = auroc_ci(pos, clean, gp, gp)
            au[ds] = {"n_pos": int(len(pos)), "n_neg": int(len(clean)), "auroc": float(a),
                      "ci95": ci,
                      "ci95_type": f"group bootstrap ({nb} draws, seed {SEED})"}
        aurocs[v] = au

    # ablations at sigma=3 and on the brightness control
    ab = {}
    for stat in SHARPNESS_STATS:
        row = {}
        for v in ("blur30", "blur30_bright"):
            rs = ev
            clean = np.asarray([r["variants"]["clean"][stat] for r in rs], float)
            pos = np.asarray([r["variants"][v][stat] for r in rs], float)
            gp = groups_for(rs)
            nan = int(np.isnan(pos).sum() + np.isnan(clean).sum())
            if nan:
                keep = ~(np.isnan(pos) | np.isnan(clean))
                pos, clean = pos[keep], clean[keep]
                gp = [g for g, k in zip(gp, keep) if k]
            a, ci, nb = auroc_ci(pos, clean, gp, gp)
            row[v] = {"auroc": float(a), "ci95": ci, "n_pos": int(len(pos)),
                      "n_neg": int(len(clean)), "dropped_nan": nan}
        ab[stat] = row

    # false-reject side: clean human-good windows
    frr = {}
    for ds in ("indego", "ood160", "pooled"):
        for scope, rows in (("eval", subset(ds)), ("all", recs if ds == "pooled"
                                                   else [r for r in recs
                                                         if r["dataset"] == ds])):
            clean = np.asarray([r["variants"]["clean"]["lapvar_cal"] for r in rows], float)
            hit = clean < thr
            ci, nb = group_rate_ci(hit, groups_for(rows))
            s0_ship = [six_rule(r["variants"]["clean"])[0] for r in rows]
            frr[f"{ds}_{scope}"] = {
                "n": int(len(rows)),
                "n_groups": len({(r["dataset"], r["group"]) for r in rows}),
                "shipped_five_rule_exits": int(sum(1 for x in s0_ship if x is not None)),
                "blur_rule_exits": int(hit.sum()),
                "extra_exits_after_blur": int(sum(
                    1 for r, s in zip(rows, s0_ship)
                    if s is None and r["variants"]["clean"]["lapvar_cal"] < thr)),
                "false_reject_rate": float(hit.mean()),
                "ci95": ci,
                "ci95_type": f"group bootstrap ({nb} draws, seed {SEED})",
            }

    # what the brightness-only variants do (must not be rejected)
    bright_ctrl = {}
    for v in ("dark", "bright"):
        for ds in ("indego", "ood160", "pooled"):
            rs = subset(ds)
            vals = np.asarray([r["variants"][v]["lapvar_cal"] for r in rs], float)
            hit = vals < thr
            ci, nb = group_rate_ci(hit, groups_for(rs))
            bright_ctrl[f"{v}_{ds}"] = {"n": int(len(rs)), "reject_rate": float(hit.mean()),
                                        "median_lapvar_cal": float(np.median(vals)),
                                        "ci95": ci,
                                        "ci95_type": f"group bootstrap ({nb} draws, seed {SEED})"}

    # shipped-vs-blurred acceptance, from the S0 stage only (no head here)
    s0_effect = {}
    for v in ("blur15", "blur30", "blur80"):
        rows = []
        for r in ev:
            rule_new, _ = six_rule(r["variants"][v])
            rule_old, _ = _decide_five(r["variants"][v])
            rows.append((rule_old, rule_new))
        s0_effect[v] = {
            "n": len(rows),
            "shipped_five_exit": int(sum(1 for a, _ in rows if a is not None)),
            "with_blur_rule_exit": int(sum(1 for _, b in rows if b is not None)),
            "caught_by_blur_only": int(sum(1 for a, b in rows
                                           if a is None and b == "BLUR")),
            "n_pos": len(rows),
        }

    # threshold / catch / false-reject trade-off curve, all thresholds from CLEAN
    # calibration-split quantiles (never from a blurred window, never from EVAL)
    calib_clean = np.asarray([r["variants"]["clean"]["lapvar_cal"] for r in recs
                              if r["split"] == "calib"], float)
    curve = []
    for q in OP_QUANTILES:
        t = float(np.percentile(calib_clean, 100.0 * q))
        row = {"clean_calib_quantile": q, "threshold": t,
               "eval_false_reject": float(np.mean(
                   [r["variants"]["clean"]["lapvar_cal"] < t for r in ev]))}
        for v in ("blur15", "blur30", "blur80", "blur30_bright"):
            vals = np.asarray([r["variants"][v]["lapvar_cal"] for r in ev], float)
            row[f"catch_{v}"] = float(np.mean(vals < t))
        curve.append(row)

    # reference arm: the Youden-J threshold chosen on CALIB (not the frozen rule)
    t_y = float(cal["youden_j_on_calib"]["threshold"])
    youden_arm = {
        "threshold": t_y, "selection": cal["youden_j_on_calib"]["note"],
        "eval_false_reject": float(np.mean(
            [r["variants"]["clean"]["lapvar_cal"] < t_y for r in ev])),
    }
    for v in ("blur15", "blur30", "blur80", "blur30_bright"):
        vals = np.asarray([r["variants"][v]["lapvar_cal"] for r in ev], float)
        youden_arm[f"catch_{v}"] = float(np.mean(vals < t_y))

    head = json.load(open(os.path.join(OUT, "grid09_head.json"))) \
        if os.path.exists(os.path.join(OUT, "grid09_head.json")) else None
    head_youden = None
    if head is not None:
        head_youden = {"threshold": t_y, "arms": {}}
        for v in head["variants"]:
            rs = [r for r in head["windows"] if r["variant"] == v]
            exit_s0 = [r for r in rs if r["shipped_five_rule"] is not None
                       or r["lapvar_cal"] < t_y]
            still = [r for r in rs if r["shipped_five_rule"] is None
                     and r["lapvar_cal"] >= t_y and r["verdict"] == "accept"]
            head_youden["arms"][v] = {
                "n": len(rs), "s0_exit": len(exit_s0),
                "still_accepted": len(still),
                "still_accepted_ids": [r["id"] for r in still],
                "max_score_of_still_accepted": (max(r["score"] for r in still)
                                                if still else None),
            }
    payload = {
        "id": "GRID.09",
        "status": "done",
        "n": len(ev), "n_bad": len(ev) * 3,
        "n_good": len(ev),
        "n_note": ("n = clean human-good EVAL-split windows (the false-reject population); "
                   "n_bad = synthetic blurred positives, n x 3 sigmas (1.5 / 3 / 8) = "
                   f"{len(ev) * 3}. The brightness-shifted controls (blur x0.5, blur x1.4, "
                   "dark, bright) are extra variants reported separately, not counted in "
                   "n_bad."),
        "n_positives_by_sigma": {k: int(len(ev)) for k in ("blur15", "blur30", "blur80")},
        "label_source": ("clean windows: human gold, ONE labeller (IndEgo-150: 100 good / 50 bad) "
                         "and human-labelled Egocentric-10K (160: 138 good / 22 bad). "
                         "Defect positives are SYNTHETIC (Gaussian blur injected in memory) "
                         "on human-GOOD windows, never presented as human-labelled defects."),
        "protocol": (f"grouped even/odd split by clip (IndEgo) and factory (OOD-160); "
                     f"threshold chosen on the CALIB split only, evaluated on EVAL; "
                     f"{len(recs)} human-good windows decoded once, 8 variants each"),
        "ci95": aurocs["blur30"]["pooled"]["ci95"],
        "ci95_type": aurocs["blur30"]["pooled"]["ci95_type"],
        "device": C.DEVICE_CPU,
        "cost": f"{len(recs)} window decodes x 8 variants, {args.workers} worker(s), "
                f"nice -n 10, OMP_NUM_THREADS=2; scratch-free (no pixels written)",
        "command": ("JADAL_BUNDLE=<bundle> nice -n 10 python eval/exp_grid/grid09.py "
                    "--phase probe && python eval/exp_grid/grid09.py --phase calibrate && "
                    "python eval/exp_grid/grid09.py --phase head && "
                    "python eval/exp_grid/grid09.py --phase evaluate"),
        "git_sha": C.git_sha(),
        "seed": SEED,
        "config": {
            "variants": {k: {"sigma": v[0], "ksize": (None if v[0] is None
                                                      else ksize_for(v[0])),
                             "gain": v[1]} for k, v in VARIANTS.items()},
            "probe": "scripts/poc_run.py::s0_from_luma (production), 80x45 grey, first 60 frames",
            "probe_source_seconds": PROBE_SOURCE_S,
            "work_res": list(WORK),
            "split_rule": SPLITS,
            "frr_budget": FRR_BUDGET,
            "frozen_threshold": thr,
            "n_boot": N_BOOT,
            "downscale": "cv2 INTER_AREA to 160x90 (production ffmpeg scale=160:90 is bicubic)",
        },
        "config_sha256": C.config_sha256({"variants": VARIANTS, "thr": thr,
                                          "split": SPLITS, "frr": FRR_BUDGET,
                                          "seed": SEED, "probe": PROBE_SOURCE_S}),
        "threshold": {"value": thr, "source": "grid09_calibration.json",
                      "rule": cal["selection_rule"],
                      "n_calib": cal["n_calib_windows"],
                      "youden_reference": cal["youden_j_on_calib"],
                      "calibration_split": "CALIB = even-index IndEgo clips and "
                                           "OOD-160 factories; EVAL = odd index",
                      "shipped_default_threshold": shipped_default,
                      "rule_enabled_by_default": rule_enabled_by_default,
                      "default_note": "BLUR_LAPVAR_CAL_MIN ships 0.0 (DISABLED): the "
                                      "shipped five rules decide exactly as before. "
                                      "Every number in this file is at the CALIBRATED "
                                      "threshold above; flipping that one constant "
                                      "enables the rule."},
        "split": {"rule": SPLITS,
                  "n_calib": int(sum(1 for r in recs if r["split"] == "calib")),
                  "n_eval": len(ev),
                  "calib_groups": cal["calib_groups"],
                  "eval_groups": cal["eval_groups"]},
        "catch_rate_by_sigma_eval": per_sigma,
        "auroc_blurred_vs_clean_eval": aurocs,
        "operating_point_curve": curve,
        "youden_reference_arm_on_eval": youden_arm,
        "ablation_statistic_eval": ab,
        "false_reject_clean_good": frr,
        "brightness_only_control": bright_ctrl,
        "s0_stage_effect_on_blurred_eval": s0_effect,
        "shipped_head_arm": head,
        "shipped_head_arm_with_youden_threshold": head_youden,
        "confounds": [
            "Downscale is cv2 INTER_AREA; production decodes 160x90 with ffmpeg bicubic.",
            "Positives are synthetic blur on good windows; no real-world defocus footage was labelled.",
            "The IndEgo-only eval split has 3 clips, so its group interval is fragile; the"
            " per-dataset catch rates differ a lot (IndEgo 100% vs OOD-160 46% at sigma=3)"
            " and are therefore reported separately.",
            "IndEgo is 10 fps and is frame-repeated to 30 fps exactly as the portal does; OOD-160 is 30 fps native.",
            "lapvar_cal divides by the mean luma, so it stays PROPORTIONAL to gain (raw"
            " Laplacian variance scales with gain^2).  It is not gain-invariant: the dark"
            " control shows that a x0.5 darkening of a sharp good window can trip the"
            " rule -- the residual weakness of this statistic, reported, not hidden.",
            "One pooled threshold across two heterogeneous datasets: the per-dataset 2%"
            " clean quantiles were 0.0281 (IndEgo) and 0.0184 (OOD-160), so per-dataset"
            " operating points are an open alternative that was not shipped.",
            "The shipped-head arm runs 16 windows (8 IndEgo + 8 OOD, eval split) because"
            " the full 26-feature path costs ~0.4 s/window; its rates have n=16.",
        ],
        "not_measured": [
            "No real blurred capture was available; all positives are synthetic.",
            "The mosaic path's blur threshold is calibrated separately (GRID.05); the two"
            " values are not comparable.",
            "jadal/poc_stream/replay.py has its own S0 copy and was not changed.",
            "No other blur family (defocus disc, motion smear) was tested; Gaussian only.",
        ],
    }
    C.save_results(RESULTS, payload)
    print(json.dumps({
        "auroc_blur30_pooled_eval": aurocs["blur30"]["pooled"],
        "catch_blur30_pooled_eval": per_sigma["blur30"]["pooled"],
        "catch_blur80_pooled_eval": per_sigma["blur80"]["pooled"],
        "false_reject_pooled_all": frr["pooled_all"],
    }, indent=1, sort_keys=True))
    print(f"wrote {RESULTS}")


def _decide_five(probe):
    """The five SHIPPED rules only (blur rule removed) — the current behaviour."""
    for rule, key, op, thr in CASC.S0_RULES[:5]:
        if key not in probe:
            continue
        val = float(probe[key])
        if (val < thr) if op == "<" else (val > thr):
            return rule, {"feature": key, "value": round(val, 6)}
    return None, None


# --------------------------------------------------------------- head phase ----
def decode_source_rgb(path: str, start_s: float, seconds: float):
    """Decode ``seconds`` of source RGB uint8 (n,H,W,3) + native fps.

    BGR frames are reversed to RGB once and the defect is injected on these
    source pixels (see ``apply_variant_rgb``); only then is the frame downscaled
    to the 160x90 working resolution, exactly like the shipped path.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)
    want = max(1, int(round(seconds * fps)))
    out = []
    while len(out) < want:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        out.append(np.ascontiguousarray(fr[..., ::-1]))
    cap.release()
    if len(out) < 2:
        raise RuntimeError(f"no frames {path}@{start_s}")
    return out, float(fps)


def phase_head(args):
    """Shipped full path on a subset: 26 features + S1 head, clean vs blurred.

    The defect is injected at source resolution BEFORE the 160x90 downscale, so
    this reproduces what the shipped pipeline does with a blurred upload.
    """
    thr = float(s0_blur.BLUR_LAPVAR_CAL_MIN_CALIBRATED)
    if thr <= 0.0:
        raise SystemExit(
            "BLUR_LAPVAR_CAL_MIN_CALIBRATED is not frozen: run --phase calibrate and "
            "record the threshold, then run the head arm so the split is real")
    if float(s0_blur.BLUR_LAPVAR_CAL_MIN) <= 0.0:
        print("head arm: the SHIPPED default is 0.0 (rule disabled); the before/after "
              f"numbers below use the CALIBRATED threshold {thr} explicitly", flush=True)
    g = C.load_gold()
    o = C.load_ood160()
    _d, recs = _load_cache()
    by_id = {r["id"]: r for r in recs}
    ood_rel = {w["window"]: w["video"] for w in
               json.load(open(C._ood160_manifest(C.BUNDLE)))["windows"]}
    picks, seen = [], {"indego": 0, "ood160": 0}
    for i, wid in enumerate(g["ids"]):
        if int(g["y"][i]) != 1 or by_id.get(wid, {}).get("split") != "eval":
            continue
        if seen["indego"] >= args.head_n:
            break
        seen["indego"] += 1
        picks.append((wid, "indego", g["path"][i], float(g["start_s"][i])))
    for i, wid in enumerate(o["ids"]):
        if int(o["y"][i]) != 1 or by_id.get(wid, {}).get("split") != "eval":
            continue
        if seen["ood160"] >= args.head_n:
            break
        seen["ood160"] += 1
        rel = ood_rel[wid]
        picks.append((wid, "ood160",
                      rel if os.path.isabs(rel) else os.path.join(C.BUNDLE, rel), 0.0))
    print(f"head arm: {len(picks)} windows (eval split, human-good)", flush=True)
    clf, scaler, head_sha, head_doc = CASC.train_head()
    head_variants = ("clean", "blur30", "blur80")
    rows = []
    t0 = time.time()
    for wid, ds, path, start in picks:
        src, fps = decode_source_rgb(path, start, 10.0)
        rep = max(1, int(round(30.0 / fps))) if fps < 29.5 else 1
        idx = np.repeat(np.arange(len(src)), rep)[:300]
        small = {v: [] for v in head_variants}
        for k in idx:
            for v in head_variants:
                rgb = apply_variant_rgb(src[k], v)
                small[v].append(cv2.resize(rgb, (WORK[0], WORK[1]),
                                           interpolation=cv2.INTER_AREA))
        for v in head_variants:
            full = np.stack(small[v]).astype(np.uint8)
            if len(full) < 300:                  # pad exactly like the portal
                full = np.concatenate([full, np.repeat(full[-1:], 300 - len(full))])
            f64 = full.astype(np.float64) / 255.0
            lum = poc_run.luma_of(f64)
            s0 = poc_run.s0_from_luma(lum[:60])
            sub = poc_run.boxcar_to_10fps(f64)
            feats = CASC.cpu_window_features_alias(lum, sub, f64)
            verdict, stage, reasons, score = CASC.decide_window(s0, feats, clf, scaler)
            rule5, _ = _decide_five(s0)
            rules6 = [tuple(r) for r in CASC.S0_RULES[:5]] + [("BLUR", "lapvar_cal", "<", thr)]
            rule6, _ = P.decide_with(s0, rules6)
            rows.append({"id": wid, "dataset": ds, "variant": v,
                         "verdict": verdict, "stage": stage,
                         "score": score, "fired": [r["code"] for r in reasons],
                         "shipped_five_rule": rule5,
                         "with_calibrated_blur_rule": rule6,
                         "lapvar_cal": s0["lapvar_cal"]})
        print(f"  {wid} {ds} {time.time()-t0:.0f}s", flush=True)

    def summ(v):
        rs = [r for r in rows if r["variant"] == v]
        acc = [r for r in rs if r["verdict"] == "accept"]
        sc = [r["score"] for r in rs if r["score"] is not None]
        return {"n": len(rs), "accept": len(acc),
                "accept_rate": len(acc) / len(rs) if rs else float("nan"),
                "median_score": float(np.median(sc)) if sc else None,
                "max_score": float(np.max(sc)) if sc else None,
                "score_n": len(sc),
                "s0_exit_shipped_five": int(sum(1 for r in rs
                                                if r["shipped_five_rule"] is not None)),
                "s0_exit_with_calibrated_blur_rule": int(sum(
                    1 for r in rs
                    if r["with_calibrated_blur_rule"] is not None))}

    payload = {"id": "GRID.09", "phase": "head", "n_windows": len(picks),
               "windows": rows, "variants": list(head_variants),
               "summary": {v: summ(v) for v in head_variants},
               "head_sha256": head_sha, "head_n_train": head_doc["n_train"],
               "calibrated_blur_threshold_used": thr,
               "shipped_default_blur_threshold": float(s0_blur.BLUR_LAPVAR_CAL_MIN),
               "rule_enabled_by_default": bool(s0_blur.BLUR_LAPVAR_CAL_MIN > 0.0),
               "note": ("median_score is the shipped logistic head's p(usable); "
                        "accept requires p >= 0.75 (threshold 0.6, margin 0.25)"),
               "git_sha": C.git_sha(), "device": C.DEVICE_CPU,
               "command": ("JADAL_BUNDLE=<bundle> python eval/exp_grid/grid09.py "
                           f"--phase head --head-n {args.head_n}")}
    with open(os.path.join(OUT, "grid09_head.json"), "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print(json.dumps(payload["summary"], indent=1, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=("probe", "calibrate", "head", "evaluate", "all"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--head-n", type=int, default=8)
    args = ap.parse_args()
    if args.workers > 2:
        raise SystemExit("resource budget: at most 2 worker processes")
    if args.phase in ("probe", "all"):
        phase_probe(args)
    if args.phase in ("calibrate", "all"):
        phase_calibrate(args)
    # NOTE: the head arm is deliberately NOT part of `all`: it must run after the
    # threshold constant is frozen from the calibration phase.
    if args.phase == "head":
        phase_head(args)
    if args.phase in ("evaluate", "all"):
        phase_evaluate(args)


if __name__ == "__main__":
    main()
