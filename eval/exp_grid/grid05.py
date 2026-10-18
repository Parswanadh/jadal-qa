#!/usr/bin/env python3
"""GRID.05 — S0 parity between the shipped 2-second probe and the mosaic canvas.

The mosaic path cannot call the shipped probe (it scores one composite canvas,
not a frame stream), so ``jadal.mosaic.s0_parity`` rebuilds the same probe
dictionary from canvas statistics.  This experiment measures whether the two
verdicts agree.

Three test sets
---------------
1. SYNTHETIC DEFECT CLIPS with known labels, generated under ``/tmp/opencode/``
   (never in the repo) from IndEgo source windows.  The pixel defects come from
   ``scripts/make_demo_clips.py`` (good / blurred / too_dark / overexposed /
   frozen); ``black`` and ``lens_blocked`` are added here because the shipped
   generator does not have them, and their construction is recorded in the
   results JSON.  Source windows are taken from CALIB-split clips only, so the
   EVAL-split clean windows stay untouched.
2. The 12 UNLABELLED clips in ``interview/demo_kit/``: no labels exist, so only
   the per-clip verdicts and the agreement are reported, never a catch rate.
3. CLEAN human-good windows, no defect and three in-memory variants (clean /
   frozen / sigma=3 blur mosaic), for the false-reject side and an independent
   synthetic catch arm.

Protocol (pre-registered)
-------------------------
* The four rules whose statistic is unit-identical on both paths (BLACK,
  TOO_DARK, LENS_BLOCKED, OVEREXPOSED) inherit the SHIPPED thresholds unchanged.
* FROZEN and BLUR are calibrated on CLEAN data only: same grouped even/odd split
  by clip/factory as GRID.09, threshold = 2nd percentile of the clean CALIB
  distribution.  The frozen constants live in ``jadal/mosaic/s0_parity.py`` and
  are asserted against ``grid05_calibration.json`` before evaluation.
* The production path is ``scripts/poc_run.py::s0_from_luma`` +
  ``jadal.poc.cascade.s0_decide`` (the five shipped rules, plus GRID.09's BLUR
  rule which is part of the rule table in this branch).

Phases
------
  --phase clips      generate the defect clips, compare both paths -> grid05_clips.json
  --phase windows    clean windows x {clean, frozen, blur3} -> grid05_windows.json
  --phase calibrate  choose the two mosaic thresholds -> grid05_calibration.json
  --phase evaluate   parity + catch + false-reject tables -> GRID.05.json
  --phase all        clips, windows, calibrate (evaluate must follow the freeze)

No pixels are committed; the scratch directory is under /tmp/opencode/.
"""
from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_grid"))

import common as C  # noqa: E402
import cv2  # noqa: E402
import grid_util as GU  # noqa: E402
import make_demo_clips as M  # noqa: E402
from jadal.mosaic import s0_parity as P  # noqa: E402
from jadal.poc import cascade as CASC  # noqa: E402
from jadal.poc import s0_blur  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results", "exp_grid")
CLIP_DIR = "/tmp/opencode/grid05_clips"
DEMO_KIT = "/home/parshu/projects/awing/interview/demo_kit"
CACHE_CLIPS = os.path.join(OUT, "grid05_clips.json")
CACHE_WINDOWS = os.path.join(OUT, "grid05_windows.json")
CALIB = os.path.join(OUT, "grid05_calibration.json")
RESULTS = os.path.join(OUT, "GRID.05.json")

SEED = 0
N_TILES = 9
TILE_PX = 224
WINDOW_S = 10.0
FRR_BUDGET = 0.02
SPLIT_RULE = "even/odd index over the sorted unique groups of each dataset (even -> calib, odd -> eval)"

#: defects generated here that scripts/make_demo_clips.py does not have
ADDED_VARIANTS = (
    ("black", "black", {"op": "constant_frame", "value": 0,
                        "why": "all-zero frames: BLACK at the production rule"}),
    ("lens_blocked", "lens_blocked", {"op": "constant_frame", "value": 110,
                                      "why": "uniform mid-grey: zero spatial variance, "
                                             "not dark/bright enough for the other rules"}),
)
#: the pixel defects compared (container defects of make_demo_clips are skipped:
#: neither path is a container validator)
DEFECT_VARIANTS = ("good", "blurred", "too_dark", "overexposed", "frozen",
                   "black", "lens_blocked")
N_WINDOWS_PER_SOURCE = 2
CLIP_SECONDS = 12.0


# ------------------------------------------------------------- clip making ----
def apply_variant_ext(frames, variant):
    """``make_demo_clips.apply_variant`` + the two defects it does not have."""
    for name, _defect, params in ADDED_VARIANTS:
        if variant == name:
            return [np.full_like(f, int(params["value"])) for f in frames]
    return M.apply_variant(frames, variant)


def calib_source_windows():
    """Two good windows from each CALIB-split IndEgo clip (deterministic).

    CALIB clips only, so the EVAL-split clean windows are never inside the
    synthetic test clips.
    """
    g = C.load_gold()
    clips = sorted(set(str(c) for c in g["clip"]))
    calib_clips = [c for i, c in enumerate(clips) if i % 2 == 0]
    out = []
    for c in calib_clips:
        rows = [(float(g["start_s"][i]), g["path"][i])
                for i in range(len(g["ids"]))
                if str(g["clip"][i]) == c and int(g["y"][i]) == 1]
        for start, path in sorted(rows)[:N_WINDOWS_PER_SOURCE]:
            out.append({"clip": c, "path": path, "start_s": start})
    return out


def phase_clips(args):
    os.makedirs(CLIP_DIR, exist_ok=True)
    sources = calib_source_windows()
    entries = []
    for si, src in enumerate(sources):
        frames, _start_frame, size, src_fps = M.read_window(
            src["path"], src["start_s"], CLIP_SECONDS)
        for v in DEFECT_VARIANTS:
            defect = {"good": "none"}.get(v, v)
            params = {}
            for name, d, prm in ADDED_VARIANTS:
                if name == v:
                    defect, params = d, prm
            for vv, dd, _lab, prm in M.VARIANTS:
                if vv == v:
                    defect, params = dd, prm
            name = f"s{si}_{src['clip']}_{v}.mp4"
            path = os.path.join(CLIP_DIR, name)
            M.write_frames(apply_variant_ext(frames, v), path)
            entries.append({
                "file": name, "defect": defect, "variant": v,
                "parameters": params, "source_clip": src["clip"],
                "source_window_start_s": src["start_s"], "source_fps": src_fps,
                "frame_size": [size[0], size[1]], "fps": M.FPS,
                "n_frames": len(frames),
                "sha256": M.sha256_file(path),
                "label": "SYNTHETIC-GOOD" if v == "good" else "SYNTHETIC-DEFECT",
            })
        print(f"  source {si+1}/{len(sources)} ({src['clip']}) written", flush=True)
    for e in entries:
        p = os.path.join(CLIP_DIR, e["file"])
        if e["variant"] == "good" or M.is_decodable(p):
            continue
        raise RuntimeError(f"generated clip does not decode: {e['file']}")

    # the 12 unlabelled demo-kit clips
    demo = [{"file": os.path.basename(p), "path": p, "defect": None,
             "sha256": M.sha256_file(p)}
            for p in sorted(f"{DEMO_KIT}/{f}" for f in os.listdir(DEMO_KIT)
                            if f.endswith(".mp4"))]

    rows = []
    for e in entries:
        r = compare_clip(os.path.join(CLIP_DIR, e["file"]), e["file"],
                         "synthetic", e["defect"])
        r.update({"sha256": e["sha256"], "source_clip": e["source_clip"],
                  "variant": e["variant"], "label": e["label"],
                  "parameters": e["parameters"]})
        rows.append(r)
    drows = []
    for e in demo:
        r = compare_clip(e["path"], e["file"], "demo_kit", None)
        r["sha256"] = e["sha256"]
        drows.append(r)
    payload = {"id": "GRID.05", "phase": "clips",
               "scratch_dir": CLIP_DIR,
               "generator": "scripts/make_demo_clips.py helpers (read_window/write_frames/"
                            "apply_variant/is_decodable) + black/lens_blocked added here",
               "generated": entries, "synthetic": rows, "demo_kit": drows,
               "probe_config": {"n_tiles": N_TILES, "tile_px": TILE_PX,
                                "window_s": WINDOW_S, "timestamps": True,
                                "mask_timestamp": True},
               "git_sha": C.git_sha(), "seed": SEED, "device": C.DEVICE_CPU,
               "command": "python eval/exp_grid/grid05.py --phase clips"}
    with open(CACHE_CLIPS, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print(f"wrote {CACHE_CLIPS}  ({len(rows)} synthetic + {len(drows)} demo-kit clips)")


def compare_clip(path: str, name: str, origin: str, defect):
    """Both probes on the SAME decoded clip: shipped 2 s probe + mosaic probe.

    Frames are decoded once at source resolution; the shipped probe gets the
    first 2 s (frame-repeated to 30 fps), the mosaic gets the 9 cell-centred tile
    frames from the 10 s window.  Both poolings are probed: ``strict`` (primary)
    and ``features`` (S1's letterbox mask, the ablation).
    """
    t0 = time.time()
    idx, times, fps, avail = GU.plan_clip_indices(path, 0.0, WINDOW_S, N_TILES)
    two_s_n = max(2, int(round(2.0 * fps)))
    keep = set(int(i) for i in idx) | set(range(two_s_n))
    frames, fps2, n_read = GU.decode_rgb(path, 0.0, WINDOW_S, keep=keep)
    missing = [int(i) for i in idx if int(i) not in frames]
    if missing:
        raise RuntimeError(f"{name}: planned frames missing after decode: {missing}")
    two_s = [frames[i] for i in sorted(frames) if i < two_s_n]
    prod = GU.production_probe_from_frames(two_s, fps)
    t_build = time.perf_counter()
    canvas, side = GU.build_mosaic_from_frames(
        frames, idx, times, n_tiles=N_TILES, tile_px=TILE_PX,
        window_id=os.path.splitext(name)[0], source=origin, fps_used=N_TILES / WINDOW_S,
        timestamps=True)
    t_build = (time.perf_counter() - t_build) * 1000.0
    t_probe = time.perf_counter()
    mos = P.mosaic_probe(canvas, side, padding_detect="strict")
    mos_feat = P.mosaic_probe(canvas, side, padding_detect="features")
    t_probe = (time.perf_counter() - t_probe) * 1000.0
    five, ev5 = _five_decide(prod)
    six, ev6 = CASC.s0_decide(prod)
    inh, evi = P.inherited_s0_decide(mos)
    m6, evm = P.mosaic_s0_decide(mos)
    inh_f, _ = P.inherited_s0_decide(mos_feat)
    m6_f, _ = P.mosaic_s0_decide(mos_feat)
    return {
        "file": name, "origin": origin, "defect": defect,
        "native_fps": fps, "n_frames_available": avail, "n_frames_read": n_read,
        "production_probe": prod, "mosaic_probe": mos,
        "mosaic_probe_features_pooling": mos_feat,
        "production_five_rule": five, "production_five_evidence": ev5,
        "production_six_rule": six, "production_six_evidence": ev6,
        "mosaic_inherited_rule": inh, "mosaic_inherited_evidence": evi,
        "mosaic_six_rule": m6, "mosaic_six_evidence": evm,
        "mosaic_inherited_rule_features_pooling": inh_f,
        "mosaic_six_rule_features_pooling": m6_f,
        "agree_any_fire": (five is not None) == (inh is not None),
        "agree_exact_rule": five == inh,
        "agree_any_fire_full": (six is not None) == (m6 is not None),
        "agree_exact_rule_full": six == m6,
        "agree_any_fire_full_features_pooling": (six is not None) == (m6_f is not None),
        "ms_build": round(t_build, 1), "ms_mosaic_probe": round(t_probe, 1),
        "ms_total": round((time.time() - t0) * 1000.0, 1),
    }


def _five_decide(probe):
    """The five SHIPPED rules only (i.e. without GRID.09's blur rule)."""
    for rule, key, op, thr in CASC.S0_RULES[:5]:
        if key not in probe:
            continue
        val = float(probe[key])
        if (val < thr) if op == "<" else (val > thr):
            return rule, {"feature": key, "value": round(val, 6), "op": op,
                          "threshold": thr}
    return None, None


# ---------------------------------------------------------- clean windows ----
def _init_worker():
    os.nice(10)


def process_window(task):
    """One clean human-good window: clean / frozen / sigma=3 mosaics + probes."""
    wid, dataset, group, path, start_s = task
    idx, times, fps, avail = GU.plan_clip_indices(path, start_s, WINDOW_S, N_TILES)
    frames, fps2, n_read = GU.decode_rgb(path, start_s, WINDOW_S,
                                        keep=set(int(i) for i in idx))
    missing = [int(i) for i in idx if int(i) not in frames]
    if missing:
        raise RuntimeError(f"{wid}: planned frames missing after decode: {missing}")
    t_build = time.perf_counter()
    canvas, side = GU.build_mosaic_from_frames(
        frames, idx, times, n_tiles=N_TILES, tile_px=TILE_PX, window_id=wid,
        source=dataset, fps_used=N_TILES / WINDOW_S, timestamps=True)
    ms_build = (time.perf_counter() - t_build) * 1000.0
    t_probe = time.perf_counter()
    clean = P.mosaic_probe(canvas, side, padding_detect="strict")
    clean_feat = P.mosaic_probe(canvas, side, padding_detect="features")
    ms_probe = (time.perf_counter() - t_probe) * 1000.0

    frozen_frames = {int(k): frames[int(idx[0])] for k in idx}
    fc, fs = GU.build_mosaic_from_frames(
        frozen_frames, idx, times, n_tiles=N_TILES, tile_px=TILE_PX,
        window_id=wid + "_frozen", source=dataset,
        fps_used=N_TILES / WINDOW_S, timestamps=True)
    frozen = P.mosaic_probe(fc, fs, padding_detect="strict")
    frozen_feat = P.mosaic_probe(fc, fs, padding_detect="features")

    sig = 3.0
    k = 2 * int(np.ceil(3.0 * sig)) + 1
    blurred = {int(i): cv2.GaussianBlur(frames[int(i)], (k, k), sig) for i in idx}
    bc, bs = GU.build_mosaic_from_frames(
        blurred, idx, times, n_tiles=N_TILES, tile_px=TILE_PX,
        window_id=wid + "_blur3", source=dataset,
        fps_used=N_TILES / WINDOW_S, timestamps=True)
    blur = P.mosaic_probe(bc, bs, padding_detect="strict")
    blur_feat = P.mosaic_probe(bc, bs, padding_detect="features")
    return wid, {"id": wid, "dataset": dataset, "group": group,
                 "native_fps": fps, "n_frames_available": avail,
                 "n_frames_read": n_read,
                 "variants": {"clean": clean, "frozen": frozen, "blur30": blur},
                 "variants_features_pooling": {"clean": clean_feat,
                                               "frozen": frozen_feat,
                                               "blur30": blur_feat},
                 "ms_build": round(ms_build, 1), "ms_probe": round(ms_probe, 1)}


def phase_windows(args):
    g = C.load_gold()
    o = C.load_ood160()
    ood_rel = {w["window"]: w["video"] for w in
               json.load(open(C._ood160_manifest(C.BUNDLE)))["windows"]}
    tasks = []
    for i, wid in enumerate(g["ids"]):
        if int(g["y"][i]) != 1:
            continue
        tasks.append((wid, "indego", str(g["clip"][i]), g["path"][i],
                      float(g["start_s"][i])))
    for i, wid in enumerate(o["ids"]):
        if int(o["y"][i]) != 1:
            continue
        rel = ood_rel[wid]
        tasks.append((wid, "ood160", str(o["factory"][i]),
                      rel if os.path.isabs(rel) else os.path.join(C.BUNDLE, rel), 0.0))
    print(f"GRID.05 windows: {len(tasks)} human-good windows x 3 variants, "
          f"{args.workers} workers", flush=True)
    t0 = time.time()
    store, done = {}, 0
    if args.workers <= 1:
        _init_worker()
        for t in tasks:
            wid, rec = process_window(t)
            store[wid] = rec
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker) as ex:
            for wid, rec in ex.map(process_window, tasks, chunksize=2):
                store[wid] = rec
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    recs = list(store.values())
    for r in recs:
        ds = r["dataset"]
        groups = sorted({x["group"] for x in recs if x["dataset"] == ds})
        r["split"] = "calib" if groups.index(r["group"]) % 2 == 0 else "eval"
    payload = {"id": "GRID.05", "phase": "windows", "n_windows": len(store),
               "split_rule": SPLIT_RULE, "seed": SEED,
               "probe_config": {"n_tiles": N_TILES, "tile_px": TILE_PX,
                                "window_s": WINDOW_S, "timestamps": True,
                                "mask_timestamp": True},
               "variants": {"clean": "no defect",
                            "frozen": "all 9 tiles = tile 0",
                            "blur30": "sigma=3.0 Gaussian at source res, ksize 19"},
               "git_sha": C.git_sha(), "device": C.DEVICE_CPU,
               "command": "python eval/exp_grid/grid05.py --phase windows",
               "windows": store}
    with open(CACHE_WINDOWS, "w") as f:
        json.dump(payload, f, sort_keys=True)
    print(f"wrote {CACHE_WINDOWS} ({len(store)} windows, {time.time()-t0:.0f}s)")


# -------------------------------------------------------------- calibration ----
def _wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def phase_calibrate(args):
    d = json.load(open(CACHE_WINDOWS))
    recs = list(d["windows"].values())
    calib = [r for r in recs if r["split"] == "calib"]

    def thresholds(key):
        te = np.asarray([r[key]["clean"]["temp_e"] for r in calib], float)
        lv = np.asarray([r[key]["clean"]["lapvar_cal"] for r in calib], float)
        return (float(np.percentile(te, 100.0 * FRR_BUDGET)),
                float(np.percentile(lv, 100.0 * FRR_BUDGET)))

    thr_te, thr_lv = thresholds("variants")
    thr_te_f, thr_lv_f = thresholds("variants_features_pooling")

    def youden(vkey, key, pos_key):
        pos = np.asarray([r[vkey][pos_key][key] for r in calib], float)
        neg = np.asarray([r[vkey]["clean"][key] for r in calib], float)
        best = (-1.0, None)
        for t in np.unique(np.round(np.concatenate([pos, neg]), 10)):
            j = float((pos < t).mean()) + float((neg >= t).mean()) - 1.0
            if j > best[0]:
                best = (j, float(t))
        return {"threshold": best[1], "j": best[0],
                "note": "reference arm only; not the frozen rule"}

    per_ds = {}
    for ds in sorted({r["dataset"] for r in calib}):
        v = np.asarray([r["variants"]["clean"]["lapvar_cal"] for r in calib
                        if r["dataset"] == ds], float)
        t = np.asarray([r["variants"]["clean"]["temp_e"] for r in calib
                        if r["dataset"] == ds], float)
        leak = np.asarray([r["variants"]["clean"]["bg_leak_frac"] for r in calib
                           if r["dataset"] == ds], float)
        leakf = np.asarray([r["variants_features_pooling"]["clean"]["bg_leak_frac"]
                            for r in calib if r["dataset"] == ds], float)
        per_ds[ds] = {"n": int(len(v)),
                      "lapvar_cal_p02": float(np.percentile(v, 2.0)),
                      "lapvar_cal_p50": float(np.median(v)),
                      "temp_e_p02": float(np.percentile(t, 2.0)),
                      "temp_e_p50": float(np.median(t)),
                      "bg_leak_frac_median_strict": float(np.median(leak)),
                      "bg_leak_frac_median_features": float(np.median(leakf))}
    payload = {
        "id": "GRID.05", "phase": "calibrate",
        "n_calib_windows": len(calib),
        "n_calib_groups": len({(r["dataset"], r["group"]) for r in calib}),
        "frr_budget": FRR_BUDGET,
        "selection_rule": ("largest threshold whose clean-only false-reject rate on the "
                           "CALIB split is <= 2% = 2nd percentile of the clean CALIB "
                           "distribution (applies to FROZEN and BLUR alike)"),
        "primary_pooling": "strict",
        "selected_temp_e_max": thr_te,
        "selected_lapvar_cal_min": thr_lv,
        "ablation_features_pooling": {
            "selected_temp_e_max": thr_te_f,
            "selected_lapvar_cal_min": thr_lv_f,
            "note": "S1's features.letterbox_mask: padding under the timestamp box leaks"},
        "youden_j_on_calib": {
            "frozen_strict": youden("variants", "temp_e", "frozen"),
            "blur_strict": youden("variants", "lapvar_cal", "blur30"),
            "frozen_features_pooling": youden("variants_features_pooling", "temp_e",
                                              "frozen"),
            "blur_features_pooling": youden("variants_features_pooling", "lapvar_cal",
                                            "blur30"),
        },
        "per_dataset": per_ds,
        "split_rule": SPLIT_RULE,
        "calib_groups": sorted({f"{r['dataset']}:{r['group']}" for r in calib}),
        "eval_groups": sorted({f"{r['dataset']}:{r['group']}" for r in recs
                               if r["split"] == "eval"}),
        "git_sha": C.git_sha(), "seed": SEED, "device": C.DEVICE_CPU,
        "command": "python eval/exp_grid/grid05.py --phase calibrate",
    }
    with open(CALIB, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print(json.dumps({k: payload[k] for k in
                      ("selected_temp_e_max", "selected_lapvar_cal_min",
                       "ablation_features_pooling", "youden_j_on_calib",
                       "per_dataset")}, indent=1, sort_keys=True))
    print(f"wrote {CALIB}")


# ----------------------------------------------------------------- evaluate ----
def phase_evaluate(args):
    if not os.path.exists(CALIB):
        raise SystemExit("run --phase calibrate first")
    cal = json.load(open(CALIB))
    if abs(P.MOSAIC_TEMP_E_MAX - float(cal["selected_temp_e_max"])) > 1e-15 or \
       abs(P.MOSAIC_BLUR_LAPVAR_CAL_MIN - float(cal["selected_lapvar_cal_min"])) > 1e-15:
        raise SystemExit(
            f"frozen constants ({P.MOSAIC_TEMP_E_MAX}, {P.MOSAIC_BLUR_LAPVAR_CAL_MIN}) "
            f"do not match the calibration ({cal['selected_temp_e_max']}, "
            f"{cal['selected_lapvar_cal_min']}); freeze them before evaluating")

    cd = json.load(open(CACHE_CLIPS))
    demo = cd["demo_kit"]
    wd = json.load(open(CACHE_WINDOWS))
    recs = list(wd["windows"].values())
    abl = cal["ablation_features_pooling"]

    def feat_rules():
        return list(P.INHERITED_RULES) + [
            ("FROZEN", "temp_e", "<", float(abl["selected_temp_e_max"])),
            ("BLUR", "lapvar_cal", "<", float(abl["selected_lapvar_cal_min"])),
        ]

    def feat_rule_of(probe):
        return P.decide_with(probe, feat_rules())[0]

    def mosaic_five_rules():
        """The mosaic's answer to the SAME FIVE shipped rules (no blur rule).

        This is the apples-to-apples parity target: the four inherited rules with
        their shipped thresholds, plus FROZEN at its separately calibrated mosaic
        threshold.  Comparing the production five-rule verdict against the
        four-rule inherited subset instead would count the FROZEN defect as a
        disagreement by construction.
        """
        return list(P.INHERITED_RULES) + [("FROZEN", "temp_e", "<",
                                           P.MOSAIC_TEMP_E_MAX)]

    def mosaic_five_rule_of(probe):
        return P.decide_with(probe, mosaic_five_rules())[0]

    def decorate(r):
        """Derive every verdict HERE from the cached probe values.

        The clip cache stores statistics only; the decisions depend on the frozen
        thresholds, so they are recomputed at evaluation time rather than trusted
        from whatever the constants were when the cache was written.
        """
        prod = r["production_probe"]
        mos = r["mosaic_probe"]
        mf = r["mosaic_probe_features_pooling"]
        five, ev5 = _five_decide(prod)
        six, ev6 = CASC.s0_decide(prod)
        inh, evi = P.inherited_s0_decide(mos)
        m6, evm = P.mosaic_s0_decide(mos)
        m6f, evmf = P.decide_with(mf, feat_rules())
        out = dict(r)
        out.update({
            "production_five_rule": five, "production_five_evidence": ev5,
            "production_six_rule": six, "production_six_evidence": ev6,
            "mosaic_inherited_rule": inh, "mosaic_inherited_evidence": evi,
            "mosaic_six_rule": m6, "mosaic_six_evidence": evm,
            "mosaic_six_rule_features_pooling": m6f,
            "mosaic_six_evidence_features_pooling": evmf,
            "agree_any_fire": (five is not None) == (inh is not None),
            "agree_exact_rule": five == inh,
            "agree_any_fire_full": (six is not None) == (m6 is not None),
            "agree_exact_rule_full": six == m6,
            "agree_any_fire_full_features_pooling": (six is not None) == (m6f is not None),
            "mosaic_five_rule": mosaic_five_rule_of(mos),
            "agree_five_vs_mosaic_five": five == mosaic_five_rule_of(mos),
        })
        return out

    syn = [decorate(r) for r in cd["synthetic"]]
    demo = [decorate(r) for r in demo]

    # per-defect catch table on the labelled synthetic clips
    catch = {}
    for defect in ("blur", "too_dark", "overexposed", "frozen", "black",
                   "lens_blocked", "none"):
        rows = [r for r in syn if r["defect"] == defect]
        if not rows:
            continue
        n = len(rows)
        prod5 = sum(1 for r in rows if r["production_five_rule"] is not None)
        prod6 = sum(1 for r in rows if r["production_six_rule"] is not None)
        mos6 = sum(1 for r in rows if r["mosaic_six_rule"] is not None)
        mosf = sum(1 for r in rows
                   if r["mosaic_six_rule_features_pooling"] is not None)
        catch[defect] = {
            "n": n,
            "n_source_clips": len({r["source_clip"] for r in rows}),
            "production_five_rule_caught": prod5,
            "production_with_blur_caught": prod6,
            "mosaic_caught": mos6,
            "mosaic_caught_features_pooling": mosf,
            "production_five_rate": prod5 / n, "production_five_ci95": _wilson(prod5, n),
            "production_with_blur_rate": prod6 / n,
            "production_with_blur_ci95": _wilson(prod6, n),
            "mosaic_rate": mos6 / n, "mosaic_rate_ci95": _wilson(mos6, n),
            "production_fired_rules": _hist(r["production_six_rule"] for r in rows),
            "mosaic_fired_rules": _hist(r["mosaic_six_rule"] for r in rows),
            "rule_match_rate": (sum(1 for r in rows if r["production_five_rule"]
                                    == r["mosaic_inherited_rule"]) / n),
            "agree_any_fire_rate": sum(1 for r in rows if r["agree_any_fire"]) / n,
            "agree_any_fire_full_rate": sum(1 for r in rows
                                            if r["agree_any_fire_full"]) / n,
            "agree_any_fire_full_rate_features_pooling":
                sum(1 for r in rows if r["agree_any_fire_full_features_pooling"]) / n,
            "ci95_type": "Wilson score interval over clips (binomial, ungrouped)",
        }

    # false-reject on clean human-good windows.  "extra" is relative to the
    # SHIPPED five rules on the SAME windows, joined from the GRID.09 probe cache
    # (same 238 windows, same session, production probe on the clean variant).
    prod5_by_id = {}
    g9 = os.path.join(OUT, "probe_blur_windows.json")
    if os.path.exists(g9):
        for wid, r in json.load(open(g9))["windows"].items():
            prod5_by_id[wid] = _five_decide(r["variants"]["clean"])[0]

    def frr(rows, label, vkey="variants"):
        n = len(rows)
        ids = [r["id"] for r in rows]
        have_prod = [i for i in ids if i in prod5_by_id]
        mos_any = [P.mosaic_s0_decide(r[vkey]["clean"])[0] for r in rows]
        mos_inh = [P.inherited_s0_decide(r[vkey]["clean"])[0] for r in rows]
        mos_fro = sum(1 for r in rows if r[vkey]["clean"]["temp_e"]
                      < P.MOSAIC_TEMP_E_MAX)
        mos_blu = sum(1 for r in rows if r[vkey]["clean"]["lapvar_cal"]
                      < P.MOSAIC_BLUR_LAPVAR_CAL_MIN)
        extra = sum(1 for r in rows
                    if P.mosaic_s0_decide(r[vkey]["clean"])[0] is not None
                    and prod5_by_id.get(r["id"]) is None)
        prod_rej = sum(1 for i in ids if prod5_by_id.get(i) is not None)
        disagree = sum(1 for r in rows
                       if P.inherited_s0_decide(r[vkey]["clean"])[0]
                       != _four_shipped(r["id"], prod5_by_id))
        leak = float(np.median([r[vkey]["clean"]["bg_leak_frac"] for r in rows]))
        return {"label": label, "n": n, "pooling": vkey,
                "n_groups": len({(r["dataset"], r["group"]) for r in rows}),
                "production_joined_windows": len(have_prod),
                "production_five_rule_rejects": prod_rej,
                "mosaic_inherited4_rejects": sum(1 for x in mos_inh if x is not None),
                "mosaic_inherited4_disagreements": disagree,
                "mosaic_frozen_rule_rejects": mos_fro,
                "mosaic_blur_rule_rejects": mos_blu,
                "mosaic_any_rejects": sum(1 for x in mos_any if x is not None),
                "mosaic_only_rejects": extra,
                "bg_leak_frac_median": leak,
                "false_reject_rate": sum(1 for x in mos_any if x is not None) / n
                if n else float("nan"),
                "false_reject_ci95": _wilson(sum(1 for x in mos_any if x is not None), n),
                "mosaic_only_false_reject_rate": extra / n if n else float("nan"),
                "mosaic_only_false_reject_ci95": _wilson(extra, n),
                "ci95_type": "Wilson score interval over windows (binomial, ungrouped)"}

    # ablation: S1's letterbox mask, with its own clean-only thresholds
    def frr_ablation(rows, label):
        n = len(rows)
        fired = [feat_rule_of(r["variants_features_pooling"]["clean"]) for r in rows]
        extra = sum(1 for r in rows
                    if feat_rule_of(r["variants_features_pooling"]["clean"]) is not None
                    and prod5_by_id.get(r["id"]) is None)
        return {"label": label, "n": n, "pooling": "features",
                "mosaic_any_rejects": sum(1 for x in fired if x is not None),
                "false_reject_rate": sum(1 for x in fired if x is not None) / n
                if n else float("nan"),
                "false_reject_ci95": _wilson(sum(1 for x in fired if x is not None), n),
                "mosaic_only_rejects": extra,
                "bg_leak_frac_median": float(np.median(
                    [r["variants_features_pooling"]["clean"]["bg_leak_frac"]
                     for r in rows])),
                "ci95_type": "Wilson score interval over windows (binomial, ungrouped)"}

    frr_tab = {"pooled_all": frr(recs, "all human-good windows"),
               "pooled_eval": frr([r for r in recs if r["split"] == "eval"],
                                  "EVAL-split human-good windows"),
               "indego_all": frr([r for r in recs if r["dataset"] == "indego"],
                                 "IndEgo-150 good windows"),
               "indego_eval": frr([r for r in recs if r["dataset"] == "indego"
                                   and r["split"] == "eval"], "IndEgo EVAL split"),
               "ood160_all": frr([r for r in recs if r["dataset"] == "ood160"],
                                 "OOD-160 good windows"),
               "ood160_eval": frr([r for r in recs if r["dataset"] == "ood160"
                                   and r["split"] == "eval"], "OOD-160 EVAL split")}
    frr_abl = {"pooled_all": frr_ablation(recs, "all human-good windows"),
               "pooled_eval": frr_ablation([r for r in recs if r["split"] == "eval"],
                                           "EVAL-split human-good windows"),
               "ood160_all": frr_ablation([r for r in recs if r["dataset"] == "ood160"],
                                          "OOD-160 good windows")}

    # independent synthetic catch arm on the same clean windows
    syn_arm = {}
    for v, rule in (("frozen", "FROZEN"), ("blur30", "BLUR")):
        for scope in ("eval", "all"):
            rows = [r for r in recs if scope == "all" or r["split"] == "eval"]
            fired = sum(1 for r in rows
                        if P.mosaic_s0_decide(r["variants"][v])[0] is not None)
            by_rule = _hist(P.mosaic_s0_decide(r["variants"][v])[0] for r in rows)
            syn_arm[f"{v}_{scope}"] = {
                "n": len(rows), "caught": fired,
                "catch_rate": fired / len(rows) if rows else float("nan"),
                "catch_ci95": _wilson(fired, len(rows)),
                "fired_rules": by_rule,
                "ci95_type": "Wilson score interval over windows (binomial, ungrouped)"}
    # empty scaffold so the field exists even if a future edit drops the loop
    if not syn_arm:  # pragma: no cover
        syn_arm = {}

    agree = {
        "n": len(syn),
        "agree_five_vs_mosaic_five_rate": sum(
            1 for r in syn if r["agree_five_vs_mosaic_five"]) / len(syn),
        "agree_five_vs_mosaic_five_ci95": _wilson(
            sum(1 for r in syn if r["agree_five_vs_mosaic_five"]), len(syn)),
        "agree_five_vs_mosaic_five_note": (
            "the FAIR five-rule parity target: the four inherited rules at their "
            "shipped thresholds plus FROZEN at its calibrated mosaic threshold"),
        "agree_five_vs_inherited4_fire_rate": sum(
            1 for r in syn if r["agree_any_fire"]) / len(syn),
        "agree_five_vs_inherited4_note": (
            "production five rules vs the mosaic's FOUR inherited rules only: FROZEN "
            "defects disagree by construction, so this is not a parity failure"),
        "agree_exact_rule_rate": sum(1 for r in syn if r["agree_exact_rule"]) / len(syn),
        "agree_exact_rule_ci95": _wilson(sum(1 for r in syn if r["agree_exact_rule"]),
                                         len(syn)),
        "agree_any_fire_full_rate": sum(1 for r in syn
                                        if r["agree_any_fire_full"]) / len(syn),
        "agree_any_fire_full_ci95": _wilson(sum(1 for r in syn
                                                if r["agree_any_fire_full"]), len(syn)),
        "agree_any_fire_full_rate_features_pooling":
            sum(1 for r in syn if r["agree_any_fire_full_features_pooling"]) / len(syn),
        "disagreements": [
            {"file": r["file"], "defect": r["defect"],
             "production": r["production_five_rule"],
             "mosaic_inherited": r["mosaic_inherited_rule"],
             "mosaic_five": r["mosaic_five_rule"],
             "production_with_blur": r["production_six_rule"],
             "mosaic_full": r["mosaic_six_rule"],
             "mosaic_full_features_pooling": r["mosaic_six_rule_features_pooling"],
             "bg_leak_frac_strict": r["mosaic_probe"]["bg_leak_frac"],
             "bg_leak_frac_features": r["mosaic_probe_features_pooling"]["bg_leak_frac"]}
            for r in syn if not r["agree_five_vs_mosaic_five"]
            or not r["agree_any_fire_full"]],
        "ci95_type": "Wilson score interval over clips (binomial, ungrouped)",
    }
    demo_disagree = [
        {"file": r["file"], "production": r["production_six_rule"],
         "mosaic": r["mosaic_six_rule"],
         "mosaic_features_pooling": r["mosaic_six_rule_features_pooling"],
         "production_p95": r["production_probe"]["p95"],
         "mosaic_p95": r["mosaic_probe"]["p95"],
         "mosaic_p95_features_pooling":
             r["mosaic_probe_features_pooling"]["p95"],
         "bg_leak_frac_strict": r["mosaic_probe"]["bg_leak_frac"],
         "bg_leak_frac_features": r["mosaic_probe_features_pooling"]["bg_leak_frac"],
         "n_frames_available": r["n_frames_available"]}
        for r in demo if not r["agree_any_fire_full"]]
    demo_tab = [{"file": r["file"], "native_fps": r["native_fps"],
                 "n_frames_available": r["n_frames_available"],
                 "production_five_rule": r["production_five_rule"],
                 "production_six_rule": r["production_six_rule"],
                 "mosaic_six_rule": r["mosaic_six_rule"],
                 "mosaic_six_rule_features_pooling":
                     r["mosaic_six_rule_features_pooling"],
                 "agree_any_fire_full": r["agree_any_fire_full"],
                 "agree_any_fire_full_features_pooling":
                     r["agree_any_fire_full_features_pooling"],
                 "production_lapvar_cal": round(r["production_probe"]["lapvar_cal"], 6),
                 "mosaic_lapvar_cal": round(r["mosaic_probe"]["lapvar_cal"], 6),
                 "production_temp_e": round(r["production_probe"]["temp_e"], 6),
                 "mosaic_it_mad": round(r["mosaic_probe"]["temp_e"], 6),
                 "production_p95": round(r["production_probe"]["p95"], 4),
                 "mosaic_p95": round(r["mosaic_probe"]["p95"], 4),
                 "mosaic_p95_features_pooling":
                     round(r["mosaic_probe_features_pooling"]["p95"], 4),
                 "bg_leak_frac_strict": round(r["mosaic_probe"]["bg_leak_frac"], 5),
                 "bg_leak_frac_features":
                     round(r["mosaic_probe_features_pooling"]["bg_leak_frac"], 5)}
                for r in demo]

    payload = {
        "id": "GRID.05",
        "status": "done",
        "n": len(syn) + len(demo), "n_bad": sum(1 for r in syn
                                                if r["defect"] not in (None, "none")),
        "n_good": sum(1 for r in syn if r["defect"] in (None, "none")),
        "label_source": ("synthetic defect clips generated locally with known injected "
                         "defects (SYNTHETIC-GOOD / SYNTHETIC-DEFECT, never presented as "
                         "human labels); the 12 interview/demo_kit clips are UNLABELLED "
                         "real uploads and are reported for agreement only; clean windows "
                         "are human gold (ONE labeller)"),
        "protocol": ("same decoded 480p pixels through both probes: shipped 2 s probe "
                     "(s0_from_luma + s0_decide) vs mosaic probe on the 3x3 224 px "
                     "canvas; the four unit-identical rules inherit the shipped "
                     "thresholds, FROZEN and BLUR are calibrated on the clean CALIB "
                     "split only and evaluated on the EVAL split"),
        "ci95": agree["agree_any_fire_full_ci95"],
        "ci95_type": agree["ci95_type"],
        "device": C.DEVICE_CPU,
        "cost": (f"{len(syn) + len(demo)} clip comparisons + {len(recs)} clean windows x 3 "
                 f"mosaic variants, 2 workers, nice -n 10, OMP_NUM_THREADS=2, "
                 f"scratch under {CLIP_DIR}"),
        "command": ("python eval/exp_grid/grid05.py --phase clips && "
                    "python eval/exp_grid/grid05.py --phase windows && "
                    "python eval/exp_grid/grid05.py --phase calibrate && "
                    "python eval/exp_grid/grid05.py --phase evaluate"),
        "git_sha": C.git_sha(), "seed": SEED,
        "config": {"n_tiles": N_TILES, "tile_px": TILE_PX, "window_s": WINDOW_S,
                   "timestamps": True, "mask_timestamp": True,
                   "split_rule": SPLIT_RULE, "frr_budget": FRR_BUDGET,
                   "frozen_temp_e_max": P.MOSAIC_TEMP_E_MAX,
                   "frozen_lapvar_cal_min": P.MOSAIC_BLUR_LAPVAR_CAL_MIN,
                   "inherited_rules": [list(r) for r in P.INHERITED_RULES],
                   "production_blur_threshold": s0_blur.BLUR_LAPVAR_CAL_MIN,
                   "added_defects": [{"variant": v, "defect": d, "parameters": p}
                                     for v, d, p in ADDED_VARIANTS],
                   "window_variants": {"frozen": "all 9 tiles = tile 0",
                                       "blur30": "sigma=3.0 Gaussian, ksize 19"}},
        "config_sha256": C.config_sha256(
            {"n_tiles": N_TILES, "tile_px": TILE_PX, "window_s": WINDOW_S,
             "split": SPLIT_RULE, "frr": FRR_BUDGET, "seed": SEED,
             "temp_e_max": P.MOSAIC_TEMP_E_MAX,
             "lapvar_min": P.MOSAIC_BLUR_LAPVAR_CAL_MIN,
             "prod_blur": s0_blur.BLUR_LAPVAR_CAL_MIN}),
        "thresholds": {
            "inherited_from_shipped": [list(r) for r in P.INHERITED_RULES],
            "pooling": "strict (padding detection that survives the timestamp box)",
            "mosaic_frozen_temp_e_max": {"value": P.MOSAIC_TEMP_E_MAX,
                                         "source": "grid05_calibration.json",
                                         "rule": cal["selection_rule"],
                                         "n_calib": cal["n_calib_windows"]},
            "mosaic_blur_lapvar_cal_min": {"value": P.MOSAIC_BLUR_LAPVAR_CAL_MIN,
                                           "source": "grid05_calibration.json",
                                           "rule": cal["selection_rule"],
                                           "n_calib": cal["n_calib_windows"]},
            "youden_reference_arm": cal["youden_j_on_calib"],
            "ablation_features_pooling": cal["ablation_features_pooling"],
            "production_blur_lapvar_cal_min": s0_blur.BLUR_LAPVAR_CAL_MIN,
        },
        "split": {"rule": SPLIT_RULE,
                  "n_calib": sum(1 for r in recs if r["split"] == "calib"),
                  "n_eval": sum(1 for r in recs if r["split"] == "eval"),
                  "calib_groups": cal["calib_groups"],
                  "eval_groups": cal["eval_groups"]},
        "catch_rate_synthetic_clips": catch,
        "mosaic_false_reject_clean": frr_tab,
        "mosaic_false_reject_clean_features_pooling_ablation": frr_abl,
        "mosaic_catch_on_window_variants": syn_arm,
        "agreement_synthetic_clips": agree,
        "decisions_note": ("every verdict in this file is derived at evaluation time from "
                           "the cached probe statistics and the frozen thresholds in "
                           "jadal/mosaic/s0_parity.py / jadal/poc/s0_blur.py, not read "
                           "back from the probe cache"),
        "demo_kit_clips": demo_tab,
        "demo_kit_disagreements": demo_disagree,
        "bg_leak_frac_median_by_dataset": {
            r["dataset"]: float(np.median(
                [x["variants"]["clean"]["bg_leak_frac"] for x in recs
                 if x["dataset"] == r["dataset"]]))
            for r in recs},
        "bg_leak_frac_median_features_pooling_by_dataset": {
            r["dataset"]: float(np.median(
                [x["variants_features_pooling"]["clean"]["bg_leak_frac"] for x in recs
                 if x["dataset"] == r["dataset"]]))
            for r in recs},
        "ms_mosaic_build_median": float(np.median([r["ms_build"] for r in recs])),
        "ms_mosaic_probe_median": float(np.median([r["ms_probe"] for r in recs])),
        "confounds": [
            "The synthetic defects are injected by this repo's own generator, not observed"
            " in the wild; the demo_kit clips carry no labels at all.",
            "Both paths decode the same cv2 frames here, so this measures the PROBES, not"
            " the two decode front ends (GRID.07 measures those).",
            "FROZEN and BLUR thresholds are not comparable with the GRID.09 full-pipeline"
            " values: different statistic units.",
            "Per-defect n is the number of generated clips (2 windows x 4 CALIB source"
            " clips), so the Wilson intervals are wide.",
            "The generator's too_dark (x0.25) and overexposed (gain 1.7 + 30) defects are"
            " milder than the shipped operating points on most source windows, so both"
            " paths miss them; that is a property of the injected strength, not a parity"
            " difference.",
            "Pooled luma percentiles use the STRICT padding detector; the features-pooling"
            " ablation is reported separately and is measurably contaminated on 16:9"
            " sources.",
        ],
        "not_measured": [
            "No human labels exist for the generated clips or for demo_kit; the labels are"
            " 'SYNTHETIC-GOOD'/'SYNTHETIC-DEFECT' by construction.",
            "Container defects of make_demo_clips (wrong_fps, truncated, garbage,"
            " reupload_duplicate) were skipped: neither probe is a container validator.",
            "No real lens-blocked or real defocus capture was available.",
        ],
    }
    C.save_results(RESULTS, payload)
    print(json.dumps({"agreement": {k: agree[k] for k in
                                    ("agree_any_fire_full_rate", "agree_exact_rule_rate")},
                      "catch": {k: {"n": v["n"], "prod": v["production_five_rate"],
                                    "mosaic": v["mosaic_rate"]}
                                for k, v in catch.items()},
                      "frr": {k: v["false_reject_rate"] for k, v in frr_tab.items()}},
                     indent=1, sort_keys=True))
    print(f"wrote {RESULTS}")


def _hist(rules):
    h = {}
    for r in rules:
        k = r if r is not None else "none"
        h[k] = h.get(k, 0) + 1
    return h


def _four_shipped(wid, prod5_by_id):
    """The shipped decision restricted to the four inherited rules' keys.

    ``prod5_by_id`` holds the shipped five-rule verdict; if that verdict is one of
    the four inherited codes it is returned, otherwise None.  FROZEN cannot be
    compared across the two paths (its statistic has different units), so it is
    deliberately excluded from the parity count.
    """
    four = {r[0] for r in P.INHERITED_RULES}
    v = prod5_by_id.get(wid)
    return v if v in four else None


def _prod_five(probe):
    return _five_decide(probe)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=("clips", "windows", "calibrate", "evaluate", "all"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.workers > 2:
        raise SystemExit("resource budget: at most 2 worker processes")
    if args.phase in ("clips", "all"):
        phase_clips(args)
    if args.phase in ("windows", "all"):
        phase_windows(args)
    if args.phase in ("calibrate", "all"):
        phase_calibrate(args)
    # evaluate runs only after the constants in jadal/mosaic/s0_parity.py are frozen
    if args.phase == "evaluate":
        phase_evaluate(args)


if __name__ == "__main__":
    main()
