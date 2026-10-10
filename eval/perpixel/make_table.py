"""M1: build unified feature tables for gold + synth, cache float32.

Outputs (eval/results/perpixel/):
  gold.npz   X float32 (150,F), ids, clips, y int8, reasons, names, T (frames/row)
  synth.npz  same for the 60 synthetic windows
  series_gold.npz  per-window (T,8) float32 series (object arrays: S_list, ids)
  dup_sigs.json    dup.signature for all 150 gold windows
  meta.json  feature definitions, units, resolutions, git sha, command
Regenerate: PYTHONNOUSERSITE=1 JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models \
  /home/parshu/envs/jadal/bin/python eval/perpixel/make_table.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

SPEC_DIR = "/home/parshu/projects/jadal-qa-wt/spectral_data"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "perpixel")


def _work(fn, args):
    from jadal.perpixel import features as F
    from jadal.perpixel import series as S
    path, start, end = args
    try:
        row, names, Smat, meta = F.extract_row(path, start, end)
        return (fn, True, row.astype(np.float64), names, Smat,
                meta, "")
    except Exception as e:  # noqa: BLE001 - record per-window failure
        return (fn, False, None, None, None, None, f"{type(e).__name__}: {e}")


def _range_for(it):
    if "start_s" in it:
        return it["path"], it["start_s"], it["end_s"]
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", it["path"]],
                       capture_output=True, text=True, timeout=60)
    return it["path"], 0.0, float(p.stdout.strip())


def build(manifest_name, out_npz, series_npz=None, workers=4):
    from concurrent.futures import ProcessPoolExecutor
    with open(os.path.join(SPEC_DIR, manifest_name)) as f:
        m = json.load(f)
    items = m["items"]
    t0 = time.time()
    rows, ok_ids, clips, ys, reasons, Ts, Slist = [], [], [], [], [], [], []
    names = None
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_work, it["id"], _range_for(it)): it
                for it in items}
        res = {}
        for fu in futs:
            fn, ok, row, nm, Sm, meta, err = fu.result()
            res[fn] = (ok, row, nm, Sm, meta, err)
    fails = []
    for it in items:
        ok, row, nm, Sm, meta, err = res[it["id"]]
        if not ok:
            fails.append({"id": it["id"], "err": err})
            continue
        if names is None:
            names = nm
        rows.append(row.astype(np.float32))
        Slist.append(Sm)
        ok_ids.append(it["id"])
        clips.append(it["clip"] if "clip" in it else it.get("source_id", ""))
        ys.append(it["label"])
        reasons.append(it.get("reason", it.get("degradation", "")))
        Ts.append(Sm.shape[0])
    X = np.stack(rows).astype(np.float32)
    np.savez_compressed(out_npz, X=X, ids=np.array(ok_ids),
                        clips=np.array(clips),
                        y=np.array(ys, dtype=np.int8),
                        reasons=np.array(reasons), names=np.array(names),
                        T=np.array(Ts, dtype=np.int32))
    if series_npz is not None:
        np.savez_compressed(series_npz,
                            ids=np.array(ok_ids),
                            S=np.array(Slist, dtype=object))
    dt = time.time() - t0
    print(f"{manifest_name}: {len(rows)}/{len(items)} ok, "
          f"fails={len(fails)} in {dt:.1f}s", flush=True)
    for fl in fails[:10]:
        print("  FAIL", fl, flush=True)
    return {"manifest": manifest_name, "n_ok": len(rows),
            "n_total": len(items), "seconds": dt, "fails": fails,
            "F": int(X.shape[1])}


def build_dup_sigs():
    from jadal.spectral import dup
    with open(os.path.join(SPEC_DIR, "manifest_gold.json")) as f:
        m = json.load(f)
    sigs = {}
    for it in m["items"]:
        sigs[it["id"]] = dup.signature(it["path"], it["start_s"], it["end_s"])
    with open(os.path.join(OUT, "dup_sigs.json"), "w") as f:
        json.dump(sigs, f)
    print(f"dup_sigs: {len(sigs)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    r1 = build("manifest_gold.json", os.path.join(OUT, "gold.npz"),
               os.path.join(OUT, "series_gold.npz"))
    r2 = build("manifest_synth.json", os.path.join(OUT, "synth.npz"))
    build_dup_sigs()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    meta = {
        "git_sha": sha,
        "command": ("PYTHONNOUSERSITE=1 "
                    "JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models "
                    "/home/parshu/envs/jadal/bin/python eval/perpixel/make_table.py"),
        "resolutions": {
            "armB_T1": "160x160 + ffprobe packet stats (jadal/spectral/features.py, read-only)",
            "gpu_ref": "160x90 stack (same decode as series; fps=T/10s measured)",
            "series": "every decoded frame at 160x90 rgb24 via ffmpeg scale=160:90",
        },
        "feature_groups": {
            "B_*": "40 arm-B T1 scalars (8 families); units per docs/reports/spectral_features.md",
            "G_*": "27 GPU CPU-reference scalars (jadal/gpu/features.py FEATURE_NAMES); "
                   "luma fractions in [0,1], energies raw, shift in px, illum normalised",
            "P_*": "jadal/perpixel/series.py descriptors: band fractions of non-DC energy, "
                   "entropy nats/ln(nbins), ac peak/lag(frames), burstiness=std/|mean|, "
                   "motion floor=median+MAD of diff_e, still_run_frac, shift px@160x90, "
                   "saturation fractions, dct_hf log1p-mean",
        },
        "series_channels": ["luma_mean[0,1]", "lap_var raw", "diff_e mean|d|",
                            "skin_frac[0,1]", "shift_px@160x90 vs frame0",
                            "under_frac luma<0.05", "over_frac luma>0.95",
                            "dct_hf mean|DCT| 8x8-blocked"],
        "gold": r1, "synth": r2,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
