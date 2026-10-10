"""H3 / E1.06: fps x resolution Pareto cross on IndEgo gold (pixel-26).

Grid: fps in {1, 2, 3.33, 5} x res in {480, 360, 240}px, CRF23 libx264 veryfast.
Each gold window is CUT + re-encoded at the cell (fps filter + square scale),
then decoded to 160x90 and featurised with the frozen 26 cpu per-pixel features
(same code path as the cached baseline and X1/RES). LOCO logistic C=0.3
balanced + StandardScaler, clip-bootstrap CI B=1000 seed0 (same as E1.01/RES).

Path remap is runtime-only (bundle manifest still carries laptop /home/parshu
prefixes; labels are never edited).

CPU-only. Run:
  export OMP_NUM_THREADS=2
  python3 eval/exp_fps_res/run_fps_res.py            # all 12 cells
  python3 eval/exp_fps_res/run_fps_res.py f5_r480    # subset by cell names
Merge/eval:
  python3 eval/exp_fps_res/merge_fps_res.py
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from eval.gpu.decode import decode_window  # noqa: E402
from jadal.gpu.features import cpu_window_features  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
JDATA = os.environ.get("JADAL_DATA", os.path.join(BUNDLE, "data"))
GOLD = os.path.join(BUNDLE, "labels", "manifest_gold.json")
WORK = os.environ.get("FPSRES_WORK", "/workspace/scratch/fpsres_work")
OUT = os.path.join(ROOT, "eval", "results", "exp_fps_res")
DW, DH = 160, 90
CRF = 23

FPS_LIST = [5.0, 10.0 / 3.0, 2.0, 1.0]
RES_LIST = [480, 360, 240]


def cell_name(fps, res):
    if abs(fps - 10.0 / 3.0) < 1e-9:
        tag = "3p33"
    elif float(fps).is_integer():
        tag = f"{int(fps)}"
    else:
        tag = str(fps)
    return f"f{tag}_r{res}"


CELLS = [(fps, res) for fps in FPS_LIST for res in RES_LIST]

FF_FPS = {5.0: "fps=5", 10.0 / 3.0: "fps=3.333333", 2.0: "fps=2", 1.0: "fps=1"}


def remap(path):
    if path.startswith("/home/parshu/projects/awing/data/"):
        return os.path.join(JDATA, path[len("/home/parshu/projects/awing/data/"):])
    return path


def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def process_task(args):
    wid, src, start_s, dur_s, fps, res = args
    cell = cell_name(fps, res)
    od = os.path.join(WORK, cell)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"{wid}.mp4")
    enc_ms = 0.0
    if not (os.path.exists(out) and os.path.getsize(out) > 1000):
        vf = f"{FF_FPS[fps]},scale={res}:{res}"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error", "-nostdin",
               "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", src, "-an",
               "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
               "-crf", str(CRF), "-pix_fmt", "yuv420p", "-threads", "1", out]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True)
        enc_ms = (time.perf_counter() - t0) * 1000.0
        if p.returncode != 0:
            return cell, wid, {"error": f"enc rc={p.returncode}: {p.stderr[-200:].decode('utf-8', 'replace')}"}
    try:
        nbytes = os.path.getsize(out)
        t0 = time.perf_counter()
        rgb, fps_meas = decode_window(out, 0.0, dur_s, DW, DH, "rgb24")
        dec_ms = (time.perf_counter() - t0) * 1000.0
        luma = luma_of(rgb)
        t1 = time.perf_counter()
        f = cpu_window_features(luma, rgb, fps=float(fps))
        feat_ms = (time.perf_counter() - t1) * 1000.0
    except Exception as e:
        return cell, wid, {"error": f"dec/feat: {e}"}
    return cell, wid, {"full": f, "bytes": nbytes, "n_frames": int(rgb.shape[0]),
                       "fps_meas": round(float(fps_meas), 3),
                       "enc_ms": round(enc_ms, 1), "dec_ms": round(dec_ms, 1),
                       "feat_ms": round(feat_ms, 1)}


def load_gold():
    M = {i["id"]: i for i in json.load(open(GOLD))["items"]}
    items = [{"id": i["id"], "path": remap(i["path"]), "start_s": i["start_s"],
              "dur_s": round(i["end_s"] - i["start_s"], 2),
              "clip": i["clip"], "label": i["label"]}
             for i in sorted(M.values(), key=lambda x: x["id"])]
    missing = [it["path"] for it in items if not os.path.exists(it["path"])]
    if missing:
        raise RuntimeError(f"{len(missing)} source paths missing, e.g. {missing[:3]}")
    return items


def main():
    only = set(sys.argv[1:]) or None
    jobs = int(os.environ.get("FPSRES_JOBS", "2"))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    items = load_gold()
    cells = [c for c in CELLS if only is None or cell_name(*c) in only]
    print(f"cells={[cell_name(*c) for c in cells]} windows={len(items)} jobs={jobs}", flush=True)
    tasks = [(it["id"], it["path"], it["start_s"], it["dur_s"], fps, res)
             for (fps, res) in cells for it in items]
    t0 = time.perf_counter()
    feats = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for cell, wid, row in ex.map(process_task, tasks):
            feats.setdefault(cell, {})[wid] = row
    wall = round(time.perf_counter() - t0, 1)
    for cell in sorted(feats):
        ok = sum(1 for r in feats[cell].values() if "full" in r)
        err = len(feats[cell]) - ok
        json.dump(feats[cell], open(os.path.join(OUT, f"_feats_{cell}.json"), "w"))
        print(f"{cell}: ok={ok} err={err}", flush=True)
    print(f"wall={wall}s work={WORK}", flush=True)


if __name__ == "__main__":
    main()
