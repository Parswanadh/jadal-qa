"""X1 / RES: resolution + compression sweep on the 150 human-labelled IndEgo gold windows.

Re-encodes each gold window with libx264 at a grid of (resolution, CRF), re-extracts
the 26 per-pixel features with the SAME frozen code path as the cached baseline
(decode -> 160x90, cpu_window_features), and evaluates LOCO AUROC with the identical
split + protocol as `tools/port/baseline_logistic_loco.py` (C=0.3, StandardScaler,
class_weight=balanced, leave-one-clip-out; CI = clip-bootstrap B=1000 seed 0).

CPU-only (numpy + sklearn + ffmpeg). GPU not required.
Outputs: eval/results/exp_res/<name>.json
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
from eval.gpu.decode import decode_window            # noqa: E402
from jadal.gpu.features import cpu_window_features   # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
GOLD = os.path.join(BUNDLE, "labels", "manifest_gold.json")
CACHED = os.path.join(BUNDLE, "caches", "gpu", "features.json")
WORK = os.environ.get("RES_WORK", "/workspace/scratch/res_work")
OUT = os.path.join(ROOT, "eval", "results", "exp_res")
DW, DH = 160, 90

# name -> (resolution px, crf).  resolution is the square encode edge (source is 480x480).
CONFIGS = [
    ("r480_crf23", 480, 23),
    ("r360_crf23", 360, 23),
    ("r240_crf23", 240, 23),
    ("r180_crf23", 180, 23),
    ("r480_crf18", 480, 18),
    ("r480_crf28", 480, 28),
    ("r480_crf33", 480, 33),
    ("r360_crf28", 360, 28),
    ("r240_crf28", 240, 28),
    ("r120_crf23", 120, 23),   # below the 160x90 decode target: find the knee
]


def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def process_window(args):
    """Encode (+reuse existing) then decode + extract 26 features. One pool hop."""
    wid, path, start_s, dur_s, res, crf = args
    od = os.path.join(WORK, f"{res}_{crf}")
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"{wid}.mp4")
    enc_ms = 0.0
    if not (os.path.exists(out) and os.path.getsize(out) > 1000):
        cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error", "-nostdin",
               "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", path, "-an",
               "-vf", f"scale={res}:{res}", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", "2", out]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True)
        enc_ms = (time.perf_counter() - t0) * 1000.0
        if p.returncode != 0:
            return wid, {"error": f"enc rc={p.returncode}: {p.stderr[-200:].decode('utf-8','replace')}"}
    else:
        _, enc_ms = 0.0, 0.0
    nbytes = os.path.getsize(out)
    try:
        t0 = time.perf_counter()
        rgb, fps = decode_window(out, 0.0, dur_s, DW, DH, "rgb24")
        dec_ms = (time.perf_counter() - t0) * 1000.0
        luma = luma_of(rgb)
        t1 = time.perf_counter()
        f = cpu_window_features(luma, rgb, fps=min(fps, 30.0))
        tf_ms = (time.perf_counter() - t1) * 1000.0
    except Exception as e:
        return wid, {"error": f"dec/feat: {e}"}
    return wid, {"full": f, "bytes": nbytes, "enc_ms": round(enc_ms, 1),
                 "dec_ms": round(dec_ms, 1), "feat_ms": round(tf_ms, 1)}


def loco_proba(X, y, clips, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for c in sorted(set(clips)):
        tr, te = clips != c, clips == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def clip_bootstrap_ci(y, p, clips, B=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    names = sorted(set(clips))
    out = []
    for _ in range(B):
        cs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(clips == c)[0] for c in cs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def load_gold():
    M = {i["id"]: i for i in json.load(open(GOLD))["items"]}
    return M


def run_config(name, res, crf, items, jobs=6):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        got = dict(ex.map(process_window, [(it["id"], it["path"], it["start_s"], it["dur_s"], res, crf) for it in items]))
    rows = {}
    for it in items:
        wid = it["id"]
        r = got.get(wid, {})
        if "full" in r:
            rows[wid] = r
    wall = round(time.perf_counter() - t0, 1)
    return rows, wall


def main():
    only = sys.argv[1:] or None
    jobs = int(os.environ.get("RES_JOBS", "6"))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    M = load_gold()
    items = [{"id": i["id"], "path": i["path"], "start_s": i["start_s"],
              "dur_s": round(i["end_s"] - i["start_s"], 2), "clip": i["clip"], "label": i["label"]}
             for i in sorted(M.values(), key=lambda x: x["id"])]
    cfgs = [c for c in CONFIGS if only is None or c[0] in only]
    for name, res, crf in cfgs:
        spec = os.path.join(OUT, f"{name}.json")
        if os.path.exists(spec):
            print("skip existing", name)
            continue
        rows, wall = run_config(name, res, crf, items, jobs)
        ids = [it["id"] for it in items if it["id"] in rows]
        names = sorted(rows[ids[0]]["full"].keys())
        X = np.nan_to_num(np.array([[rows[k]["full"][n] for n in names] for k in ids], float))
        y = np.array([M[k]["label"] for k in ids])
        clips = np.array([M[k]["clip"] for k in ids])
        p = loco_proba(X, y, clips, C=0.3)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p))
        lo, hi = clip_bootstrap_ci(y, p, clips)
        res_out = {
            "experiment": "RES.01..03", "combo": name, "resolution_px": res, "crf": crf,
            "n": len(ids), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
            "label_source": "human (IndEgo gold, 100 good/50 bad)",
            "protocol": "re-encode each gold window (libx264 veryfast) -> decode 160x90 -> 26 cpu per-pixel features -> LOCO logistic C=0.3 class_weight=balanced + StandardScaler; CI=clip-bootstrap B=1000 seed0",
            "split": "leave-one-clip-out over the 7 gold clips (identical to cached baseline)",
            "auroc": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "bytes_per_window": int(np.mean([rows[k]["bytes"] for k in ids])),
            "enc_ms_per_window": round(float(np.mean([rows[k]["enc_ms"] for k in ids])), 1),
            "dec_ms_per_window": round(float(np.mean([rows[k]["dec_ms"] for k in ids])), 1),
            "feat_ms_per_window": round(float(np.mean([rows[k]["feat_ms"] for k in ids])), 1),
            "total_ms_per_window": round(float(np.mean([rows[k]["enc_ms"] + rows[k]["dec_ms"] + rows[k]["feat_ms"] for k in ids])), 1),
            "wall_s": wall, "command": "eval/exp_res/run_res.py " + name,
        }
        json.dump(res_out, open(spec, "w"), indent=1)
        json.dump(rows, open(os.path.join(OUT, f"_feats_{name}.json"), "w"))
        print(f"{name:12s} r={res:4d} crf={crf:2d} AUROC={auc:.4f} [{lo:.3f},{hi:.3f}] "
              f"bytes={res_out['bytes_per_window']:>7d} tot_ms={res_out['total_ms_per_window']:.0f} wall={wall}s")


if __name__ == "__main__":
    main()
