"""E1 cost / Pareto: accuracy vs cost per data regime.

Regimes
  indego_480p_10fps_in_domain : E1.01 fps sweep {10,5,3.33,2,1} + E1.04
                                frame-selection strategies at an equal
                                20-frame budget (both LOCO on 150 human windows)
  ego10k_1080p_30fps_ood      : E1.03 fps sweep {30,15,10,5,3,2,1} on the
                                ORIGINAL 1080p30 sources (frozen IndEgo model,
                                which does NOT transfer -> the accuracy axis is
                                reported but is NOT usable for a choice;
                                `accuracy_meaningful: false`)

Cost: one decode of the full native window (300 frames at 1080p30, 100 frames at
IndEgo 480p10) + the 26 cpu_window_features on the subsampled frames.

Two cost sources are reported:
  * run_ms   : mean over the windows of the actual shard run (the pod was busy;
               the run means live in E1.01/E1.03/E1.04.json)
  * clean_ms : min over --reps repeats of one representative window with
               OMP_NUM_THREADS=1 and cv2 threads=1. Used for the Pareto axis,
               because the shard runs shared the pod with other agents.

Feature cost is also fitted (feat_ms = intercept + slope * n_frames) so the
per-frame figure is not a single noisy division.

-> eval/results/exp_fps/pareto.json
Regen: python3 eval/exp_fps/pareto.py --reps 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import (QUEUE1, RESULTS, decode_window, git_sha, gpu_features,
                     load_manifest, pareto_front, subsample_idx)

OOD_STEPS = {30: 1, 15: 2, 10: 3, 5: 6, 3: 10, 2: 15, 1: 30}
IND_STEPS = {10: 1, 5: 2, 3.33: 3, 2: 5, 1: 10}


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def bench(gf, decode_fn, steps, fps_of, reps):
    """min-of-`reps` decode ms + min-of-`reps` feature ms per fps setting.

    Decode is done once per rep for the whole native window (the same work in
    every fps setting); features are timed on the already-decoded frames.
    """
    best_dec, frames = None, None
    for _ in range(reps):
        rgb, luma, dms = decode_fn()
        best_dec = dms if best_dec is None else min(best_dec, dms)
        frames = (rgb, luma)
    rgb, luma = frames
    out = {}
    for fps, step in steps.items():
        idx = subsample_idx(len(luma), step)
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            gf.cpu_window_features(luma[idx].copy(), rgb[idx].copy(),
                                   fps=float(fps_of(fps, step)))
            ms = (time.perf_counter() - t0) * 1000.0
            best = ms if best is None else min(best, ms)
        out[fps] = {"n_frames": int(len(idx)),
                    "decode_ms": round(best_dec, 1),
                    "decode_ms_per_frame": round(best_dec / max(len(luma), 1), 4),
                    "feat_ms": round(best, 1),
                    "feat_ms_per_frame": round(best / max(len(idx), 1), 4)}
    return out


def fit_feat(clean):
    """feat_ms ~ intercept + slope * n_frames (least squares over fps settings)."""
    n = np.array([v["n_frames"] for v in clean.values()], dtype=float)
    t = np.array([v["feat_ms"] for v in clean.values()], dtype=float)
    if len(n) < 2:
        return {"slope_ms_per_frame": None, "intercept_ms": None}
    slope, intercept = np.polyfit(n, t, 1)
    pred = slope * n + intercept
    ss = 1 - ((t - pred) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-9)
    return {"slope_ms_per_frame": round(float(slope), 4),
            "intercept_ms": round(float(intercept), 2),
            "r2": round(float(ss), 4)}


def row(regime, tag, nfr, auroc, ci95, dec_ms, feat_ms, acc_meaningful, src):
    return {"regime": regime, "tag": tag, "frames_per_window": nfr,
            "auroc": auroc, "ci95": ci95,
            "decode_ms_per_window": round(dec_ms, 1),
            "feat_ms_per_window": round(feat_ms, 1),
            "total_ms_per_window": round(dec_ms + feat_ms, 1),
            "accuracy_meaningful": acc_meaningful, "source": src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--benches", type=int, default=1,
                    help="repeat the whole bench N times; report the spread "
                         "(the pod is shared, so cost varies run to run)")
    ap.add_argument("--no-bench", action="store_true")
    a = ap.parse_args()
    gf = gpu_features()
    load_avg = os.getloadavg()[0]

    if a.no_bench:
        ci, co, spread = {}, {}, None
    else:
        it = load_manifest()[0]
        s = json.load(open(QUEUE1))["segments"][0]
        runs = []
        for _ in range(max(a.benches, 1)):
            runs.append({
                "indego": bench(gf, lambda: decode_window(it["path"], it["start_s"]),
                                IND_STEPS, lambda fps, step: fps, a.reps),
                "ood": bench(gf, lambda: decode_window(s["source_path"],
                                                       s["source_start_s"],
                                                       n_frames=300),
                             OOD_STEPS, lambda fps, step: 30.0 / step, a.reps)})
        ci, co = runs[0]["indego"], runs[0]["ood"]
        for r in runs[1:]:
            for fps in ci:
                ci[fps] = {k: min(ci[fps][k], r["indego"][fps][k]) for k in ci[fps]}
            for fps in co:
                co[fps] = {k: min(co[fps][k], r["ood"][fps][k]) for k in co[fps]}

        def rng(runs_, reg, key):
            v = [r[reg][fps][key] for r in runs_ for fps in r[reg]]
            return [round(min(v), 4), round(max(v), 4)]
        spread = {"n_bench_runs": len(runs), "n_reps_each": a.reps,
                  "indego_decode_ms_per_frame": rng(runs, "indego", "decode_ms_per_frame"),
                  "ood_decode_ms_per_frame": rng(runs, "ood", "decode_ms_per_frame"),
                  "indego_feat_ms_per_frame": rng(runs, "indego", "feat_ms_per_frame"),
                  "ood_feat_ms_per_frame": rng(runs, "ood", "feat_ms_per_frame")}

    rows = []
    e101 = load("E1.01.json") or load("e1_01.json")
    e103 = load("E1.03.json")
    e104 = load("E1.04.json")

    if e101:
        for r in e101["rows"]:
            fps, nfr = r["fps"], r["frames_per_window"]
            c = ci.get(fps) or ci.get(str(fps))
            if c:
                dec_ms, feat_ms = c["decode_ms"], c["feat_ms"]
            else:
                dec_ms = r["decode_ms_mean"]
                feat_ms = r["feat_ms_mean"]
            rows.append(row("indego_480p_10fps_in_domain", f"{fps}fps", nfr,
                            r["auroc"], r["ci95"], dec_ms, feat_ms, True, "E1.01"))
    if e104:
        # every arm computes the features on 20 frames -> same feature cost
        c = ci.get(2) or ci.get("2")
        dec_ms = c["decode_ms"] if c else e104["decode_ms_mean"]
        feat_ms = c["feat_ms"] if c else e104["cost"]["feat_ms_per_window_by_arm"]["uniform20"]
        for r in e104["rows"]:
            rows.append(row("indego_480p_10fps_in_domain", r["strategy"], 20,
                            r["auroc"], r["ci95"], dec_ms, feat_ms, True, "E1.04"))
    if e103:
        for r in e103["frozen_rows"]:
            fps, nfr = r["fps"], r["n_frames_mean"]
            c = co.get(fps) or co.get(str(fps))
            if c:
                dec_ms, feat_ms = c["decode_ms"], c["feat_ms"]
            else:
                dec_ms = e103["decode_ms_mean_300f"]
                feat_ms = r["feat_ms_mean"]
            rows.append(row("ego10k_1080p_30fps_ood", f"{fps}fps", nfr,
                            r["frozen_auroc"], r["ci95_window_boot"],
                            dec_ms, feat_ms, False, "E1.03"))

    fronts = {}
    for regime in sorted(set(r["regime"] for r in rows)):
        sub = [r for r in rows if r["regime"] == regime]
        idx = pareto_front([(r["auroc"], r["total_ms_per_window"]) for r in sub])
        fronts[regime] = [sub[i]["tag"] for i in idx]
        for i, r in enumerate(sub):
            r["pareto_front"] = i in idx

    res = {"experiment": "E1.cost", "n_rows": len(rows),
           "device": "A40 (pod host); decode=CPU cv2 160x90 (cv2 threads=1), "
                     "features=CPU float64 reference, OMP_NUM_THREADS=1",
           "cost_note": "clean_ms = min over %d reps x %d bench runs of one "
                        "representative window (OMP_NUM_THREADS=1, cv2 threads=1); "
                        "the shard-run means in E1.01/E1.03/E1.04.json are 2-4x "
                        "slower because the pod is shared. Host load average at "
                        "bench: %.1f" % (a.reps, max(a.benches, 1), load_avg),
           "bench_spread": spread,
           "feat_fit": {"indego": fit_feat(ci), "ood": fit_feat(co)},
           "git_sha": git_sha(),
           "command": "python3 eval/exp_fps/pareto.py --reps %d" % a.reps,
           "regimes": {
               "indego_480p_10fps_in_domain":
                   "150 IndEgo human-labelled windows, 7 clips, LOCO",
               "ego10k_1080p_30fps_ood":
                   "60 human-labelled Egocentric-10K windows (51 good / 9 bad; "
                   "n_bad=9 exploratory, CI by window is optimistic); frozen "
                   "IndEgo model -> AUROC not usable for a regime choice"},
           "clean_bench_indego": ci, "clean_bench_ood": co,
           "pareto_fronts": fronts, "rows": rows}
    json.dump(res, open(os.path.join(RESULTS, "pareto.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['regime'][:26]:26s} {r['tag']:>10s} nfr={r['frames_per_window']:>6} "
              f"AUROC {r['auroc']:.4f} dec/win {r['decode_ms_per_window']:>7.1f}ms "
              f"feat/win {r['feat_ms_per_window']:>7.1f}ms tot/win "
              f"{r['total_ms_per_window']:>7.1f}ms "
              f"{'FRONT' if r.get('pareto_front') else ''}")
    print("feat fit:", json.dumps(res["feat_fit"]))
    print("fronts:", json.dumps(fronts))


if __name__ == "__main__":
    main()
