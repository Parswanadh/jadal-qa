"""E1 cost / Pareto: accuracy vs cost per data regime.

Regimes
  indego_480p_10fps_in_domain : E1.01 fps sweep {10,5,3.33,2,1} + E1.04
                                frame-selection strategies at an equal
                                20-frame budget (both LOCO on 150 human windows)
  ego10k_1080p_30fps_ood      : E1.03 fps sweep {30,15,10,5,3,2,1} on the
                                ORIGINAL 1080p30 sources (frozen IndEgo model,
                                which does NOT transfer -> accuracy axis is
                                reported but not usable for a choice)

Cost axes: decode ms/frame and feature ms/frame. Two cost sources are reported:
  * run_ms      : mean over the windows of the actual shard run (pod was busy;
                  host load average is recorded in the JSON)
  * clean_ms    : min over --reps repeats of one representative window, run
                  with OMP_NUM_THREADS=1 and cv2 threads=1 (least-contended
                  estimate). This is the number used for the Pareto axis.

-> eval/results/exp_fps/pareto.json
Regen: python3 eval/exp_fps/pareto.py [--reps 3]
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


def bench_indego(gf, reps):
    """(fps -> (decode_ms_per_frame, feat_ms_per_frame)) on one IndEgo window."""
    it = load_manifest()[0]
    out = {}
    for fps, step in IND_STEPS.items():
        best_dec, best_feat, n = None, None, 0
        for _ in range(reps):
            rgb, luma, dms = decode_window(it["path"], it["start_s"])
            idx = subsample_idx(len(luma), step)
            t0 = time.perf_counter()
            gf.cpu_window_features(luma[idx].copy(), rgb[idx].copy(),
                                   fps=float(fps))
            fms = (time.perf_counter() - t0) * 1000.0
            n = len(idx)
            best_dec = dms if best_dec is None else min(best_dec, dms)
            best_feat = fms if best_feat is None else min(best_feat, fms)
        out[fps] = {"n_frames": n,
                    "decode_ms_per_frame": round(best_dec / max(n, 1), 4),
                    "feat_ms_per_frame": round(best_feat / max(n, 1), 4)}
    return out


def bench_ood(gf, reps):
    """(fps -> (decode_ms_per_frame, feat_ms_per_frame)) on one 1080p OOD window."""
    s = json.load(open(QUEUE1))["segments"][0]
    out = {}
    for fps, step in OOD_STEPS.items():
        best_dec, best_feat, n = None, None, 0
        for _ in range(reps):
            rgb, luma, dms = decode_window(s["source_path"], s["source_start_s"],
                                           n_frames=300)
            idx = subsample_idx(len(luma), step)
            t0 = time.perf_counter()
            gf.cpu_window_features(luma[idx].copy(), rgb[idx].copy(),
                                   fps=float(30.0 / step))
            fms = (time.perf_counter() - t0) * 1000.0
            n = len(idx)
            best_dec = dms if best_dec is None else min(best_dec, dms)
            best_feat = fms if best_feat is None else min(best_feat, fms)
        out[fps] = {"n_frames": n,
                    "decode_ms_per_frame": round(best_dec / max(n, 1), 4),
                    "feat_ms_per_frame": round(best_feat / max(n, 1), 4)}
    return out


def row(regime, tag, nfr, auroc, ci95, dec_mpf, feat_mpf, acc_meaningful, src):
    dec = dec_mpf * nfr
    feat = feat_mpf * nfr
    return {"regime": regime, "tag": tag, "frames_per_window": nfr,
            "auroc": auroc, "ci95": ci95,
            "decode_ms_per_frame": round(dec_mpf, 4),
            "feat_ms_per_frame": round(feat_mpf, 4),
            "decode_ms_per_window": round(dec, 1),
            "feat_ms_per_window": round(feat, 1),
            "total_ms_per_window": round(dec + feat, 1),
            "accuracy_meaningful": acc_meaningful, "source": src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-bench", action="store_true")
    a = ap.parse_args()
    gf = gpu_features()
    load_avg = os.getloadavg()[0]
    clean_i = {} if a.no_bench else bench_indego(gf, a.reps)
    clean_o = {} if a.no_bench else bench_ood(gf, a.reps)

    rows = []
    e101, e103, e104 = load("E1.01.json"), load("E1.03.json"), load("E1.04.json")

    if e101:
        for r in e101["rows"]:
            fps = r["fps"]
            nfr = r["frames_per_window"]
            c = clean_i.get(fps) or clean_i.get(str(fps))
            if c:
                dec_mpf, feat_mpf = c["decode_ms_per_frame"], c["feat_ms_per_frame"]
            else:
                dec_mpf = r["decode_ms_mean"] / max(r["n_scored"], 1)
                feat_mpf = r["feat_ms_mean"] / max(r["n_scored"], 1)
            rows.append(row("indego_480p_10fps_in_domain", f"{fps}fps", nfr,
                            r["auroc"], r["ci95"], dec_mpf, feat_mpf, True,
                            "E1.01"))
    if e104:
        dec_mpf = (e104["decode_ms_mean"] / 100.0)  # 100 native frames decoded
        for r in e104["rows"]:
            c = clean_i.get(2)
            fmpf = c["feat_ms_per_frame"] if c else r["feat_ms_mean"] / max(r["n_frames_mean"], 1)
            rows.append(row("indego_480p_10fps_in_domain", r["strategy"], 20,
                            r["auroc"], r["ci95"], dec_mpf, fmpf, True, "E1.04"))
    if e103:
        for r in e103["frozen_rows"]:
            fps = r["fps"]
            nfr = r["n_frames_mean"]
            c = clean_o.get(fps) or clean_o.get(str(fps))
            if c:
                dec_mpf, feat_mpf = c["decode_ms_per_frame"], c["feat_ms_per_frame"]
            else:
                dec_mpf = e103["decode_ms_per_frame"]
                feat_mpf = r["feat_ms_per_frame"]
            rows.append(row("ego10k_1080p_30fps_ood", f"{fps}fps", nfr,
                            r["frozen_auroc"], r["ci95_window_boot"],
                            dec_mpf, feat_mpf, False, "E1.03"))

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
           "cost_note": "clean_ms = min over %d repeats of one representative "
                        "window; run means are in E1.01/E1.03/E1.04.json. "
                        "Host load average at bench: %.1f" % (a.reps, load_avg),
           "git_sha": git_sha(),
           "command": "python3 eval/exp_fps/pareto.py --reps 3",
           "regimes": {"indego_480p_10fps_in_domain":
                       "150 IndEgo human-labelled windows, 7 clips, LOCO",
                       "ego10k_1080p_30fps_ood":
                       "60 human-labelled Egocentric-10K windows (51 good/9 bad, "
                       "n_bad=9 exploratory, CI by window is optimistic); frozen "
                       "IndEgo model -> AUROC not usable for a regime choice"},
           "clean_bench_indego": clean_i, "clean_bench_ood": clean_o,
           "pareto_fronts": fronts, "rows": rows}
    json.dump(res, open(os.path.join(RESULTS, "pareto.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['regime'][:26]:26s} {r['tag']:>10s} nfr={r['frames_per_window']:>5} "
              f"AUROC {r['auroc']:.4f} dec/f {r['decode_ms_per_frame']:.3f}ms "
              f"feat/f {r['feat_ms_per_frame']:.3f}ms tot/win {r['total_ms_per_window']:.0f}ms "
              f"{'FRONT' if r.get('pareto_front') else ''}")
    print("fronts:", json.dumps(fronts))


if __name__ == "__main__":
    main()
