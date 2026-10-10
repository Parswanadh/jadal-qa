#!/usr/bin/env python3
"""L3.01 analysis: full-corpus optimized pipeline vs the frozen L2.01 baseline.

Reads eval/results/exp_latency_fast/{windows.jsonl,run_meta.json,clips.jsonl}
(+ windows_stride2.jsonl / run_meta_stride2.json for the every-2nd policy run)
and the E8.01 reference, writes L3.01.json + docs/reports/exp_latency_fast.md
+ 2 figures.

Regen: python3 eval/exp_latency_fast/analyze_fast.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
IN = os.environ.get("L3_IN", os.path.join(REPO, "eval", "results", "exp_latency_fast"))
IMG = os.path.join(REPO, "docs", "experiments", "exp_latency_fast", "images")
os.makedirs(IMG, exist_ok=True)

REF_E8 = "/workspace/wt/exp-rescore/eval/results/exp_rescore/E8.01.json"
REF_LINES = "/workspace/artifacts/windows_rescored.jsonl"
BASELINE_PARTIAL = os.environ.get(
    "L3_BASELINE_PARTIAL",
    "/workspace/wt/m-latency-full/eval/results/exp_latency_full/windows.jsonl.partial")

STAGES = ["io_probe", "decode", "pixel_features", "descriptors", "eg2_embed",
          "mosaic", "decision", "write"]
PRETTY = {"io_probe": "io_probe", "decode": "decode(nvdec3)",
          "pixel_features": "pixel-26", "descriptors": "descriptors",
          "eg2_embed": "eg2_embed", "mosaic": "mosaic",
          "decision": "decision", "write": "write"}

# Frozen L2.01 baseline (the comparison the task fixes; measured with
# --workers 4, OMP_NUM_THREADS=1, 300 pixel-frames/window, 2 fps EG2, 10 s
# window, mosaic on). ~511 windows in the partial; per-window p50 shown.
BASELINE = {
    "id": "L2.01", "branch": "m/latency-full", "workers": 4, "omp_threads": "1",
    "pixel_frames_per_window": 300, "eg2_fps": 2.0, "window_s": 10.0,
    "mosaic": True,
    "windows_per_s": 0.84,
    "per_window_p50_ms": {"total": 4437.0, "decode": 1679.0, "eg2_embed": 991.0,
                          "pixel_features": 817.0, "descriptors": 709.0,
                          "mosaic": 204.0},
    "verdict_distribution": {"accept": 296, "reject": 83, "review": 1167},
    "profile_distribution": {"default": 6, "factory_1080p30": 1540},
}


def pct(a):
    a = np.asarray(a, float)
    if not len(a):
        return {}
    return {"n": int(len(a)), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
            "mean": round(float(a.mean()), 2), "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2)}


def gpu_util_for_interval(smi, t0, t1):
    u = [s["gpu_pct"] for s in smi if t0 <= s["t"] <= t1]
    return float(np.mean(u)) if u else None


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    rows = load_rows(os.path.join(IN, "windows.jsonl"))
    meta = json.load(open(os.path.join(IN, "run_meta.json")))
    clips = load_rows(os.path.join(IN, "clips.jsonl"))
    n = len(rows)
    opt = meta.get("optimizations", {})
    print(f"windows={n} clips={len(clips)} opt={opt}")

    # ---- reference verdicts (E8.01 == frozen baseline verdicts) ----
    ref_dist = {}
    ref_per_window = {}
    if os.path.exists(REF_E8):
        ref_dist = json.load(open(REF_E8)).get("verdict_distribution", {})
    if os.path.exists(REF_LINES):
        for ln in open(REF_LINES):
            try:
                d = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if d.get("type") == "header":
                ref_dist = d.get("verdict_distribution", ref_dist)
                continue
            ref_per_window[d["window_id"]] = d
    if not ref_dist:
        ref_dist = BASELINE["verdict_distribution"]

    from collections import Counter
    vd = Counter(r["verdict"] for r in rows)
    pd = Counter(r["profile"] for r in rows)
    share = {k: round(100.0 * v / max(n, 1), 2) for k, v in sorted(vd.items())}
    ref_share = {k: round(100.0 * v / max(sum(ref_dist.values()), 1), 2)
                 for k, v in ref_dist.items()}
    compared, mism = 0, []
    for r in rows:
        ref = ref_per_window.get(r["window_id"])
        if ref is None:
            continue
        compared += 1
        if r["verdict"] != ref["verdict"]:
            mism.append({"window_id": r["window_id"], "ours": r["verdict"],
                         "ref": ref["verdict"]})

    # ---- per-stage latency ----
    stages = {PRETTY[s]: pct([r["latency_ms"].get(s, 0.0) for r in rows])
              for s in STAGES}
    means = {s: float(np.mean([r["latency_ms"].get(s, 0.0) for r in rows]))
             for s in STAGES}
    pipe_total = np.array([sum(r["latency_ms"].get(s, 0.0) for s in STAGES)
                           for r in rows])
    total_sum = float(pipe_total.sum())
    shares = {PRETTY[s]: round(100.0 * means[s] * n / total_sum, 2)
              for s in STAGES}
    bottleneck = max(STAGES, key=lambda s: means[s])

    # ---- time ----
    wall = meta["wall_s"]
    hours = meta["manifest"]["total_hours"]
    time_block = {
        "wall_s": wall, "wall_min": round(wall / 60.0, 2),
        "n_windows": n, "windows_per_s": round(n / wall, 4),
        "windows_per_min": round(60.0 * n / wall, 1),
        "corpus_hours": hours,
        "footage_h_per_wall_h": round(hours / (wall / 3600.0), 3),
        "mean_ms_per_window": round(float(pipe_total.mean()), 1),
        "p50_ms_per_window": round(float(np.percentile(pipe_total, 50)), 1),
        "p95_ms_per_window": round(float(np.percentile(pipe_total, 95)), 1),
        "workers": meta["workers"], "omp_threads": meta["omp_threads"],
        "window_stride": meta.get("window_stride", 1),
        "n_manifest_windows": meta.get("n_manifest_windows"),
        "max_rss_gb": round(meta.get("max_rss_gb", 0.0), 2),
    }

    # ---- speedup vs frozen baseline ----
    bp = BASELINE["per_window_p50_ms"]
    fst = stages
    stage_speedup = {}
    for s in ("decode", "pixel_features", "descriptors", "eg2_embed", "mosaic"):
        key = PRETTY[s]
        if s in bp and fst.get(key, {}).get("p50"):
            stage_speedup[key] = round(bp[s] / fst[key]["p50"], 3)
    stage_speedup["total"] = round(
        bp["total"] / float(np.percentile(pipe_total, 50)), 3)
    base_wps = BASELINE["windows_per_s"]
    fast_wps = n / wall
    base_wall_s = BASELINE["verdict_distribution"] and \
        sum(BASELINE["verdict_distribution"].values()) / base_wps
    base_footage = hours / (base_wall_s / 3600.0)
    speedup = {
        "baseline": {"workers": BASELINE["workers"], "windows_per_s": base_wps,
                     "wall_s_implied": round(base_wall_s, 1),
                     "footage_h_per_wall_h": round(base_footage, 3),
                     "per_window_p50_ms": bp},
        "optimized": {"workers": meta["workers"], "windows_per_s": round(fast_wps, 4),
                      "wall_s": wall,
                      "footage_h_per_wall_h": time_block["footage_h_per_wall_h"],
                      "per_window_p50_ms": {PRETTY[s]: fst[PRETTY[s]].get("p50")
                                            for s in STAGES}},
        "per_stage_p50_speedup": stage_speedup,
        "overall_throughput_speedup": round(fast_wps / base_wps, 3),
        "overall_footage_speedup": round(
            time_block["footage_h_per_wall_h"] / base_footage, 3),
        "per_worker_normalized_speedup": round(
            (fast_wps / meta["workers"]) / (base_wps / BASELINE["workers"]), 3),
        "note": ("per-stage p50 speedup isolates the adopted optimizations; the "
                 "overall throughput speedup also reflects 4 -> %d workers "
                 "(cgroup quota 7.65 cores)." % meta["workers"]),
    }

    # ---- adopted every-2nd policy run (stride 2), if present ----
    policy = None
    s2_meta_path = os.path.join(IN, "run_meta_stride2.json")
    s2_rows_path = os.path.join(IN, "windows_stride2.jsonl")
    if os.path.exists(s2_meta_path) and os.path.exists(s2_rows_path):
        m2 = json.load(open(s2_meta_path))
        r2 = load_rows(s2_rows_path)
        tot2 = np.array([sum(x["latency_ms"].get(s, 0.0) for s in STAGES)
                         for x in r2])
        policy = {
            "what": "adopted E3.01 every-2nd-window grid, measured (not derived)",
            "window_stride": m2.get("window_stride"),
            "n_windows": len(r2),
            "wall_s": m2["wall_s"], "windows_per_s": m2["windows_per_s"],
            "footage_h_per_wall_h": m2["footage_h_per_wall_h"],
            "p50_ms_per_window": round(float(np.percentile(tot2, 50)), 1),
            "speedup_vs_baseline_throughput": round(
                m2["windows_per_s"] / base_wps, 3),
            "speedup_vs_baseline_footage": round(
                m2["footage_h_per_wall_h"] / base_footage, 3),
            "verdict_distribution": m2.get("verdict_distribution"),
        }

    # derived every-2nd policy from the stride-1 run (each processed window
    # represents 2 windows of footage; wall halves at the same windows/s)
    derived_every2 = {
        "what": "E3.01 every-2nd-window grid, derived from the stride-1 run",
        "windows_per_s": round(fast_wps, 4),
        "wall_s_for_773_windows": round(wall / 2.0, 1),
        "footage_h_per_wall_h": round(2.0 * time_block["footage_h_per_wall_h"], 3),
        "speedup_vs_baseline_footage": round(
            2.0 * time_block["footage_h_per_wall_h"] / base_footage, 3),
    }

    # ---- GPU util + VRAM ----
    smi = meta.get("smi", [])
    util_all = [s["gpu_pct"] for s in smi]
    vram_all = [s["vram_mib"] for s in smi]
    per_stage_gpu = {}
    for s in STAGES:
        vals = []
        for r in rows:
            w = r.get("stage_wall", {}).get(s)
            if not w:
                continue
            u = gpu_util_for_interval(smi, w[0], w[1])
            if u is not None:
                vals.append(u)
        per_stage_gpu[PRETTY[s]] = pct(vals) if vals else {}
    eg2_vram = [r["gpu"]["eg2_vram_peak_mib"] for r in rows
                if r.get("gpu", {}).get("eg2_vram_peak_mib")]
    gpu_block = {
        "device": "A40 (46068 MiB)",
        "smi_samples": len(smi),
        "run_util_pct": pct(util_all) if util_all else {},
        "run_vram_mib": pct(vram_all) if vram_all else {},
        "vram_peak_mib": round(max(vram_all), 1) if vram_all else None,
        "gpu_util_pct_per_stage": per_stage_gpu,
        "eg2_vram_peak_mib_per_window": pct(eg2_vram) if eg2_vram else {},
        "cores": {"cgroup_quota_cores": 7.65, "nproc": 96,
                  "workers": meta["workers"], "omp_threads": meta["omp_threads"],
                  "note": "OMP_NUM_THREADS=1 per worker; multiprocessing across "
                          "windows is the intended parallelism"},
    }

    # ---- frames actually used (proves the fps cuts) ----
    frames = {
        "pixel": pct([r["n_frames"]["pixel"] for r in rows]),
        "eg2": pct([r["n_frames"]["eg2"] for r in rows]),
        "native": pct([r["n_frames"]["native"] for r in rows]),
    }

    # ---- per-clip table ----
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)
    clip_rows = []
    for c in clips:
        rs = by_clip.get(c["clip"], [])
        if not rs:
            continue
        tot = np.array([sum(r["latency_ms"].get(s, 0.0) for s in STAGES)
                        for r in rs])
        clip_rows.append({
            "clip": c["clip"], "factory": c["factory"],
            "n_windows": len(rs), "duration_s": c.get("duration_s"),
            "ffprobe_cold_ms": c.get("ffprobe_cold_ms"),
            "mean_total_ms": round(float(tot.mean()), 1),
            "p95_total_ms": round(float(np.percentile(tot, 95)), 1),
            "mean_decode_ms": round(float(np.mean(
                [r["latency_ms"]["decode"] for r in rs])), 1),
            "mean_eg2_ms": round(float(np.mean(
                [r["latency_ms"]["eg2_embed"] for r in rs])), 1),
            "mean_pixel_ms": round(float(np.mean(
                [r["latency_ms"]["pixel_features"] for r in rs])), 1),
            "serial_clip_s": round(float(tot.sum() / 1000.0), 1),
            "footage_min": round(len(rs) * 10.0 / 60.0, 2),
        })

    # ---- clean (uncontended, 1 worker) probe ----
    clean = None
    cpath = os.path.join(IN, "probe_clean.json")
    if os.path.exists(cpath):
        cp = json.load(open(cpath))
        cp["per_stage_ms"] = {PRETTY[k]: v for k, v in cp["per_stage_ms"].items()}
        clean = {
            "what": cp["what"], "n": cp["n"], "workers": cp["workers"],
            "per_stage_ms": cp["per_stage_ms"], "total_ms": cp["total_ms"],
            "frames_per_window": cp["frames_per_window"],
            "decode_4s_vs_10s_1worker": {
                k: v for k, v in cp["decode_4s_vs_10s_1worker"].items()
                if k != "rows"},
            "per_stage_p50_speedup_vs_baseline": {},
        }
        for s in ("decode", "pixel_features", "descriptors", "eg2_embed", "mosaic",
                  "total"):
            bp50 = BASELINE["per_window_p50_ms"].get(s)
            if s == "total":
                key, fp50 = "total", cp["total_ms"]["p50"]
            else:
                key = PRETTY[s]
                fp50 = cp["per_stage_ms"].get(key, {}).get("p50")
            if bp50 and fp50:
                clean["per_stage_p50_speedup_vs_baseline"][key] = round(
                    bp50 / fp50, 3)

    # ---- score shift under sparse sampling (frozen heads) ----
    score_shift = None
    fsc = np.array([r["score"] for r in rows])
    rsc = np.array([ref_per_window[r["window_id"]]["score"] for r in rows
                    if r["window_id"] in ref_per_window])
    if len(rsc) == len(fsc):
        score_shift = {
            "full_run_score_mean": round(float(fsc.mean()), 4),
            "reference_score_mean": round(float(rsc.mean()), 4),
            "delta_mean": round(float(fsc.mean() - rsc.mean()), 4),
            "delta_p50": round(float(np.percentile(fsc - rsc, 50)), 4),
            "note": ("the frozen ensemble_v1 heads were fit on full-sampling "
                     "features; feeding sparser views shifts the standardized "
                     "features and therefore the score. See probe_variants for "
                     "the per-optimization attribution."),
        }
    vpath = os.path.join(IN, "probe_variants.json")
    if os.path.exists(vpath):
        score_shift = score_shift or {}
        score_shift["variants"] = json.load(open(vpath))["variants"]

    # ---- optional live baseline cross-check ----
    live_base = None
    if os.path.exists(BASELINE_PARTIAL):
        try:
            brows = [json.loads(l)["row"] for l in open(BASELINE_PARTIAL)
                     if l.strip() and json.loads(l).get("row")]
        except Exception:  # noqa: BLE001
            brows = []
        if len(brows) >= 100:
            btot = [sum(r["latency_ms"].get(s, 0.0) for s in STAGES) for r in brows]
            live_base = {
                "path": BASELINE_PARTIAL, "n": len(brows),
                "p50_ms_per_window": round(float(np.percentile(btot, 50)), 1),
                "p95_ms_per_window": round(float(np.percentile(btot, 95)), 1),
                "per_stage_p50_ms": {PRETTY[s]: round(float(np.percentile(
                    [r["latency_ms"].get(s, 0.0) for r in brows], 50)), 1)
                    for s in STAGES},
                "note": "live re-run of the baseline on the shared pod (peer job); "
                        "cross-check only, not the frozen comparison",
            }

    res = {
        "id": "L3.01", "track": "latency", "device": "A40",
        "title": "Full-corpus pipeline at full resources + adopted optimizations",
        "label_source": "none (unlabelled windows; verdicts DESCRIPTIVE)",
        "recipe": "ensemble_v1 (EG2+pixel equal-weight) + profiles/router (E4.04)",
        "protocol": ("same harness as L2.01 (repo E8.01 pipeline, perf_counter per "
                     "stage); scoring code unchanged; adopted input optimizations "
                     "E1.01/E1.02/E3.01/E3.03/E2 applied as flags"),
        "optimizations_applied": {
            "E1.01_pixel_fps": opt.get("pixel_fps"),
            "E1.02_eg2_fps": opt.get("eg2_fps"),
            "E3.03_early_exit_s": opt.get("early_exit_s"),
            "E3.01_window_stride": meta.get("window_stride"),
            "E2_skip_mosaic": opt.get("skip_mosaic"),
            "X1_360p_crf28": "corpus ships 1080p HEVC only; X1 is a bytes win, "
                             "not a decode-compute win on local disk",
            "resources": {"workers": meta["workers"], "omp_threads": meta["omp_threads"],
                          "gpu_slots": os.environ.get("JADAL_GPU_SLOTS", "6")},
        },
        "command": ("source /workspace/env.sh && export OMP_NUM_THREADS=1 "
                    "JADAL_GPU_SLOTS=6 && bash tools/pod/gpu_run.sh python3 "
                    "eval/exp_latency_fast/run_fast.py --workers 7 --pixel-fps 3.33 "
                    "--eg2-fps 1 --early-exit-s 4 --skip-mosaic --no-compare "
                    "--window-stride 1 --resume"),
        "git_sha": meta.get("git_sha"), "run_utc": meta.get("run_utc"),
        "contention_caveat": ("peer opencode jobs shared the pod during the run "
                              "(a concurrent baseline re-run held ~4 cores on the "
                              "same 7.65-core cgroup); the frozen baseline p50s "
                              "were themselves measured under pod load ~10-13/96."),
        "outcomes": {
            "windows_processed": n, "windows_requested": meta.get("n_requested"),
            "failures": meta.get("n_errors", 0),
            "error_examples": meta.get("errors", [])[:5],
            "verdict_distribution": dict(sorted(vd.items())),
            "verdict_share_pct": share,
            "profile_distribution": dict(sorted(pd.items())),
            "reference_E8_01_verdict_distribution": ref_dist,
            "reference_E8_01_verdict_share_pct": ref_share,
            "verdict_mismatch_count": len(mism),
            "verdict_mismatch_examples": mism[:10],
            "per_window_compared": compared,
            "verdict_agreement_pct": (round(100.0 * (compared - len(mism)) /
                                            compared, 2) if compared else None),
            "note": ("verdicts are DESCRIPTIVE (no labels). The optimized run uses "
                     "sparser sampling, so verdicts need not be bit-identical; the "
                     "distribution should be close."),
        },
        "time": time_block,
        "per_stage_ms": stages,
        "stage_mean_ms": {PRETTY[s]: round(means[s], 2) for s in STAGES},
        "stage_share_pct": shares,
        "bottleneck": PRETTY[bottleneck],
        "bottleneck_mean_ms": round(means[bottleneck], 2),
        "frames_per_window": frames,
        "speedup_vs_baseline": speedup,
        "clean_probe_1worker": clean,
        "score_shift_under_sparse_sampling": score_shift,
        "adopted_policy_every2": policy,
        "adopted_policy_every2_derived": derived_every2,
        "gpu": gpu_block,
        "live_baseline_crosscheck": live_base,
        "per_clip_table": clip_rows,
        "n_clips": len(clip_rows),
        "honesty_note": json.load(open(REF_E8)).get("honesty_note")
        if os.path.exists(REF_E8) else None,
    }
    json.dump(res, open(os.path.join(IN, "L3.01.json"), "w"), indent=1,
              sort_keys=True)
    print(json.dumps({"outcomes": {k: v for k, v in res["outcomes"].items()
                                   if "examples" not in k},
                      "time": res["time"], "stage_share_pct": shares,
                      "speedup": speedup, "policy": policy,
                      "bottleneck": res["bottleneck"]}, indent=1))

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [PRETTY[s] for s in STAGES]
    p50 = [stages[l]["p50"] for l in labels]
    p95 = [stages[l]["p95"] for l in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.2, p50, 0.4, label="optimized p50", color="#2b8cbe")
    ax.bar(x + 0.2, p95, 0.4, label="optimized p95", color="#a6bddb")
    base_keys = {"decode(nvdec3)": "decode", "pixel-26": "pixel_features",
                 "descriptors": "descriptors", "eg2_embed": "eg2_embed",
                 "mosaic": "mosaic"}
    bx, by = [], []
    for i, l in enumerate(labels):
        bk = base_keys.get(l)
        if bk and bk in BASELINE["per_window_p50_ms"]:
            bx.append(i)
            by.append(BASELINE["per_window_p50_ms"][bk])
    ax.scatter(bx, by, marker="D", color="#e34a33", zorder=5,
               label="frozen baseline p50")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ms / window")
    ax.set_title(f"L3.01 per-stage latency, optimized pipeline (n={n}, A40, "
                 f"{meta['workers']} workers) vs frozen L2.01 baseline")
    ax.legend()
    for xi, v in zip(x - 0.2, p50):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + 0.2, p95):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "per_stage_p50_p95.png"), dpi=120)

    # figure 2: per-stage speedup + throughput
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    keys = ["decode(nvdec3)", "pixel-26", "descriptors", "eg2_embed", "mosaic",
            "total"]
    sp = [stage_speedup.get(k, 0.0) for k in keys]
    ax1.bar(range(len(keys)), sp, color="#41ab5d")
    ax1.axhline(1.0, color="k", lw=0.8, ls="--")
    ax1.set_xticks(range(len(keys)))
    ax1.set_xticklabels(keys, rotation=20, ha="right")
    ax1.set_ylabel("speedup (baseline p50 / optimized p50)")
    ax1.set_title("Per-stage speedup vs frozen baseline")
    for i, v in enumerate(sp):
        ax1.text(i, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    bars = [base_wps, fast_wps]
    ax2.bar([0, 1], bars, color=["#e34a33", "#2b8cbe"], width=0.55)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"baseline\n(4 workers)", f"optimized\n({meta['workers']} workers)"])
    ax2.set_ylabel("windows / s")
    ax2.set_title(f"Throughput {base_wps:.2f} -> {fast_wps:.2f} win/s "
                  f"({fast_wps/base_wps:.2f}x)")
    for i, v in enumerate(bars):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "speedup_and_throughput.png"), dpi=120)

    print("wrote L3.01.json + 2 figures")
    write_report(res)


def write_report(res):
    t = res["time"]
    o = res["outcomes"]
    st = res["per_stage_ms"]
    sh = res["stage_share_pct"]
    g = res["gpu"]
    sp = res["speedup_vs_baseline"]
    pol = res["adopted_policy_every2"]
    fr = res["frames_per_window"]
    L = []
    A = L.append
    A("# L3.01 — Full-corpus pipeline at full resources + adopted optimizations\n")
    A(f"**Registry id:** L3.01 · **Branch:** `m/latency-fast` · **Device:** A40 · "
      f"**git_sha:** `{res.get('git_sha')}`\n")
    A("## What this is\n")
    A("The **same harness as L2.01** (`eval/exp_latency_fast/run_fast.py` imports "
      "the repo E8.01 pipeline and only inserts `perf_counter`), run over the full "
      "**1,546-window / 58-clip / 4.404 h** Egocentric-10K corpus at **full "
      "resources** (`--workers 7` on the 7.65-core cgroup, `JADAL_GPU_SLOTS=6`, "
      "NVDEC three-branch decode) with the pod's **adopted optimizations** applied. "
      "The scoring code (`ensemble_prob`/`verdict_for`/`features_block`) is "
      "unchanged, so verdicts stay comparable.\n")
    A("### Adopted optimizations applied\n")
    A("| id | change | flag |")
    A("|---|---|---|")
    A(f"| E1.01 | pixel-26 view at 3.33 fps (~34 vs ~300 frames) | "
      f"`--pixel-fps {res['optimizations_applied']['E1.01_pixel_fps']}` |")
    A(f"| E1.02 | EG2 view at 1 fps (10 vs 20 images) | "
      f"`--eg2-fps {res['optimizations_applied']['E1.02_eg2_fps']}` |")
    A(f"| E3.03 | observe only the first 4 s of a 10 s window | "
      f"`--early-exit-s {res['optimizations_applied']['E3.03_early_exit_s']}` |")
    A(f"| E3.01 | every-2nd-window grid (separate stride-2 run below) | "
      f"`--window-stride {res['optimizations_applied']['E3.01_window_stride']}` |")
    A(f"| E2 | skip the mosaic evidence artifact (not a model input) | "
      f"`--skip-mosaic {res['optimizations_applied']['E2_skip_mosaic']}` |")
    A(f"| X1 | 360p@CRF28 source | not applied — {res['optimizations_applied']['X1_360p_crf28']} |")
    A(f"\nFrames actually decoded per window (p50): pixel "
      f"{fr['pixel'].get('p50')}, EG2 {fr['eg2'].get('p50')}, native "
      f"{fr['native'].get('p50')}.\n")
    A("## Throughput (full corpus, measured)\n")
    A(f"- **Wall:** {t['wall_s']} s ({t['wall_min']} min) · "
      f"**{t['windows_per_s']} windows/s** · "
      f"**{t['footage_h_per_wall_h']} footage-h / wall-h** "
      f"(corpus {t['corpus_hours']} h)")
    A(f"- Per window: mean {t['mean_ms_per_window']} ms, p50 {t['p50_ms_per_window']} ms, "
      f"p95 {t['p95_ms_per_window']} ms · {t['workers']} workers, "
      f"`OMP_NUM_THREADS={t['omp_threads']}`")
    A(f"- vs frozen baseline **{sp['baseline']['windows_per_s']} windows/s** → "
      f"**{sp['overall_throughput_speedup']}x** throughput "
      f"({sp['overall_footage_speedup']}x footage-h/wall-h); "
      f"per-worker-normalized {sp['per_worker_normalized_speedup']}x.\n")
    A("## Per-stage latency (ms / window) vs baseline\n")
    A("| stage | p50 | p95 | p99 | mean | share | baseline p50 | speedup |")
    A("|---|---|---|---|---|---|---|---|")
    for s in ["io_probe", "decode(nvdec3)", "pixel-26", "descriptors",
              "eg2_embed", "mosaic", "decision", "write"]:
        d = st[s]
        bk = {"decode(nvdec3)": "decode", "pixel-26": "pixel_features",
              "eg2_embed": "eg2_embed", "descriptors": "descriptors",
              "mosaic": "mosaic"}.get(s)
        bv = sp["baseline"]["per_window_p50_ms"].get(bk) if bk else None
        su = sp["per_stage_p50_speedup"].get(s)
        A(f"| {s} | {d['p50']} | {d['p95']} | {d['p99']} | {d['mean']} | "
          f"{sh[s]}% | {bv if bv is not None else '—'} | "
          f"{str(su)+'x' if su else '—'} |")
    A(f"\n**Bottleneck:** `{res['bottleneck']}` "
      f"({res['bottleneck_mean_ms']} ms mean, {sh[res['bottleneck']]}% of the "
      "per-window total).\n")
    cp = res.get("clean_probe_1worker")
    if cp:
        A("### Clean (uncontended, 1 worker) per-window cost\n")
        A("The 7-worker run serialises on the single A40 NVDEC engine, which "
          "inflates per-window decode. A 1-worker probe on the same optimized "
          "pipeline recovers the uncontended per-stage cost:\n")
        A("| stage | p50 | baseline p50 | speedup |")
        A("|---|---|---|---|")
        for s in ["decode(nvdec3)", "pixel-26", "descriptors", "eg2_embed",
                  "mosaic"]:
            d = cp["per_stage_ms"].get(s, {})
            bk = {"decode(nvdec3)": "decode", "pixel-26": "pixel_features",
                  "eg2_embed": "eg2_embed", "descriptors": "descriptors",
                  "mosaic": "mosaic"}.get(s)
            bv = sp["baseline"]["per_window_p50_ms"].get(bk)
            su = cp["per_stage_p50_speedup_vs_baseline"].get(s)
            A(f"| {s} | {d.get('p50')} | {bv if bv is not None else '—'} | "
              f"{str(su)+'x' if su else '—'} |")
        A(f"| **total** | **{cp['total_ms']['p50']}** | **4437** | "
          f"**{cp['per_stage_p50_speedup_vs_baseline'].get('total')}x** |")
        d = cp["decode_4s_vs_10s_1worker"]
        A(f"\n1-worker decode: 4 s p50 {d['decode4_p50_ms']} ms vs 10 s p50 "
          f"{d['decode10_p50_ms']} ms — early exit saves only ~"
          f"{round(100*(1-d['decode4_p50_ms']/d['decode10_p50_ms']))}% of decode "
          "because container seek + NVDEC setup dominate, not frame count.\n")
    A("## Outcomes vs the frozen baseline\n")
    A(f"- **Windows:** {o['windows_processed']} processed, failures {o['failures']}.")
    A("\n| verdict | L3 n | L3 share | frozen baseline n | share |")
    A("|---|---|---|---|---|")
    for k in ("accept", "reject", "review"):
        A(f"| {k} | {o['verdict_distribution'].get(k, 0)} | "
          f"{o['verdict_share_pct'].get(k, 0)}% | "
          f"{o['reference_E8_01_verdict_distribution'].get(k, 0)} | "
          f"{o['reference_E8_01_verdict_share_pct'].get(k, 0)}% |")
    A(f"\nPer-window verdict agreement with the E8.01 reference: "
      f"**{o['verdict_agreement_pct']}%** ({o['per_window_compared']} compared, "
      f"{o['verdict_mismatch_count']} mismatches). Profile distribution "
      f"`{o['profile_distribution']}`. {o['note']}\n")
    ss = res.get("score_shift_under_sparse_sampling")
    if ss:
        A("### Score / verdict shift under sparse sampling (frozen heads)\n")
        A(f"- Frozen-head score mean: reference {ss['reference_score_mean']} → "
          f"optimized **{ss['full_run_score_mean']}** "
          f"(Δ **+{ss['delta_mean']}**, Δp50 +{ss['delta_p50']}).")
        if ss.get("variants"):
            A("\n| variant (n=24 spread windows) | score mean | Δ vs baseline | "
              "mean p_eg2 | mean p_pixel |")
            A("|---|---|---|---|---|")
            for name in ("baseline", "pixel_3.33fps", "eg2_1fps", "early_exit_4s",
                         "all_adopted"):
                v = ss["variants"].get(name)
                if not v:
                    continue
                A(f"| {name} | {v['score_mean']} | "
                  f"{v.get('score_mean_delta_vs_baseline')} | "
                  f"{v['mean_p_eg2']} | {v['mean_p_pixel']} |")
            A("\n**Finding:** the score shift is driven almost entirely by the "
              "**E1.01 pixel view at 3.33 fps** (pixel-head probability "
              "p_pixel jumps; EG2 at 1 fps is score-neutral, consistent with "
              "E1.02). The frozen pixel head's standardisation was fit on "
              "full-sampling features, so sparse sampling moves the score and "
              "the verdict distribution **is not close** to the baseline "
              "(accept 296→876). This is a real blocker for shipping sparse "
              "sampling with the *frozen* head: it needs a refit/recalibration "
              "on subsampled features (E1.01's AUROC verdict used a refit "
              "logistic, not the frozen head).\n")
    if pol:
        A("## Adopted every-2nd policy (E3.01)\n")
        d2 = res.get("adopted_policy_every2_derived")
        if d2:
            A(f"- **Derived** from the stride-1 run: same {d2['windows_per_s']} "
              f"windows/s, but each processed window represents 2 windows of "
              f"footage → **{d2['footage_h_per_wall_h']} footage-h/wall-h** "
              f"({d2['speedup_vs_baseline_footage']}x vs baseline).")
        A(f"- **Measured** (stride-2 run, partly contended by a concurrent pod "
          f"job): {pol['n_windows']} windows, wall {pol['wall_s']} s, "
          f"**{pol['windows_per_s']} windows/s**, "
          f"**{pol['footage_h_per_wall_h']} footage-h/wall-h** → "
          f"**{pol['speedup_vs_baseline_throughput']}x** throughput / "
          f"**{pol['speedup_vs_baseline_footage']}x** footage vs baseline.\n")
    A("## GPU / CPU\n")
    A(f"- GPU util over the run: p50 {g['run_util_pct'].get('p50')}%, "
      f"mean {g['run_util_pct'].get('mean')}%, p95 {g['run_util_pct'].get('p95')}%.")
    A(f"- VRAM peak (nvidia-smi): {g['vram_peak_mib']} MiB of 46068 MiB; "
      f"EG2 per-window torch peak p50 "
      f"{g['eg2_vram_peak_mib_per_window'].get('p50')} MiB.")
    A(f"- CPU: {g['cores']['workers']} workers on the "
      f"{g['cores']['cgroup_quota_cores']}-core cgroup quota, "
      f"`OMP_NUM_THREADS={g['cores']['omp_threads']}`.\n")
    A("## Per-clip table\n")
    A("| clip | factory | n win | mean total ms | p95 ms | mean decode ms | "
      "mean eg2 ms | serial s |")
    A("|---|---|---|---|---|---|---|---|")
    for c in res["per_clip_table"]:
        A(f"| {c['clip'].split('/')[-1]} | {c['factory']} | {c['n_windows']} | "
          f"{c['mean_total_ms']} | {c['p95_total_ms']} | {c['mean_decode_ms']} | "
          f"{c['mean_eg2_ms']} | {c['serial_clip_s']} |")
    A("\n## Figures\n")
    A("![per-stage p50/p95 vs baseline]"
      "(../experiments/exp_latency_fast/images/per_stage_p50_p95.png)\n")
    A("![speedup and throughput]"
      "(../experiments/exp_latency_fast/images/speedup_and_throughput.png)\n")
    A("## Caveats\n")
    A(f"- {res['contention_caveat']}")
    A("- Verdicts are **descriptive** (no human labels for these windows); the "
      "accept/reject share is not a quality measure. Sparser sampling can change "
      "individual verdicts; only the distribution is expected to stay close.")
    A("- Early exit observes only 4 s, so the native descriptor branch yields ~3 "
      "frames instead of 8 (documented in `contract.versions.sampling`).")
    A("- X1 (360p@CRF28) was not applied: the corpus ships 1080p HEVC only and the "
      "pipeline reads local disk, so the X1 win (bytes) does not reduce decode "
      "compute.")
    p = os.environ.get("L3_REPORT",
                       os.path.join(REPO, "docs", "reports", "exp_latency_fast.md"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
