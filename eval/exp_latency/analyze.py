"""L1.01 analysis: percentiles, shares, profiles, policy model, figures, L1.01.json."""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
IN = os.path.join(REPO, "eval", "results", "exp_latency")
IMG = os.path.join(REPO, "docs", "experiments", "exp_latency", "images")
os.makedirs(IMG, exist_ok=True)

STAGES = ["io_probe_ms", "decode_cpu_ms", "decode_nvdec_ms",
          "features_pixel26_fps10_ms", "features_pixel26_fps5_ms",
          "eg2_embed_ms", "mosaic_ms", "decision_ms", "write_ms"]
PRETTY = {"io_probe_ms": "io_probe", "decode_cpu_ms": "decode_cpu",
          "decode_nvdec_ms": "decode_nvdec",
          "features_pixel26_fps10_ms": "features@10fps",
          "features_pixel26_fps5_ms": "features@5fps",
          "eg2_embed_ms": "eg2_embed", "mosaic_ms": "mosaic",
          "decision_ms": "decision", "write_ms": "write"}


def pct(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def main():
    wins = [json.loads(l) for l in open(os.path.join(IN, "windows.jsonl"))
            if l.strip()]
    clips = json.load(open(os.path.join(IN, "clips_decode.json")))
    meta = json.load(open(os.path.join(IN, "run_meta.json")))
    ok = [w for w in wins if "decode_cpu_error" not in w]
    print(f"windows={len(wins)} ok={len(ok)} clips={len(clips)}")

    stats = {PRETTY[s]: pct([w[s] for w in ok]) for s in STAGES}
    # shares of summed stage means (default profile: cpu? use nvdec path)
    pipe_default = ["io_probe_ms", "decode_nvdec_ms",
                    "features_pixel26_fps10_ms", "eg2_embed_ms",
                    "mosaic_ms", "decision_ms", "write_ms"]
    pipe_factory = ["io_probe_ms", "decode_nvdec_ms",
                    "features_pixel26_fps5_ms", "eg2_embed_ms",
                    "mosaic_ms", "decision_ms", "write_ms"]
    means = {s: float(np.mean([w[s] for w in ok])) for s in STAGES}

    def pipe_stats(pipe):
        per_win = [sum(w[s] for s in pipe) for w in ok]
        tot = float(np.sum(per_win))
        shares = {PRETTY[s]: float(np.mean([w[s] for w in ok]) /
                  (sum(means[s] for s in pipe))) for s in pipe}
        return {"per_window_ms": pct(per_win), "total_share": shares,
                "mean_sum_ms": float(sum(means[s] for s in pipe))}

    default = pipe_stats(pipe_default)
    factory = pipe_stats(pipe_factory)
    bottleneck = max(pipe_default, key=lambda s: means[s])

    N = meta["manifest"]["n_windows"]
    HRS = meta["manifest"]["total_hours"]
    # extrapolate: full corpus at measured per-window means
    def xrate(mean_ms):
        wall_s = N * mean_ms / 1000.0
        return {"wall_s": wall_s, "wall_h": wall_s / 3600.0,
                "footage_per_wall_h": HRS / (wall_s / 3600.0),
                "windows_per_s": N / wall_s,
                "per_window_ms": mean_ms}

    default_full = xrate(default["mean_sum_ms"])
    factory_full = xrate(factory["mean_sum_ms"])
    # adopted policy: every-2nd full + 4s early-exit on the others (modelled)
    early_ms = (means["io_probe_ms"] + 0.4 * means["decode_nvdec_ms"]
                + 0.4 * means["features_pixel26_fps5_ms"]
                + means["decision_ms"] + means["write_ms"])
    policy_mean = 0.5 * factory["mean_sum_ms"] + 0.5 * early_ms
    policy = xrate(policy_mean)
    policy["early_exit_assumed_ms"] = early_ms
    policy["model"] = ("measured components; 4s-exit = io + 0.4*dec_nvdec + "
                       "0.4*feat5 + decision + write; assumes 50% windows exit")

    # CPU vs NVDEC decode (same windows)
    cpu = np.array([w["decode_cpu_ms"] for w in ok])
    nv = np.array([w["decode_nvdec_ms"] for w in ok])
    f_cpu = np.array([w.get("decode_cpu_frames", 0) for w in ok])
    f_nv = np.array([w.get("decode_nvdec_frames", 0) for w in ok])
    # per-clip table (measured NVDEC + cold probe; full-pipe extrapolated)
    clip_rows = []
    for c in clips:
        clip_rows.append({
            "clip": c["clip"], "factory": c.get("factory"),
            "io_probe_cold_ms": c.get("io_probe_cold_ms"),
            "dec_nvdec_ms_measured": c.get("dec_nvdec_ms"),
            "dec_nvdec_frames": c.get("dec_nvdec_frames"),
            "full_pipe_default_ms_extrap": default["mean_sum_ms"],
            "full_pipe_factory_ms_extrap": factory["mean_sum_ms"]})

    # GPU util per stage from smi samples: attribute by wall-clock bounds is
    # unavailable post-hoc; report run-level util distribution + EG2 VRAM peaks
    smi = meta.get("smi", [])
    util = [s["gpu_pct"] for s in smi]
    vram = [s["vram_mib"] for s in smi]
    eg2_vram = [w.get("eg2_vram_peak_mib") for w in ok
                if w.get("eg2_vram_peak_mib")]

    res = {
        "id": "L1.01", "track": "latency", "device": "A40",
        "label_source": "unlabelled (timing only; verdicts not evaluated)",
        "protocol": ("atomic perf_counter per stage; 24 windows x 12 clips x "
                     "N factories timed fully; 58-clip NVDEC+probe sweep; "
                     "full-corpus numbers EXTRAPOLATED from measured means"),
        "n": {"windows_timed": len(ok), "clips_atomic": len(meta["atomic_clips"]),
              "factories_atomic": meta["atomic_factories"],
              "clips_decode_sweep": len(clips),
              "corpus_windows": N, "corpus_clips": meta["manifest"]["n_clips"],
              "corpus_hours": HRS},
        "command": ("export OMP_NUM_THREADS=1; bash tools/pod/gpu_run.sh "
                    "python3 eval/exp_latency/run_atomic.py"),
        "git_sha": os.popen("git -C " + REPO + " rev-parse --short HEAD").read().strip(),
        "contention_caveat": ("pod load ~14/96 during run; h2_mosaic1080 + "
                              "p1_calib alive; GPU util low (~27%); CPU-decode "
                              "numbers may be mildly inflated"),
        "per_stage_ms": stats,
        "pipe_default_ms": default, "pipe_factory_ms": factory,
        "bottleneck": PRETTY[bottleneck],
        "decode_cpu_vs_nvdec": {
            "cpu_ms": pct(cpu), "nvdec_ms": pct(nv),
            "speedup_median": float(np.median(cpu) / max(np.median(nv), 1e-9)),
            "cpu_fps_median": float(np.median(f_cpu / (cpu / 1000.0))),
            "nvdec_fps_median": float(np.median(f_nv / (nv / 1000.0)))},
        "full_corpus_extrapolated": {
            "default_profile": default_full,
            "factory_1080p30_profile": factory_full,
            "adopted_policy_every2nd_plus_4s_exit_modelled": policy},
        "measured_subset_totals": {
            "windows": len(ok),
            "sum_full_pipe_default_s": float(sum(sum(w[s] for s in pipe_default)
                                                 for w in ok) / 1000.0)},
        "gpu": {"smi_samples": len(smi),
                "util_pct": pct(util) if util else {},
                "vram_mib": pct(vram) if vram else {},
                "eg2_vram_peak_mib_per_window": pct(eg2_vram) if eg2_vram else {},
                "eg2_model_load_s": meta.get("eg2_model_load_s")},
        "floor": meta.get("floor"),
        "cpu": {"omp_threads": meta.get("omp_threads"),
                "max_rss_gb": meta.get("max_rss_gb")},
        "per_clip_table": clip_rows,
        "early_exit_note": ("4s-exit cost is modelled by linear scaling of "
                            "measured decode/feature means, not directly timed"),
    }
    json.dump(res, open(os.path.join(IN, "L1.01.json"), "w"), indent=1)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [PRETTY[s] for s in pipe_default]
    mns = [means[s] for s in pipe_default]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, mns)
    ax.set_ylabel("mean ms / 10-s window")
    ax.set_title("L1.01 per-stage latency means (default profile, NVDEC path, "
                 f"n={len(ok)} windows, A40)")
    for x, v in zip(labels, mns):
        ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "latency_breakdown.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    keys = [PRETTY[s] for s in STAGES]
    data = [[w[s] for w in ok] for s in STAGES]
    bp = ax.boxplot(data, labels=keys, showfliers=False)
    ax.set_ylabel("ms (log)")
    ax.set_yscale("log")
    ax.set_title("L1.01 per-stage latency distribution (median/IQR, no fliers, "
                 "log scale)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "per_stage_percentiles.png"), dpi=120)
    print("wrote L1.01.json + 2 figures")
    print(json.dumps({k: round(v, 1) for k, v in means.items()}, indent=1))
    print("default/pipe ms:", round(default["mean_sum_ms"], 1),
          "factory:", round(factory["mean_sum_ms"], 1),
          "policy:", round(policy_mean, 1))


if __name__ == "__main__":
    main()
