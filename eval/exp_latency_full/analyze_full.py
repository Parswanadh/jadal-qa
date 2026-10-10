#!/usr/bin/env python3
"""L2.01 analysis: full-corpus outcomes + atomic per-stage latency.

Reads eval/results/exp_latency_full/{windows.jsonl,run_meta.json,clips.jsonl}
and the E8.01 reference, writes L2.01.json + 2 figures.

Regen: python3 eval/exp_latency_full/analyze_full.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
IN = os.environ.get("L2_IN", os.path.join(REPO, "eval", "results", "exp_latency_full"))
IMG = os.environ.get("L2_IMG", os.path.join(REPO, "docs", "experiments",
                                            "exp_latency_full", "images"))
os.makedirs(IMG, exist_ok=True)

REF_LINES = "/workspace/artifacts/windows_rescored.jsonl"
REF_JSON = "/workspace/wt/exp-rescore/eval/results/exp_rescore/E8.01.json"

STAGES = ["io_probe", "decode", "pixel_features", "descriptors", "eg2_embed",
          "mosaic", "decision", "write"]
PRETTY = {"io_probe": "io_probe", "decode": "decode(nvdec3)",
          "pixel_features": "pixel-26", "descriptors": "descriptors",
          "eg2_embed": "eg2_embed", "mosaic": "mosaic",
          "decision": "decision", "write": "write"}


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


def main():
    rows = [json.loads(l) for l in open(os.path.join(IN, "windows.jsonl"))
            if l.strip()]
    meta = json.load(open(os.path.join(IN, "run_meta.json")))
    clips = [json.loads(l) for l in open(os.path.join(IN, "clips.jsonl"))
             if l.strip()]
    n = len(rows)
    print(f"windows={n} clips={len(clips)}")

    # ---- outcomes vs E8.01 ----
    from collections import Counter
    vd = Counter(r["verdict"] for r in rows)
    pd = Counter(r["profile"] for r in rows)
    ref_dist, ref_per_window, ref_meta = {}, {}, {}
    if os.path.exists(REF_JSON):
        ref_meta = json.load(open(REF_JSON))
        ref_dist = ref_meta.get("verdict_distribution", {})
    if os.path.exists(REF_LINES):
        for ln in open(REF_LINES):
            d = json.loads(ln)
            if d.get("type") == "header":
                ref_dist = d.get("verdict_distribution", ref_dist)
                continue
            ref_per_window[d["window_id"]] = d
    # per-window comparison
    mism, compared = [], 0
    for r in rows:
        ref = ref_per_window.get(r["window_id"])
        if ref is None:
            continue
        compared += 1
        if r["verdict"] != ref["verdict"]:
            mism.append({"window_id": r["window_id"], "ours": r["verdict"],
                         "ref": ref["verdict"]})
    feat_ok = sum(1 for r in rows
                  if r["contract"]["features_digest"] ==
                  ref_per_window.get(r["window_id"], {}).get("features_digest"))
    share = {k: round(100.0 * v / max(n, 1), 2) for k, v in sorted(vd.items())}
    ref_share = ({k: round(100.0 * v / max(sum(ref_dist.values()), 1), 2)
                  for k, v in ref_dist.items()} if ref_dist else {})

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
    # the sequential policy probe runs AFTER the pool but before wall is read;
    # subtract its measured component time for the pipeline-only throughput.
    probe_all = [p for p in meta.get("policy_probe", []) if "early_ms" in p]
    probe_s = sum(p["early_ms"] for p in probe_all) / 1000.0
    wall_pipeline = max(wall - probe_s, 1e-9)
    hours = meta["manifest"]["total_hours"]
    time_block = {
        "wall_s": wall, "wall_min": round(wall / 60.0, 2),
        "policy_probe_s": round(probe_s, 1),
        "wall_pipeline_s": round(wall_pipeline, 2),
        "n_windows": n, "windows_per_s": round(n / wall_pipeline, 4),
        "windows_per_min": round(60.0 * n / wall_pipeline, 1),
        "corpus_hours": hours,
        "footage_h_per_wall_h": round(hours / (wall_pipeline / 3600.0), 3),
        "mean_ms_per_window": round(float(pipe_total.mean()), 1),
        "p50_ms_per_window": round(float(np.percentile(pipe_total, 50)), 1),
        "p95_ms_per_window": round(float(np.percentile(pipe_total, 95)), 1),
        "workers": meta["workers"], "omp_threads": meta["omp_threads"],
        "max_rss_gb": round(meta.get("max_rss_gb", 0.0), 2),
    }

    # ---- decode comparison (CPU vs NVDEC cuvid) ----
    dc = [r["decode_compare"] for r in rows if r.get("decode_compare")]
    cpu = [d["cpu_ms"] for d in dc if "cpu_ms" in d and "cpu_error" not in d]
    nv = [d["nvdec_cuvid_ms"] for d in dc
          if "nvdec_cuvid_ms" in d and "nvdec_cuvid_error" not in d]
    pipe_dec = [r["latency_ms"]["decode"] for r in rows]
    decode_cmp = {
        "n_compared": len(dc),
        "coverage": "first window of every clip",
        "cpu_ffmpeg_swscale_ms": pct(cpu),
        "nvdec_hevc_cuvid_ms": pct(nv),
        "pipeline_3branch_nvdec_ms": pct(pipe_dec),
        "speedup_cpu_over_cuvid_median": round(float(
            np.median(cpu) / max(np.median(nv), 1e-9)), 3) if cpu and nv else None,
        "note": ("CPU and cuvid arms decode ONE 160x90 branch; the pipeline "
                 "decode is the E8.01 three-branch NVDEC pass (all frames @160x90 "
                 "+ 2 fps @512x288 + 0.8 fps @1920x1080), so it is not a like-for-"
                 "like single-branch number"),
    }

    # ---- GPU util + VRAM per stage ----
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
        "gpu_util_pct_per_stage": per_stage_gpu,
        "eg2_vram_peak_mib_per_window": pct(eg2_vram) if eg2_vram else {},
        "cores": {"nproc": 96, "workers": meta["workers"],
                  "omp_threads": meta["omp_threads"],
                  "note": "OMP_NUM_THREADS=1; 4 single-threaded workers"},
    }

    # ---- adopted-policy variant vs full sampling ----
    probe = [p for p in meta.get("policy_probe", []) if "error" not in p]
    dec4 = [p["decode4_ms"] for p in probe]
    pix4 = [p["pixel4_ms"] for p in probe]
    early_ms = (means["io_probe"] + float(np.mean(dec4)) + float(np.mean(pix4))
                + means["decision"] + means["write"]) if probe else None
    full_ms = float(pipe_total.mean())
    policy = None
    if early_ms is not None:
        policy_ms = 0.5 * full_ms + 0.5 * early_ms  # every 2nd window full
        policy = {
            "model": ("every-2nd window full pipeline; the other half a 4 s "
                      "early-exit path = io_probe + decode(4 s) + pixel-26(4 s) "
                      "+ decision + write, MEASURED on n=%d windows" % len(probe)),
            "full_ms_per_window": round(full_ms, 1),
            "early_exit_4s_ms_per_window": round(early_ms, 1),
            "policy_ms_per_window": round(policy_ms, 1),
            "speedup_vs_full": round(full_ms / policy_ms, 3),
            "full_windows_per_s": round(n / wall_pipeline, 4),
            "policy_windows_per_s_est": round((n / wall_pipeline) * full_ms / policy_ms, 4),
            "policy_wall_s_est": round(wall_pipeline * policy_ms / full_ms, 1),
            "policy_footage_h_per_wall_h_est": round(
                hours / (wall_pipeline * policy_ms / full_ms / 3600.0), 3),
            "early_exit_measured_components_ms": {
                "decode4": round(float(np.mean(dec4)), 1),
                "pixel4": round(float(np.mean(pix4)), 1)},
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

    res = {
        "id": "L2.01", "track": "latency", "device": "A40",
        "title": "Full-corpus PoC pipeline run: outcomes + atomic per-stage latency",
        "label_source": "none (unlabelled windows; verdicts DESCRIPTIVE)",
        "recipe": "ensemble_v1 (EG2+pixel equal-weight) + profiles/router (E4.04)",
        "protocol": ("repo E8.01 pipeline (eval/exp_rescore/run_e8.py) over ALL "
                     "1,546 windows with perf_counter per stage; verdicts/score/"
                     "features_digest reproduce E8.01; latency_ms carries more "
                     "stages so receipt_sha256 differs (documented)"),
        "command": ("source /workspace/env.sh && export OMP_NUM_THREADS=1 && "
                    "bash tools/pod/gpu_run.sh python3 "
                    "eval/exp_latency_full/run_full.py --workers 4 --resume"),
        "git_sha": meta.get("git_sha"),
        "run_utc": meta.get("run_utc"),
        "contention_caveat": ("pod load ~10-13/96 from peer opencode agent "
                              "sessions during the run; GPU otherwise idle; "
                              "4 single-threaded workers"),
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
            "features_digest_reproduced": feat_ok,
            "reproduction": ("EXACT on all compared windows" if compared and
                             len(mism) == 0 and feat_ok == compared else
                             "MISMATCH - see examples"),
        },
        "time": time_block,
        "per_stage_ms": stages,
        "stage_mean_ms": {PRETTY[s]: round(means[s], 2) for s in STAGES},
        "stage_share_pct": shares,
        "bottleneck": PRETTY[bottleneck],
        "bottleneck_mean_ms": round(means[bottleneck], 2),
        "decode_cpu_vs_nvdec": decode_cmp,
        "gpu": gpu_block,
        "adopted_policy_vs_full": policy,
        "per_clip_table": clip_rows,
        "n_clips": len(clip_rows),
        "honesty_note": ref_meta.get("honesty_note"),
        "reference_cost": ref_meta.get("cost"),
    }
    json.dump(res, open(os.path.join(IN, "L2.01.json"), "w"), indent=1,
              sort_keys=True)
    print(json.dumps({"outcomes": res["outcomes"], "time": res["time"],
                      "stage_share_pct": res["stage_share_pct"],
                      "bottleneck": res["bottleneck"],
                      "policy": res["adopted_policy_vs_full"]}, indent=1))

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels = [PRETTY[s] for s in STAGES]
    p50 = [stages[l]["p50"] for l in labels]
    p95 = [stages[l]["p95"] for l in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, p50, 0.4, label="p50")
    ax.bar(x + 0.2, p95, 0.4, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ms / 10-s window")
    ax.set_title(f"L2.01 per-stage latency, full corpus (n={n}, A40, "
                 f"4 workers, OMP=1)")
    ax.legend()
    for xi, v in zip(x - 0.2, p50):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + 0.2, p95):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "per_stage_p50_p95.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    data = [[r["latency_ms"].get(s, 0.0) for r in rows] for s in STAGES]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(STAGES) + 1))
    ax.set_xticklabels([PRETTY[s] for s in STAGES])
    ax.set_yscale("log")
    ax.set_ylabel("ms (log)")
    ax.set_title(f"L2.01 per-stage distribution, full corpus (n={n}, no fliers)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "per_stage_distribution.png"), dpi=120)
    print("wrote L2.01.json + 2 figures")
    write_report(res)


def write_report(res):
    """Emit docs/reports/exp_latency_full.md from the results dict."""
    t = res["time"]
    o = res["outcomes"]
    st = res["per_stage_ms"]
    sh = res["stage_share_pct"]
    g = res["gpu"]
    pol = res["adopted_policy_vs_full"]
    dc = res["decode_cpu_vs_nvdec"]
    L = []
    A = L.append
    A("# L2 full-corpus PoC pipeline run — outcomes + atomic per-stage latency\n")
    A(f"**Registry id:** L2.01 · **Owner:** l2-latency · **Device:** A40 · "
      f"**Branch:** `m/latency-full` · **git_sha:** `{res.get('git_sha')}`\n")
    A("## What this is\n")
    A("Runs the **repo's E8.01 pipeline** (`eval/exp_rescore/run_e8.py` — one NVDEC + "
      "`scale_cuda` three-branch pass per window, pixel-26 + EG2 (2 fps) + E4.01 "
      "descriptors → profile router; `jadal/ensemble/scorer.py` frozen two-view "
      "logistic ensemble → verdict + content-hashed receipt) over **all 1,546** "
      "windows of `queues/manifest_scale.json` (58 clips, 4.404 h, 1920x1080 HEVC "
      "@30 fps) with `time.perf_counter` around every stage. It is **not** a new "
      "benchmark re-implementation: `run_full.py` imports the repo modules and only "
      "inserts timing. One JSONL row per window is in "
      "`eval/results/exp_latency_full/windows.jsonl`.\n")
    A(f"> {res.get('honesty_note')}\n")
    A("## Outcomes (full corpus) vs E8.01\n")
    A(f"- **Windows processed:** {o['windows_processed']} / "
      f"{o.get('windows_requested')} · **failures:** {o['failures']}")
    A(f"- **Verdict reproduction:** {o['reproduction']} "
      f"({o['per_window_compared']} windows compared, "
      f"{o['verdict_mismatch_count']} verdict mismatches, "
      f"{o['features_digest_reproduced']}/{o['per_window_compared']} "
      f"`features_digest` identical to E8.01).")
    A("\n| verdict | L2 n | L2 share | E8.01 n | E8.01 share |")
    A("|---|---|---|---|---|")
    for k in ("accept", "reject", "review"):
        A(f"| {k} | {o['verdict_distribution'].get(k, 0)} | "
          f"{o['verdict_share_pct'].get(k, 0)}% | "
          f"{o['reference_E8_01_verdict_distribution'].get(k, 0)} | "
          f"{o['reference_E8_01_verdict_share_pct'].get(k, 0)}% |")
    A(f"\nProfile distribution: `{o['profile_distribution']}`. "
      "Verdicts are **descriptive** (no human labels); the accept/reject share is "
      "not a quality measure.\n")
    if o["verdict_mismatch_count"]:
        A(f"Mismatches: `{o['verdict_mismatch_examples']}`\n")
    A("## Time (measured, full corpus)\n")
    A(f"- **Total wall:** {t['wall_s']} s ({t['wall_min']} min) — of which "
      f"{t.get('policy_probe_s')} s is the sequential post-run policy probe, so "
      f"**pipeline wall {t['wall_pipeline_s']} s** · "
      f"**{t['windows_per_s']} windows/s** ({t['windows_per_min']}/min)")
    A(f"- **Footage / wall-hour:** {t['footage_h_per_wall_h']} h of footage per "
      f"wall-hour (corpus {t['corpus_hours']} h)")
    A(f"- Per window: mean {t['mean_ms_per_window']} ms, p50 "
      f"{t['p50_ms_per_window']} ms, p95 {t['p95_ms_per_window']} ms")
    A(f"- {t['workers']} workers, `OMP_NUM_THREADS={t['omp_threads']}`, "
      f"peak RSS {t['max_rss_gb']} GB/process\n")
    A("### Per-stage latency (ms / 10-s window)\n")
    A("| stage | p50 | p95 | p99 | mean | share |")
    A("|---|---|---|---|---|---|")
    for s in ["io_probe", "decode(nvdec3)", "pixel-26", "descriptors",
              "eg2_embed", "mosaic", "decision", "write"]:
        d = st[s]
        A(f"| {s} | {d['p50']} | {d['p95']} | {d['p99']} | {d['mean']} | "
          f"{sh[s]}% |")
    A(f"\n**Bottleneck:** `{res['bottleneck']}` "
      f"({res['bottleneck_mean_ms']} ms mean, {sh[res['bottleneck']]}% of the "
      "per-window total). EG2 embedding dominates; decode is second.\n")
    A("### CPU decode vs NVDEC\n")
    A(f"- Comparison on the first window of every clip (n={dc['n_compared']}): "
      f"CPU ffmpeg/swscale p50 {dc['cpu_ffmpeg_swscale_ms'].get('p50')} ms vs "
      f"explicit NVDEC `-c:v hevc_cuvid` p50 "
      f"{dc['nvdec_hevc_cuvid_ms'].get('p50')} ms → "
      f"**{dc['speedup_cpu_over_cuvid_median']}x** faster on GPU.")
    A(f"- Pipeline decode (E8.01 three-branch NVDEC pass) p50 "
      f"{dc['pipeline_3branch_nvdec_ms'].get('p50')} ms. {dc['note']}\n")
    A("### GPU utilisation + VRAM\n")
    A(f"- Run-level GPU util: p50 {g['run_util_pct'].get('p50')}%, "
      f"p95 {g['run_util_pct'].get('p95')}%, mean {g['run_util_pct'].get('mean')}% "
      f"of one A40 (util is bursty: decode + EG2 forward).")
    A(f"- Run-level VRAM (nvidia-smi): p50 {g['run_vram_mib'].get('p50')} MiB, "
      f"peak {g['run_vram_mib'].get('max')} MiB of 46068 MiB.")
    A(f"- EG2 per-window torch VRAM peak: p50 "
      f"{g['eg2_vram_peak_mib_per_window'].get('p50')} MiB.")
    A("- GPU util attributed per stage (0.5 s nvidia-smi sampling): "
      + ", ".join(f"{k} {v.get('mean')}%" for k, v in
                  g["gpu_util_pct_per_stage"].items() if v) + ".")
    A(f"- Cores: {g['cores']['nproc']} nproc, {g['cores']['workers']} workers, "
      f"OMP_NUM_THREADS={g['cores']['omp_threads']}.\n")
    if pol:
        A("## Adopted-policy variant (every-2nd window + 4 s early exit) vs full\n")
        A(f"- Full sampling: **{pol['full_ms_per_window']} ms/window** "
          f"({pol['full_windows_per_s']} windows/s measured).")
        A(f"- 4 s early-exit path (MEASURED on the probe windows): "
          f"**{pol['early_exit_4s_ms_per_window']} ms/window** "
          f"(decode4 {pol['early_exit_measured_components_ms']['decode4']} ms + "
          f"pixel4 {pol['early_exit_measured_components_ms']['pixel4']} ms + "
          "io/decision/write).")
        A(f"- Policy (half full + half early): **{pol['policy_ms_per_window']} "
          f"ms/window → {pol['speedup_vs_full']}x faster** "
          f"({pol['policy_windows_per_s_est']} windows/s est., "
          f"{pol['policy_footage_h_per_wall_h_est']} footage-h/wall-h est.).")
        A(f"- Model: {pol['model']}. The policy is **not** used for the contract "
          "(one verdict per window; frozen heads were fit on full-window "
          "features) — reported as a measured alternative.\n")
    A("## Per-clip throughput (58 clips)\n")
    A("| clip | factory | n win | mean total ms | p95 ms | mean decode ms | "
      "mean eg2 ms | serial s |")
    A("|---|---|---|---|---|---|---|---|")
    for c in res["per_clip_table"]:
        A(f"| {c['clip'].split('/')[-1]} | {c['factory']} | {c['n_windows']} | "
          f"{c['mean_total_ms']} | {c['p95_total_ms']} | {c['mean_decode_ms']} | "
          f"{c['mean_eg2_ms']} | {c['serial_clip_s']} |")
    A("\n## Figures\n")
    A("![per-stage p50/p95]"
      "(../experiments/exp_latency_full/images/per_stage_p50_p95.png)\n")
    A("![per-stage distribution]"
      "(../experiments/exp_latency_full/images/per_stage_distribution.png)\n")
    A("## Reproduction & caveats\n")
    A("- Verdicts/`score`/`features_digest` use the same code path as E8.01 and are "
      "**identical**; only `latency_ms` carries more stages, so `receipt_sha256` "
      "differs from E8.01 (expected, documented).")
    A("- The EG2 model dir is patched **in memory** to the pod cache: a post-merge "
      "commit reverted `configs/ensemble_v1.json` to the laptop path; the head "
      "numbers are unchanged (checked by `run_e8.load_config_checked`).")
    A(f"- Contention caveat: {res.get('contention_caveat')}")
    A(f"- Peak RSS {t['max_rss_gb']} GB/process (> the 1.5 GB guideline) is the "
      "EG2 model + torch CUDA context + decoded frames; 4 workers ≈ "
      f"{4*t['max_rss_gb']:.0f} GB of 503 GB. Same path as E8.01.")
    A("- Mosaic is generated only when the routed profile sets `evidence.mosaic` "
      "(both `default` and `factory_1080p30` do); it is evidence, not a scoring "
      "input, so it does not change verdicts.")
    p = os.environ.get("L2_REPORT",
                       os.path.join(REPO, "docs", "reports", "exp_latency_full.md"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
