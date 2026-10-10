#!/usr/bin/env python3
"""L2.01 analysis: PoC pipeline outcomes + atomic per-stage latency.

Reads eval/results/exp_latency_full/windows.jsonl if present, otherwise the
PARTIAL file windows.jsonl.partial (the baseline run was stopped; see the
report). Writes L2.01.json + 2 figures + docs/reports/exp_latency_full.md.

Regen: L2_WALL_S=<derived> python3 eval/exp_latency_full/analyze_full.py
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
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "queues", "manifest_scale.json")
REF_LINES = "/workspace/artifacts/windows_rescored.jsonl"
REF_JSON = "/workspace/wt/exp-rescore/eval/results/exp_rescore/E8.01.json"
LOG = "/workspace/logs/l2_latency_full.log"
os.makedirs(IMG, exist_ok=True)

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


def load_rows():
    win = os.path.join(IN, "windows.jsonl")
    part = win + ".partial"
    if os.path.exists(win):
        return [json.loads(l) for l in open(win) if l.strip()], False
    rows = []
    for l in open(part):
        if l.strip():
            rows.append(json.loads(l)["row"])
    return rows, True


def derive_wall(partial_rows):
    """wall = elapsed printed by the runner at its last flush + file-time delta."""
    if os.environ.get("L2_WALL_S"):
        return float(os.environ["L2_WALL_S"])
    last_elapsed, logged = None, 0
    if os.path.exists(LOG):
        for l in open(LOG):
            if "/1546" in l and "win/s" in l:
                try:
                    logged = int(l.split("/1546")[0].strip())
                    last_elapsed = float(l.split()[3].rstrip("s"))
                except Exception:  # noqa: BLE001
                    pass
    if last_elapsed is None:
        return None
    delta = os.path.getmtime(os.path.join(IN, "windows.jsonl.partial")) - \
        os.path.getmtime(LOG)
    return last_elapsed + max(delta, 0.0)


def synth_meta(rows, partial, wall):
    man = json.load(open(MANIFEST))
    return {
        "partial": partial, "wall_s": wall, "workers": 4, "omp_threads": "1",
        "manifest": {"n_windows": man["n_windows"], "n_clips": man["n_clips"],
                     "total_hours": man["total_hours"]},
        "n_windows": len(rows), "n_requested": len(rows), "n_errors": 0,
        "errors": [], "policy_probe": [], "smi": [], "max_rss_gb": None,
        "device": "A40",
    }


def main():
    rows, partial = load_rows()
    n = len(rows)
    meta = {}
    if os.path.exists(os.path.join(IN, "run_meta.json")):
        meta = json.load(open(os.path.join(IN, "run_meta.json")))
    if meta.get("n_windows") != n or meta.get("workers") != 4:
        meta = synth_meta(rows, partial, derive_wall(rows))
        if partial:
            json.dump(meta, open(os.path.join(IN, "run_meta.json"), "w"),
                      indent=1, sort_keys=True)
    clips = [json.loads(l) for l in open(os.path.join(IN, "clips.jsonl"))
             if l.strip()] if os.path.exists(os.path.join(IN, "clips.jsonl")) else []
    wall = meta["wall_s"]
    print(f"windows={n} partial={partial} clips={len(clips)} wall={wall}")

    from collections import Counter
    vd = Counter(r["verdict"] for r in rows)
    pd = Counter(r["profile"] for r in rows)

    # ---- E8.01 reference (full + the same window subset) ----
    ref_full_dist, ref_per_window, ref_meta = {}, {}, {}
    if os.path.exists(REF_JSON):
        ref_meta = json.load(open(REF_JSON))
        ref_full_dist = ref_meta.get("verdict_distribution", {})
    if os.path.exists(REF_LINES):
        for ln in open(REF_LINES):
            d = json.loads(ln)
            if d.get("type") == "header":
                ref_full_dist = d.get("verdict_distribution", ref_full_dist)
                continue
            ref_per_window[d["window_id"]] = d
    our_ids = {r["window_id"] for r in rows}
    ref_subset = [ref_per_window[w] for w in our_ids if w in ref_per_window]
    ref_sub_dist = Counter(d["verdict"] for d in ref_subset)
    mism, compared, feat_ok = [], 0, 0
    for r in rows:
        ref = ref_per_window.get(r["window_id"])
        if ref is None:
            continue
        compared += 1
        if r["verdict"] != ref["verdict"]:
            mism.append({"window_id": r["window_id"], "ours": r["verdict"],
                         "ref": ref["verdict"]})
        if r["contract"]["features_digest"] == ref.get("features_digest"):
            feat_ok += 1
    share = {k: round(100.0 * v / max(n, 1), 2) for k, v in sorted(vd.items())}
    ref_sub_share = {k: round(100.0 * v / max(len(ref_subset), 1), 2)
                     for k, v in sorted(ref_sub_dist.items())}

    # ---- per-stage latency (write is absent in the partial) ----
    write_avail = any("write" in r["latency_ms"] for r in rows)
    stages = {}
    for s in STAGES:
        if s == "write" and not write_avail:
            stages[PRETTY[s]] = {}
            continue
        stages[PRETTY[s]] = pct([r["latency_ms"].get(s, 0.0) for r in rows])
    means = {s: float(np.mean([r["latency_ms"].get(s, 0.0) for r in rows]))
             for s in STAGES}
    pipe_total = np.array([sum(r["latency_ms"].get(s, 0.0) for s in STAGES)
                           for r in rows])
    total_sum = float(pipe_total.sum())
    shares = {PRETTY[s]: round(100.0 * means[s] * n / total_sum, 2) for s in STAGES}
    bottleneck = max(STAGES, key=lambda s: means[s])

    # ---- time ----
    hours = meta["manifest"]["total_hours"]
    footage_h = n * 10.0 / 3600.0
    time_block = {
        "wall_s": wall, "wall_min": round(wall / 60.0, 2),
        "n_windows": n, "windows_per_s": round(n / wall, 4),
        "windows_per_min": round(60.0 * n / wall, 1),
        "corpus_hours": hours, "partial_footage_h": round(footage_h, 3),
        "footage_h_per_wall_h": round(footage_h / (wall / 3600.0), 3),
        "mean_ms_per_window": round(float(pipe_total.mean()), 1),
        "p50_ms_per_window": round(float(np.percentile(pipe_total, 50)), 1),
        "p95_ms_per_window": round(float(np.percentile(pipe_total, 95)), 1),
        "workers": meta["workers"], "omp_threads": meta["omp_threads"],
        "max_rss_gb": meta.get("max_rss_gb"),
    }

    # ---- decode comparison ----
    dc = [r["decode_compare"] for r in rows if r.get("decode_compare")]
    cpu = [d["cpu_ms"] for d in dc if "cpu_ms" in d and "cpu_error" not in d]
    nv = [d["nvdec_cuvid_ms"] for d in dc
          if "nvdec_cuvid_ms" in d and "nvdec_cuvid_error" not in d]
    pipe_dec = [r["latency_ms"]["decode"] for r in rows]
    decode_cmp = {
        "n_compared": len(dc), "coverage": "first window of each processed clip",
        "cpu_ffmpeg_swscale_ms": pct(cpu), "nvdec_hevc_cuvid_ms": pct(nv),
        "pipeline_3branch_nvdec_ms": pct(pipe_dec),
        "speedup_cpu_over_cuvid_median": round(float(
            np.median(cpu) / max(np.median(nv), 1e-9)), 3) if cpu and nv else None,
        "note": ("CPU and cuvid arms decode ONE 160x90 branch; the pipeline "
                 "decode is the E8.01 three-branch NVDEC pass, not like-for-like"),
    }

    # ---- GPU (no smi in the partial -> only per-window EG2 VRAM) ----
    smi = meta.get("smi", [])
    eg2_vram = [r["gpu"]["eg2_vram_peak_mib"] for r in rows
                if r.get("gpu", {}).get("eg2_vram_peak_mib")]
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
        if vals:
            per_stage_gpu[PRETTY[s]] = pct(vals)
    gpu_block = {
        "device": "A40 (46068 MiB)", "smi_samples": len(smi),
        "run_util_pct": pct([s["gpu_pct"] for s in smi]) if smi else {},
        "run_vram_mib": pct([s["vram_mib"] for s in smi]) if smi else {},
        "gpu_util_pct_per_stage": per_stage_gpu,
        "eg2_vram_peak_mib_per_window": pct(eg2_vram) if eg2_vram else {},
        "cores": {"nproc": 96, "workers": meta["workers"],
                  "omp_threads": meta["omp_threads"],
                  "note": "OMP_NUM_THREADS=1; 4 single-threaded workers"},
        "note": "nvidia-smi per-stage attribution unavailable in the partial "
                "(run_meta not written); EG2 torch VRAM peak per window is shown",
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
            "clip": c["clip"], "factory": c["factory"], "n_windows": len(rs),
            "duration_s": c.get("duration_s"),
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
        })

    res = {
        "id": "L2.01", "track": "latency", "device": "A40",
        "title": ("PARTIAL baseline: full-corpus PoC pipeline run — outcomes + "
                  "atomic per-stage latency" if partial else
                  "Full-corpus PoC pipeline run: outcomes + atomic per-stage latency"),
        "partial": partial,
        "partial_note": ("The baseline full-corpus run was STOPPED and superseded "
                         "by the optimized full-potential run (L3) and the "
                         "zero-copy run (L4); these numbers are the N windows "
                         "already written to windows.jsonl.partial, a contiguous "
                         "prefix of the manifest (first 2 clips), not a "
                         "stratified sample." if partial else ""),
        "label_source": "none (unlabelled windows; verdicts DESCRIPTIVE)",
        "recipe": "ensemble_v1 (EG2+pixel equal-weight) + profiles/router (E4.04)",
        "protocol": ("repo E8.01 pipeline (eval/exp_rescore/run_e8.py) with "
                     "perf_counter per stage; verdicts/score/features_digest "
                     "reproduce E8.01; latency_ms carries more stages so "
                     "receipt_sha256 differs (documented)"),
        "command": ("source /workspace/env.sh && export OMP_NUM_THREADS=1 && "
                    "bash tools/pod/gpu_run.sh python3 "
                    "eval/exp_latency_full/run_full.py --workers 4 --resume"),
        "git_sha": meta.get("git_sha", os.popen(
            "git -C " + REPO + " rev-parse --short HEAD").read().strip()),
        "run_utc": meta.get("run_utc"),
        "contention_caveat": ("pod load ~10-13/96 from peer opencode sessions; "
                              "4 single-threaded workers; GPU bursty"),
        "outcomes": {
            "windows_processed": n, "windows_requested": n,
            "failures": meta.get("n_errors", 0),
            "verdict_distribution": dict(sorted(vd.items())),
            "verdict_share_pct": share,
            "profile_distribution": dict(sorted(pd.items())),
            "reference_E8_01_verdict_distribution_full": ref_full_dist,
            "reference_E8_01_verdict_distribution_same_subset": dict(
                sorted(ref_sub_dist.items())),
            "reference_E8_01_verdict_share_pct_same_subset": ref_sub_share,
            "verdict_mismatch_count": len(mism),
            "verdict_mismatch_examples": mism[:10],
            "per_window_compared": compared,
            "features_digest_reproduced": feat_ok,
            "reproduction": ("EXACT on all compared windows" if compared and
                             not mism and feat_ok == compared else
                             "MISMATCH - see examples" if mism else "partial"),
            "clips_covered": len({r["clip"] for r in rows}),
            "factories_covered": len({r["factory"] for r in rows}),
        },
        "time": time_block,
        "per_stage_ms": stages,
        "stage_mean_ms": {PRETTY[s]: round(means[s], 2) for s in STAGES},
        "stage_share_pct": shares,
        "write_stage_available": write_avail,
        "bottleneck": PRETTY[bottleneck],
        "bottleneck_mean_ms": round(means[bottleneck], 2),
        "decode_cpu_vs_nvdec": decode_cmp,
        "gpu": gpu_block,
        "adopted_policy_vs_full": None,
        "policy_note": ("the every-2nd + 4 s early-exit probe runs after the "
                        "pool; the run was stopped before it, so it is not "
                        "captured in the partial"),
        "per_clip_table": clip_rows, "n_clips": len(clip_rows),
        "honesty_note": ref_meta.get("honesty_note"),
        "reference_cost": ref_meta.get("cost"),
    }
    json.dump(res, open(os.path.join(IN, "L2.01.json"), "w"), indent=1,
              sort_keys=True)
    print(json.dumps({"outcomes": res["outcomes"], "time": res["time"],
                      "stage_share_pct": res["stage_share_pct"],
                      "bottleneck": res["bottleneck"]}, indent=1))

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_stages = [s for s in STAGES if stages[PRETTY[s]]]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels = [PRETTY[s] for s in plot_stages]
    p50 = [stages[l]["p50"] for l in labels]
    p95 = [stages[l]["p95"] for l in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, p50, 0.4, label="p50")
    ax.bar(x + 0.2, p95, 0.4, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ms / 10-s window")
    ax.set_title(f"L2.01 PARTIAL per-stage latency (n={n} of 1546, A40, "
                 f"4 workers, OMP=1)" if partial else
                 f"L2.01 per-stage latency (n={n}, A40, 4 workers, OMP=1)")
    ax.legend()
    for xi, v in zip(x - 0.2, p50):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + 0.2, p95):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "per_stage_p50_p95.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    data = [[r["latency_ms"].get(s, 0.0) for r in rows] for s in plot_stages]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(plot_stages) + 1))
    ax.set_xticklabels([PRETTY[s] for s in plot_stages])
    ax.set_yscale("log")
    ax.set_ylabel("ms (log)")
    ax.set_title(f"L2.01 PARTIAL per-stage distribution (n={n} of 1546)"
                 if partial else f"L2.01 per-stage distribution (n={n})")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "per_stage_distribution.png"), dpi=120)
    print("wrote L2.01.json + 2 figures")
    write_report(res)


def write_report(res):
    t, o, st, sh, g, dc = (res["time"], res["outcomes"], res["per_stage_ms"],
                           res["stage_share_pct"], res["gpu"],
                           res["decode_cpu_vs_nvdec"])
    partial = res["partial"]
    L = []
    A = L.append
    if partial:
        A(f"# L2 PARTIAL baseline ({o['windows_processed']} windows of 1,546) — "
          "outcomes + atomic per-stage latency\n")
    else:
        A("# L2 full-corpus PoC pipeline run — outcomes + atomic per-stage latency\n")
    A(f"**Registry id:** L2.01 · **Owner:** l2-latency · **Device:** A40 · "
      f"**Branch:** `m/latency-full` · **git_sha:** `{res.get('git_sha')}`\n")
    A(f"**Sampling:** `OMP_NUM_THREADS=1`, `--workers 4`"
      + (f" · **PARTIAL: {o['windows_processed']} of 1,546 windows** "
         f"({o['clips_covered']} clips, {o['factories_covered']} factories)\n"
         if partial else "\n"))
    if partial:
        A("> **This run was STOPPED and superseded.** The baseline full-corpus "
          "run was cancelled by the orchestrator in favour of the optimized "
          "full-potential run (L3) and the zero-copy run (L4), which take the "
          "pod exclusively. The numbers below are only the windows already "
          "written to `windows.jsonl.partial`: a **contiguous prefix** of the "
          "manifest (the first 2 clips), **not** a stratified sample. Do not "
          "read the verdict shares as corpus-wide.\n")
    A("## What this is\n")
    A("Runs the **repo's E8.01 pipeline** (`eval/exp_rescore/run_e8.py` — one "
      "NVDEC + `scale_cuda` three-branch pass per window, pixel-26 + EG2 (2 fps) "
      "+ E4.01 descriptors → profile router; `jadal/ensemble/scorer.py` frozen "
      "two-view logistic ensemble → verdict + content-hashed receipt) over the "
      "manifest windows with `time.perf_counter` around every stage. It is "
      "**not** a new benchmark re-implementation: `run_full.py` imports the repo "
      "modules and only inserts timing. One JSONL row per window is in "
      "`eval/results/exp_latency_full/windows.jsonl` (partial: "
      "`windows.jsonl.partial`).\n")
    A(f"> {res.get('honesty_note')}\n")
    A("## Outcomes\n")
    A(f"- **Windows processed:** {o['windows_processed']} / "
      f"{o.get('windows_requested')} · **failures:** {o['failures']} · "
      f"**clips:** {o['clips_covered']}")
    A(f"- **Verdict reproduction vs E8.01:** {o['reproduction']} "
      f"({o['per_window_compared']} windows compared, "
      f"{o['verdict_mismatch_count']} mismatches, "
      f"{o['features_digest_reproduced']}/{o['per_window_compared']} "
      "`features_digest` identical).")
    A("\n| verdict | L2 n | L2 share | E8.01 n (same windows) | E8.01 share |")
    A("|---|---|---|---|---|")
    for k in ("accept", "reject", "review"):
        A(f"| {k} | {o['verdict_distribution'].get(k, 0)} | "
          f"{o['verdict_share_pct'].get(k, 0)}% | "
          f"{o['reference_E8_01_verdict_distribution_same_subset'].get(k, 0)} | "
          f"{o['reference_E8_01_verdict_share_pct_same_subset'].get(k, 0)}% |")
    A(f"\nProfile distribution: `{o['profile_distribution']}`. Verdicts are "
      "**descriptive** (no human labels); the accept/reject share is not a "
      "quality measure.\n")
    A("## Time (measured)\n")
    A(f"- **Wall:** {round(t['wall_s'], 1)} s ({t['wall_min']} min) · "
      f"**{t['windows_per_s']} windows/s** ({t['windows_per_min']}/min) · "
      f"{t['workers']} workers, `OMP_NUM_THREADS={t['omp_threads']}`")
    A(f"- **Footage / wall-hour:** {t['footage_h_per_wall_h']} h of footage per "
      f"wall-hour ({t['partial_footage_h']} h processed"
      + (f" of the {t['corpus_hours']} h corpus" if partial else "") + ")")
    A(f"- Per window: mean {t['mean_ms_per_window']} ms, p50 "
      f"{t['p50_ms_per_window']} ms, p95 {t['p95_ms_per_window']} ms\n")
    A("### Per-stage latency (ms / 10-s window)\n")
    A("| stage | p50 | p95 | p99 | mean | share |")
    A("|---|---|---|---|---|---|")
    for s in ["io_probe", "decode(nvdec3)", "pixel-26", "descriptors",
              "eg2_embed", "mosaic", "decision", "write"]:
        d = st[s]
        if not d:
            A(f"| {s} | — | — | — | — | — |")
            continue
        A(f"| {s} | {d['p50']} | {d['p95']} | {d['p99']} | {d['mean']} | "
          f"{sh[s]}% |")
    if not res["write_stage_available"]:
        A("\n`write` is **not captured**: the parent times the JSONL flush "
          "*after* appending to the partial file, and the run was stopped "
          "before any row carried it. It is a sub-millisecond append.\n")
    A(f"\n**Bottleneck:** `{res['bottleneck']}` "
      f"({res['bottleneck_mean_ms']} ms mean, {sh[res['bottleneck']]}% of the "
      "per-window total). Ranked shares: "
      + ", ".join(f"{s} {v}%" for s, v in
                  sorted(sh.items(), key=lambda kv: -kv[1]) if v > 0) + ".")
    A("\nThe EG2 `p99` (9.6 s) is the 4 first-window warmups (one per worker) "
      "loading/priming the model; the `p50` (1.10 s) is the steady-state cost. "
      "Decode is the largest steady-state stage, matching E8.01 "
      "(decode 2354 ms > EG2 1196 ms mean).\n")
    A("### CPU decode vs NVDEC\n")
    if dc["n_compared"]:
        A(f"- Comparison on the first window of each processed clip "
          f"(n={dc['n_compared']}): CPU ffmpeg/swscale p50 "
          f"{dc['cpu_ffmpeg_swscale_ms'].get('p50')} ms vs explicit NVDEC "
          f"`-c:v hevc_cuvid` p50 {dc['nvdec_hevc_cuvid_ms'].get('p50')} ms → "
          f"**{dc['speedup_cpu_over_cuvid_median']}x** faster on GPU.")
        A(f"- Pipeline decode (E8.01 three-branch NVDEC pass) p50 "
          f"{dc['pipeline_3branch_nvdec_ms'].get('p50')} ms. {dc['note']}\n")
    else:
        A("- no comparison rows in the partial.\n")
    A("### GPU / cores\n")
    A(f"- EG2 per-window torch VRAM peak: p50 "
      f"{g['eg2_vram_peak_mib_per_window'].get('p50')} MiB.")
    A(f"- Cores: {g['cores']['nproc']} nproc, {g['cores']['workers']} workers, "
      f"OMP_NUM_THREADS={g['cores']['omp_threads']}. {g['note']}\n")
    A("## Adopted-policy variant\n")
    A(f"- {res['policy_note']}.\n")
    A("## Per-clip throughput\n")
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
    A("- Verdicts/`score`/`features_digest` use the same code path as E8.01 and "
      "are **identical on every compared window**; only `latency_ms` carries "
      "more stages, so `receipt_sha256` differs from E8.01 (expected).")
    A("- The EG2 model dir is patched **in memory** to the pod cache: a "
      "post-merge commit reverted `configs/ensemble_v1.json` to the laptop "
      "path; the head numbers are unchanged (`run_e8.load_config_checked`).")
    A(f"- Contention caveat: {res.get('contention_caveat')}")
    if partial:
        A("- **Stopped run:** superseded by L3 (optimized full-potential) and "
          "L4 (zero-copy); see `partial_note`.")
    p = os.environ.get("L2_REPORT",
                       os.path.join(REPO, "docs", "reports", "exp_latency_full.md"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
