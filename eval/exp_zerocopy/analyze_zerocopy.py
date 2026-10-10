#!/usr/bin/env python3
"""Z1 analysis: mosaic delta + zero-copy attempt + decode verdict.

Reads eval/results/exp_zerocopy/{mosaic,zerocopy,decode_verdict,sample}.json,
writes Z1.01/Z1.02/Z1.03 JSONs, 3 figures and docs/reports/exp_zerocopy.md.

Regen: python3 eval/exp_zerocopy/analyze_zerocopy.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
IN = os.environ.get("Z_IN", os.path.join(REPO, "eval", "results", "exp_zerocopy"))
IMG = os.environ.get("Z_IMG", os.path.join(REPO, "docs", "experiments",
                                           "exp_zerocopy", "images"))
os.makedirs(IMG, exist_ok=True)

HONESTY = ("Mosaic is an evidence artefact, not a model input. Zero-copy: the "
           "repo decode is an out-of-process ffmpeg pipe (host memory only), so "
           "true VRAM-residency needs an in-process NVDEC binding; wherever that "
           "was not possible it is reported, not faked. All numbers carry n.")


def pct(a):
    a = np.asarray(a, float)
    if not len(a):
        return {}
    return {"n": int(len(a)), "mean": round(float(a.mean()), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2)}


def _load(name):
    p = os.path.join(IN, name)
    return json.load(open(p)) if os.path.exists(p) else None


def analyze_mosaic():
    d = _load("mosaic.json")
    if not d:
        return None
    on_total, off_total, mos = [], [], []
    disagree, digest_mismatch, score_mismatch = 0, 0, 0
    for r in d["rows"]:
        on, off = r["on"], r["off"]
        on_total.append(on["latency_ms"]["total"])
        off_total.append(off["latency_ms"]["total"])
        mos.append(on["latency_ms"]["mosaic"])
        if on["verdict"] != off["verdict"]:
            disagree += 1
        if on["features_digest"] != off["features_digest"]:
            digest_mismatch += 1
        if abs(on["score"] - off["score"]) > 1e-9:
            score_mismatch += 1
    on_t = np.array(on_total); off_t = np.array(off_total); m = np.array(mos)
    delta = on_t - off_t
    return {
        "id": "Z1.01", "track": "zerocopy", "device": d.get("device"),
        "title": "Mosaic stage ON vs OFF: wall-time delta + verdict identity",
        "n": d["n"],
        "mosaic_ms": pct(m),
        "total_on_ms": pct(on_t), "total_off_ms": pct(off_t),
        "delta_ms": pct(delta),
        "delta_ms_std": round(float(delta.std()), 2),
        "delta_stage_ms": pct(m),
        "delta_note": ("the mosaic build is the ONLY difference between the two "
                       "passes, so the clean wall delta is the mosaic stage time; "
                       "the two independent passes also give a noisier cross-check "
                       "delta (run-to-run variance / pod contention)"),
        "delta_pct_of_on": round(float(100.0 * delta.mean() / max(on_t.mean(), 1e-9)), 2),
        "verdict_disagreements": disagree,
        "features_digest_mismatches": digest_mismatch,
        "score_mismatches": score_mismatch,
        "verdicts_identical": disagree == 0 and digest_mismatch == 0 and score_mismatch == 0,
        "rows": d["rows"],
    }


def analyze_zerocopy():
    d = _load("zerocopy.json")
    if not d:
        return None
    stages = ("decode", "pixel_features", "descriptors", "eg2_embed", "total")
    base = {s: [] for s in stages}
    gpu = {s: [] for s in stages}
    cos, agree = [], 0
    fdelta = {}
    bytes_saved = []
    for r in d["rows"]:
        for s in stages:
            base[s].append(r["base"]["latency_ms"].get(s, 0.0))
            gpu[s].append(r["gpu"]["latency_ms"].get(s, 0.0))
        cos.append(r["eg2_cosine"])
        agree += 1 if r["verdict_agree"] else 0
        for k, v in r["feature_delta"].items():
            fdelta.setdefault(k, []).append(abs(v))
        bd = r["base"]["bytes"]; gd = r["gpu"]["bytes"]
        base_cross = bd["pix"] + bd["eg2"] + bd["native"] + bd["eg2"]
        gpu_cross = gd.get("download", 0) + gd.get("upload", 0)
        bytes_saved.append(base_cross - gpu_cross)
    out = {
        "id": "Z1.02", "track": "zerocopy", "device": d.get("device"),
        "title": "Zero-copy GPU path attempt (NVDEC->VRAM features/embed) vs hwdownload",
        "n": d["n"], "gpu_mode": d["gpu_mode"],
        "torchcodec_available": d.get("torchcodec_available"),
        "blockers": d.get("blockers", {}),
        "stages_base_ms": {s: pct(base[s]) for s in stages},
        "stages_gpu_ms": {s: pct(gpu[s]) for s in stages},
        "speedup": {s: round(float(np.mean(base[s]) / max(np.mean(gpu[s]), 1e-9)), 3)
                    for s in stages},
        "eg2_cosine": pct(cos),
        "verdict_agreement": {"n": d["n"], "agree": agree,
                              "rate": round(agree / max(d["n"], 1), 3)},
        "feature_abs_delta_max": {k: round(float(np.max(v)), 5)
                                  for k, v in sorted(fdelta.items())},
        "feature_abs_delta_mean": {k: round(float(np.mean(v)), 5)
                                   for k, v in sorted(fdelta.items())},
        "bytes_crossing_pcie_per_window": {
            "base_mean": int(np.mean([r["base"]["bytes"]["pix"] +
                                      r["base"]["bytes"]["eg2"] +
                                      r["base"]["bytes"]["native"] +
                                      r["base"]["bytes"]["eg2"]
                                      for r in d["rows"]])),
            "gpu_mean": int(np.mean([r["gpu"]["bytes"].get("download", 0) +
                                     r["gpu"]["bytes"].get("upload", 0)
                                     for r in d["rows"]])),
            "saved_mean": int(np.mean(bytes_saved)),
        },
        "rows": d["rows"],
    }
    return out


def analyze_decode():
    d = _load("decode_verdict.json")
    if not d:
        return None
    return {
        "id": "Z1.03", "track": "zerocopy", "device": "A40",
        "title": "NVDEC vs CPU decode at 1080p30 HEVC and 480p H.264, with/without hwdownload",
        "n_1080p_windows": len(d["windows"]),
        "aggregate": d["aggregate"],
    }


def analyze_upload():
    d = _load("upload_specs.json")
    if not d:
        return None
    raw_by_tag = {s["tag"]: s for s in d.get("specs", [])}

    def _min(vals):
        vals = [v for v in vals if v is not None]
        return round(float(min(vals)), 1) if vals else None

    specs = []
    for s in d["specs"]:
        rr = raw_by_tag.get(s["tag"], {}).get("rows", [])
        nv = s["nvdec_hwdownload"]["p50_ms"] if s["nvdec_hwdownload"] else None
        cp = s["cpu"]["p50_ms"] if s["cpu"] else None
        dec_best = min([v for v in (nv, cp) if v is not None], default=None)
        ref = d.get("reference_stages") or {}
        fg = ref.get("feature_gpu_ms"); eg = ref.get("eg2_gpu_ms")
        total_best = (round(dec_best + fg + eg, 1)
                      if dec_best is not None and fg is not None and eg is not None else None)
        specs.append({
            "tag": s["tag"], "res": f"{s['w']}x{s['h']}@{s['fps']}",
            "codec": s["codec"], "crf": s["crf"], "n": s["n"],
            "size_bytes_mean": s["size_bytes"]["mean_ms"] if s["size_bytes"] else None,
            "bytes_per_frame_mean": s["bytes_per_frame"]["mean_ms"] if s["bytes_per_frame"] else None,
            "bitrate_kbps_mean": s["bitrate_kbps"]["mean_ms"] if s["bitrate_kbps"] else None,
            "decode_nvdec_hwdownload_p50": nv,
            "decode_nvdec_hwdownload_min": _min([r.get("nvdec_hwdownload_ms") for r in rr]),
            "decode_nvdec_no_download_p50": s["nvdec_no_download"]["p50_ms"] if s["nvdec_no_download"] else None,
            "decode_nvdec_no_download_min": _min([r.get("nvdec_no_download_ms") for r in rr]),
            "decode_cpu_p50": cp,
            "decode_cpu_min": _min([r.get("cpu_ms") for r in rr]),
            "cpu_over_nvdec_p50": s["cpu_over_nvdec_p50"],
            "decode_best_p50": dec_best,
            "decode_best_path": ("cpu" if (cp is not None and nv is not None and cp <= nv)
                                 else "nvdec" if nv is not None else None),
            "hwdownload_cost_p50_ms": s["hwdownload_cost_p50_ms"],
            "total_base_ms": s.get("total_base_ms"),
            "total_gpu_ms": s.get("total_gpu_ms"),
            "total_best_ms": total_best,
            "footage_h_per_wall_h_base": s.get("footage_h_per_wall_h_base"),
            "footage_h_per_wall_h_gpu": s.get("footage_h_per_wall_h_gpu"),
            "footage_h_per_wall_h_best": (round(10000.0 / total_best, 3)
                                          if total_best else None),
        })
    # recommendation: best footage-h/wall-h among specs with a validated accuracy
    # budget. X1 validated 360p@CRF28 (~40% bytes, ΔAUROC < 0.02); 240p was NOT
    # validated. 480p and 360p decode within noise, so prefer the lowest-byte
    # validated spec.
    valid = [x for x in specs if x["tag"] in
             ("1080p30_hevc", "1080p_h264_crf23", "480p_h264_crf23",
              "360p_h264_crf28")]
    maxv = max(valid, key=lambda x: (x["footage_h_per_wall_h_best"] or 0)) if valid else None
    best = maxv
    if maxv and maxv["footage_h_per_wall_h_best"]:
        near = [x for x in valid if (x["footage_h_per_wall_h_best"] or 0)
                >= 0.95 * maxv["footage_h_per_wall_h_best"]]
        if near:
            best = min(near, key=lambda x: x["bytes_per_frame_mean"] or 1e18)
    corpus = None
    if best and best["total_best_ms"]:
        n_w = 1546
        hours = 4.404
        wall1 = n_w * best["total_best_ms"] / 1000.0
        corpus = {
            "spec": best["tag"], "decode_path": best["decode_best_path"],
            "corpus_windows": n_w, "corpus_hours": hours,
            "total_best_ms_per_window": best["total_best_ms"],
            "wall_s_1worker": round(wall1, 1),
            "wall_min_1worker": round(wall1 / 60.0, 1),
            "footage_h_per_wall_h_1worker": round(hours / (wall1 / 3600.0), 3),
            "wall_min_4worker_ideal": round(wall1 / 4.0 / 60.0, 1),
            "footage_h_per_wall_h_4worker_ideal": round(hours / (wall1 / 4.0 / 3600.0), 3),
            "note": ("best combo = spec x faster decode path x GPU feature/embed; "
                     "ideal linear worker scaling (real scaling is GPU/EG2-bound, "
                     "measured at 4 workers in L2.01)"),
        }
    return {
        "id": "Z1.04", "track": "zerocopy", "device": d.get("device"),
        "title": "Upload-spec ladder: codec/resolution/CRF vs decode + zero-copy throughput",
        "n": d["sample_n"], "cpu_n": d["cpu_n"], "encoder": d.get("encoder"),
        "reference_stages": d.get("reference_stages"),
        "specs": specs,
        "recommended_spec": best["tag"] if best else None,
        "recommended_decode_path": best["decode_best_path"] if best else None,
        "corpus_extrapolation": corpus,
        "note": ("feature/EG2 costs are resolution-invariant (160x90 / 512x288); "
                 "decode is the spec-dependent stage; totals compose the measured "
                 "reference stages; zero-copy is the GPU-compute path (true "
                 "VRAM-residency is blocked, see Z1.02)"),
        "raw": d,
    }


def figures(mosaic, zc, dec, up):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mosaic:
        rows = mosaic["rows"]
        stages = ["io_probe", "decode", "pixel_features", "descriptors",
                  "eg2_embed", "mosaic", "decision"]
        on = [np.mean([r["on"]["latency_ms"].get(s, 0) for r in rows]) for s in stages]
        off = [np.mean([r["off"]["latency_ms"].get(s, 0) for r in rows]) for s in stages]
        x = np.arange(len(stages))
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(x - 0.2, on, 0.4, label="mosaic ON")
        ax.bar(x + 0.2, off, 0.4, label="mosaic OFF")
        ax.set_xticks(x); ax.set_xticklabels(stages, rotation=20, ha="right")
        ax.set_ylabel("mean ms / 10-s window")
        ax.set_title(f"Z1.01 mosaic ON vs OFF (n={mosaic['n']}, A40)")
        ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(IMG, "mosaic_on_off.png"), dpi=120)
        plt.close(fig)

    if zc:
        stages = ["decode", "pixel_features", "descriptors", "eg2_embed", "total"]
        b = [zc["stages_base_ms"][s].get("mean", 0) for s in stages]
        g = [zc["stages_gpu_ms"][s].get("mean", 0) for s in stages]
        x = np.arange(len(stages))
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(x - 0.2, b, 0.4, label="hwdownload (repo)")
        ax.bar(x + 0.2, g, 0.4, label=f"GPU path ({zc['gpu_mode']})")
        ax.set_xticks(x); ax.set_xticklabels(stages, rotation=20, ha="right")
        ax.set_ylabel("mean ms / window")
        ax.set_title(f"Z1.02 per-stage: hwdownload vs GPU path (n={zc['n']})")
        ax.legend()
        for xi, v in zip(x - 0.2, b):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        for xi, v in zip(x + 0.2, g):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(IMG, "zerocopy_stages.png"), dpi=120)
        plt.close(fig)

    if dec:
        keys = list(dec["aggregate"].keys())
        cpu = [dec["aggregate"][k]["cpu"]["p50_ms"] if dec["aggregate"][k]["cpu"] else 0 for k in keys]
        dl = [dec["aggregate"][k]["nvdec_hwdownload"]["p50_ms"] if dec["aggregate"][k]["nvdec_hwdownload"] else 0 for k in keys]
        nd = [dec["aggregate"][k]["nvdec_no_download"]["p50_ms"] if dec["aggregate"][k]["nvdec_no_download"] else 0 for k in keys]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.bar(x - 0.25, cpu, 0.25, label="CPU ffmpeg")
        ax.bar(x, dl, 0.25, label="NVDEC + hwdownload")
        ax.bar(x + 0.25, nd, 0.25, label="NVDEC, no download (null)")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right")
        ax.set_ylabel("p50 ms / 10-s window")
        ax.set_title("Z1.03 decode: CPU vs NVDEC with/without hwdownload")
        ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(IMG, "decode_verdict.png"), dpi=120)
        plt.close(fig)

    if up:
        specs = up["specs"]
        tags = [s["tag"] for s in specs]
        dl = [s["decode_nvdec_hwdownload_p50"] or 0 for s in specs]
        nd = [s["decode_nvdec_no_download_p50"] or 0 for s in specs]
        x = np.arange(len(tags))
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.bar(x - 0.2, dl, 0.4, label="NVDEC + hwdownload")
        ax.bar(x + 0.2, nd, 0.4, label="NVDEC no download")
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right")
        ax.set_ylabel("p50 decode ms / 10-s window")
        ax.set_title(f"Z1.04 upload-spec decode ladder (n={up['n']})")
        ax.legend()
        ax2 = ax.twinx()
        fh = [s["footage_h_per_wall_h_best"] or 0 for s in specs]
        ax2.plot(x, fh, "k.--", label="footage-h/wall-h (best combo)")
        ax2.set_ylabel("footage-h / wall-h (1 stream)")
        ax2.legend(loc="upper left")
        fig.tight_layout(); fig.savefig(os.path.join(IMG, "upload_specs.png"), dpi=120)
        plt.close(fig)


def write_report(mosaic, zc, dec, up):
    L = []
    A = L.append
    A("# Z1 — mosaic delta + zero-copy GPU path + decode verdict\n")
    A("**Device:** A40 · **Branch:** `m/zerocopy` · "
      "**Runner:** `eval/exp_zerocopy/run_zerocopy.py`\n")
    A(f"> {HONESTY}\n")
    A("## Z1.01 Mosaic ON vs OFF\n")
    if mosaic:
        A(f"Same 30-window sample (spread over {len({r['on']['factory'] for r in mosaic['rows']})} "
          f"factories), full repo pipeline run twice; only the mosaic build differs.\n")
        A(f"- Mosaic stage (the only difference between the two passes): mean "
          f"{mosaic['mosaic_ms']['mean']} ms, p50 {mosaic['mosaic_ms']['p50']} ms per window "
          f"→ **clean wall delta {mosaic['delta_stage_ms']['mean']} ms/window**.")
        A(f"- Two independent passes cross-check: ON mean {mosaic['total_on_ms']['mean']} ms vs "
          f"OFF mean {mosaic['total_off_ms']['mean']} ms → delta "
          f"{mosaic['delta_ms']['mean']} ± {mosaic['delta_ms_std']} ms "
          f"(noisy: run-to-run variance / pod contention; the mosaic build is the "
          f"only code difference).")
        A(f"- Verdicts identical: **{mosaic['verdicts_identical']}** "
          f"(disagreements {mosaic['verdict_disagreements']}, "
          f"`features_digest` mismatches {mosaic['features_digest_mismatches']}, "
          f"score mismatches {mosaic['score_mismatches']}).")
        A("\n![mosaic](%s)\n" % "../experiments/exp_zerocopy/images/mosaic_on_off.png")
    A("## Z1.02 Zero-copy GPU path vs hwdownload\n")
    if zc:
        A(f"- Mode: `{zc['gpu_mode']}` (torchcodec available: "
          f"{zc['torchcodec_available']}).")
        if zc["blockers"]:
            A(f"- **Blockers:** `{json.dumps(zc['blockers'])[:600]}`")
        A("\n**Why true zero-copy is blocked (attempted, not faked):** the repo's "
          "decode is an out-of-process ffmpeg whose only output is a host-memory pipe "
          "(`-f rawvideo ... pipe:1`); ffmpeg has no muxer that writes CUDA device "
          "memory, so frames cannot stay VRAM-resident across that boundary. The only "
          "in-process NVDEC binding that installs here, `torchcodec`, was tried at "
          "0.6/0.8/0.10/0.12/0.17: 0.6 imports but rejects `device='cuda'` "
          "(`Unsupported device: cuda`); 0.10/0.12 fail to load (bundled FFmpeg "
          "soname); 0.17 needs a CUDA-13 runtime (`libnvrtc.so.13`, `libcudart.so.13`) "
          "that is not present (driver 580 is CUDA 13 but only the CUDA-12.8 toolkit "
          "is installed, and the PyPI `*-cu13` packages are empty placeholders). "
          "`PyNvVideoCodec`, `cv2.cudacodec`, `decord`, `av` and `cupy` are all "
          "absent. The measured GPU path is therefore the **partial** upload path "
          "(hwdownload, then one small upload of the pix+EG2 frames); the compute "
          "stages are genuinely device-resident, the decode hop is not.\n")
        A("\n| stage | hwdownload mean ms | GPU mean ms | speedup |")
        A("|---|---|---|---|")
        for s in ("decode", "pixel_features", "descriptors", "eg2_embed", "total"):
            A(f"| {s} | {zc['stages_base_ms'][s].get('mean')} | "
              f"{zc['stages_gpu_ms'][s].get('mean')} | {zc['speedup'][s]}x |")
        A(f"\n- EG2 embedding cosine vs the repo `encode()` path: "
          f"mean {zc['eg2_cosine'].get('mean')} (p50 {zc['eg2_cosine'].get('p50')}).")
        A(f"- Verdict agreement: {zc['verdict_agreement']['agree']}/"
          f"{zc['verdict_agreement']['n']} "
          f"({zc['verdict_agreement']['rate']}).")
        bc = zc["bytes_crossing_pcie_per_window"]
        A(f"- PCIe bytes/window: base {bc['base_mean']} vs GPU {bc['gpu_mean']} "
          f"→ **saved {bc['saved_mean']} B/window** "
          f"({bc['saved_mean'] / 1e6:.2f} MB). The partial path is *negative* here: it "
          f"still hwdownloads (same as the repo) and then re-uploads the pix+EG2 "
          f"frames, so it adds ~13 MB/window. Only true in-process zero-copy would "
          f"save the ~80 MB/window round trip (pix+EG2+native download + EG2 upload).")
        worst = sorted(zc["feature_abs_delta_max"].items(),
                       key=lambda kv: -kv[1])[:6]
        A(f"- Largest pixel-feature |Δ|: " +
          ", ".join(f"`{k}` {v}" for k, v in worst) + ".")
        A("\n![zerocopy](%s)\n" % "../experiments/exp_zerocopy/images/zerocopy_stages.png")
    A("## Z1.03 Decode verdict (NVDEC vs CPU)\n")
    if dec:
        A("| arm | CPU p50 ms | NVDEC+hwdownload p50 ms | NVDEC no-download p50 ms | hwdownload cost |")
        A("|---|---|---|---|---|")
        for k, r in dec["aggregate"].items():
            cpu = r["cpu"]["p50_ms"] if r["cpu"] else None
            dl = r["nvdec_hwdownload"]["p50_ms"] if r["nvdec_hwdownload"] else None
            nd = r["nvdec_no_download"]["p50_ms"] if r["nvdec_no_download"] else None
            A(f"| {k} | {cpu} | {dl} | {nd} | {r.get('hwdownload_cost_p50_ms')} |")
        A("\n![decode](%s)\n" % "../experiments/exp_zerocopy/images/decode_verdict.png")
    A("## Upload-compressed input (deployment may choose the upload spec)\n")
    if up:
        A(f"If upload-side encoding is allowed, the codec/resolution/quality is a "
          f"free variable. Ladder on the same {up['n']}-window sample "
          f"(feature/EG2 costs are resolution-invariant: the pipeline always works "
          f"at 160x90 / 512x288, so only decode is spec-dependent). Upload segments "
          f"were transcoded with **NVENC** (`-cq`, ~CRF) to stay off the contended "
          f"pod CPU; the reference stages are from the 30-window zero-copy run.\n")
        A("| spec | bitrate kbps | B/frame | decode NVDEC p50 ms | decode NVDEC min ms | "
          "decode CPU p50 ms | best decode | total base ms | total GPU ms | total best ms | footage-h/wall-h (best) |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for s in up["specs"]:
            A(f"| {s['tag']} | {s['bitrate_kbps_mean']} | {s['bytes_per_frame_mean']} | "
              f"{s['decode_nvdec_hwdownload_p50']} | {s['decode_nvdec_hwdownload_min']} | "
              f"{s['decode_cpu_p50']} | {s['decode_best_path']} | {s['total_base_ms']} | "
              f"{s['total_gpu_ms']} | {s['total_best_ms']} | {s['footage_h_per_wall_h_best']} |")
        A(f"\n- **Recommended combo:** `{up['recommended_spec']}` upload × "
          f"**{up['recommended_decode_path']}** decode × GPU pixel-26 + GPU EG2. "
          f"X1 measured 360p@CRF28 ≈ 40% of the bytes with ΔAUROC < 0.02; here 360p "
          f"and 480p decode within noise, but at 360p the CPU decoder wins "
          f"({[s['decode_cpu_p50'] for s in up['specs'] if s['tag']=='360p_h264_crf28'][0]} ms vs NVDEC "
          f"{[s['decode_nvdec_hwdownload_p50'] for s in up['specs'] if s['tag']=='360p_h264_crf28'][0]} ms), so the best combo is "
          f"360p + CPU decode + GPU feature/embed. 240p was **not** accuracy-validated.")
        A(f"- **NVDEC-vs-CPU flip:** NVDEC wins at 1080p (HEVC {[s['decode_nvdec_hwdownload_p50'] for s in up['specs'] if s['tag']=='1080p30_hevc'][0]} ms vs CPU "
          f"{[s['decode_cpu_p50'] for s in up['specs'] if s['tag']=='1080p30_hevc'][0]} ms); at 480p it is a wash "
          f"(NVDEC {[s['decode_nvdec_hwdownload_p50'] for s in up['specs'] if s['tag']=='480p_h264_crf23'][0]} ms vs CPU {[s['decode_cpu_p50'] for s in up['specs'] if s['tag']=='480p_h264_crf23'][0]} ms); "
          f"at 360p CPU wins ({[s['decode_cpu_p50'] for s in up['specs'] if s['tag']=='360p_h264_crf28'][0]} ms vs NVDEC {[s['decode_nvdec_hwdownload_p50'] for s in up['specs'] if s['tag']=='360p_h264_crf28'][0]} ms). "
          f"So the flip point is between 480p and 360p. The 240p row is contended "
          f"(p50 inflated; min {[s['decode_nvdec_hwdownload_min'] for s in up['specs'] if s['tag']=='240p_h264_crf28'][0]} ms).")
        A(f"- **Zero-copy vs not:** `total base` = NVDEC+hwdownload decode + CPU pixel-26 "
          f"+ `encode()` EG2; `total GPU` = same decode + GPU pixel-26 "
          f"({up['reference_stages']['feature_gpu_ms']:.0f} ms) + GPU EG2 "
          f"({up['reference_stages']['eg2_gpu_ms']:.0f} ms). The GPU path is ~1.9× faster "
          f"and the gap is dominated by the pixel-26 stage (CPU "
          f"{up['reference_stages']['feature_cpu_ms']:.0f} ms → GPU "
          f"{up['reference_stages']['feature_gpu_ms']:.0f} ms). True zero-copy would "
          f"additionally drop the (here small, 160x90) hwdownload; the large native-frame "
          f"download is the descriptor branch, which also shrinks with the spec.")
        if up.get("corpus_extrapolation"):
            ce = up["corpus_extrapolation"]
            A(f"- **End-to-end extrapolation to the full corpus** "
              f"(1,546 windows / 4.404 h) for the recommended combo "
              f"`{ce['spec']}` × {ce.get('decode_path')} decode × GPU feature/embed "
              f"({ce['total_best_ms_per_window']} ms/window): "
              f"**{ce['wall_min_1worker']} min on 1 worker** "
              f"({ce['footage_h_per_wall_h_1worker']} footage-h/wall-h), "
              f"~{ce['wall_min_4worker_ideal']} min at 4 workers ideal "
              f"({ce['footage_h_per_wall_h_4worker_ideal']} footage-h/wall-h). "
              f"{ce['note']}.")
        A("\n![upload specs](%s)\n" % "../experiments/exp_zerocopy/images/upload_specs.png")
    A("## Reproduction\n")
    A("```bash\nsource /workspace/env.sh\nexport OMP_NUM_THREADS=2 JADAL_GPU_SLOTS=4\n"
      "bash tools/pod/gpu_run.sh python3 eval/exp_zerocopy/run_zerocopy.py --sample-n 30\n"
      "bash tools/pod/gpu_run.sh python3 eval/exp_zerocopy/run_upload_specs.py --sample-n 8 --reps 2 --encoder nvenc\n"
      "python3 eval/exp_zerocopy/analyze_zerocopy.py\n```\n")
    A("Verdicts are descriptive (the corpus is unlabelled). n=30 windows, one per "
      "factory; per-stage timings are wall-clock on a shared A40.\n")
    p = os.environ.get("Z_REPORT", os.path.join(REPO, "docs", "reports",
                                                "exp_zerocopy.md"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", p)


def main():
    mosaic = analyze_mosaic()
    zc = analyze_zerocopy()
    dec = analyze_decode()
    up = analyze_upload()
    if mosaic:
        json.dump(mosaic, open(os.path.join(IN, "Z1.01.json"), "w"), indent=1, sort_keys=True)
    if zc:
        json.dump(zc, open(os.path.join(IN, "Z1.02.json"), "w"), indent=1, sort_keys=True)
    if dec:
        json.dump(dec, open(os.path.join(IN, "Z1.03.json"), "w"), indent=1, sort_keys=True)
    if up:
        json.dump(up, open(os.path.join(IN, "Z1.04.json"), "w"), indent=1, sort_keys=True)
    figures(mosaic, zc, dec, up)
    write_report(mosaic, zc, dec, up)
    print(json.dumps({"mosaic": bool(mosaic), "zerocopy": bool(zc),
                      "decode": bool(dec), "upload": bool(up)}, indent=1))


if __name__ == "__main__":
    main()
