#!/usr/bin/env python3
"""Generate docs/reports/exp_acc_signals.md from A2.01/02/03 JSONs + cost logs."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "eval", "results", "exp_acc")
INP = os.path.join(RES, "inputs")
OUT = os.path.join(ROOT, "docs", "reports", "exp_acc_signals.md")

s1 = json.load(open(os.path.join(RES, "A2.01.json")))
s2 = json.load(open(os.path.join(RES, "A2.02.json")))
s3 = json.load(open(os.path.join(RES, "A2.03.json")))
cg = json.load(open(os.path.join(INP, "signals_cost_gold.json")))
co = json.load(open(os.path.join(INP, "signals_cost_ood.json")))


def f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"


L = []
L.append("# A2 — New full-resolution signals: dense flow, hand segmentation, full-res per-pixel\n")
L.append("**Pod:** jadal-qa · **Branch:** `m/acc-signals` · **PR:** (link pending) base `main`\n")
L.append("**Question.** With compute free, extract genuinely new signals at full resolution / "
         "all frames and measure the accuracy they add, alone and on top of the "
         "26-feature + EG2 ensemble.\n")
L.append("**Device:** CPU for extraction (ffmpeg decode, OpenCV Farneback flow, MediaPipe "
         "HandLandmarker) and CPU numpy/sklearn for scoring; no GPU used. "
         "**Git sha:** `%s`.\n" % s1.get("git_sha", "?"))
L.append("**Results:** `eval/results/exp_acc/A2.01.json` (alone), `A2.02.json` (incremental), "
         "`A2.03.json` (combined). **Figure:** `docs/experiments/exp_acc/images/a2_signals.png`.\n")

# data / protocol
L.append("## Data & protocol\n")
L.append("| set | n | good | bad | frames/window | protocol |")
L.append("|---|---|---|---|---|---|")
L.append(f"| IndEgo gold | 150 | 100 | 50 | 100 @480x480/10fps | leave-one-clip-out (7 clips) |")
L.append(f"| OOD-160 Egocentric-10K | 160 | 138 | 22 | 300 @854x480/30fps | frozen ensemble; factory-grouped CI |")
L.append("| hold-out-108 (round 3) | 108 | 92 | 16 | — | *not run*: the 480p transcodes "
         "(`label_windows3/`) are absent from this bundle; labels only |")
L.append("")

# signals
L.append("## 1. Signals extracted (full resolution, all frames)\n")
L.append("| group | features | method |")
L.append("|---|---|---|")
L.append("| flow | 13 | OpenCV Farneback dense flow on **every consecutive frame** of the "
         "full-res decode, downscaled to 256p for the flow solve; magnitude / divergence / "
         "curl statistics, active-area fraction, stability, static & peak fractions |")
L.append("| hand | 5 | MediaPipe HandLandmarker (Tasks, CPU) on ~60 sampled frames/window "
         "(every 2nd gold / 6th OOD frame): hand-present fraction, hand-motion fraction, "
         "hand occupancy, mean #hands, hand-region flow excess |")
L.append("| fullres | 6 | native 480x480 / 854x480 per-pixel temporal statistics: Laplacian "
         "sharpness (mean/p10/temporal CV), noise floor (MAD of median residual, mean/p90), "
         "high-frequency band energy fraction (256x256 centre FFT) |")
L.append("")
L.append("Compute cost (single-thread ms/window, median):\n")
L.append("| domain | decode | flow | hand | fullres | **total** |")
L.append("|---|---|---|---|---|---|")
for tag, c in (("gold", cg), ("ood", co)):
    L.append(f"| {tag} | {c['decode_ms_median']} | {c['flow_ms_median']} | "
             f"{c['hand_ms_median']} | {c['fullres_ms_median']} | **{c['ms_per_window_median']}** |")
L.append(f"\nExtraction wall time: gold {cg['wall_s']}s (6 procs), OOD {co['wall_s']}s (6 procs); "
         f"errors: gold {cg['errors']}, OOD {co['errors']}.\n")

# A2.01
L.append("## 2. Each signal alone (A2.01)\n")
L.append("Bad-score AUROC (higher feature = more likely bad; sign fit on gold then applied to OOD).")
L.append("")
L.append("| signal | group | gold AUROC | OOD AUROC [grouped 95% CI] |")
L.append("|---|---|---|---|")
rows = sorted(s1["signals"].items(), key=lambda kv: -(kv[1]["gold_auroc"] + kv[1]["ood_auroc"]))
for n, r in rows:
    ci = r["ood_auroc_ci95_grouped"]
    L.append(f"| `{n}` | {r['group']} | {f(r['gold_auroc'])} | {f(r['ood_auroc'])} "
             f"[{f(ci[0])}, {f(ci[1])}] |")
b = s1["baseline_ensemble"]
L.append(f"\nBaseline 26-feature + EG2 ensemble: gold LOCO {f(b['gold_loco_auroc'])}, "
         f"OOD {f(b['ood_auroc'])} "
         f"[{f(b['ood_auroc_ci95_grouped'][0])}, {f(b['ood_auroc_ci95_grouped'][1])}].\n")

# A2.02
L.append("## 3. Incremental value on top of the ensemble (A2.02)\n")
L.append(f"Baseline ensemble: gold {f(s2['baseline']['gold_auroc'])}, OOD {f(s2['baseline']['ood_auroc'])}.")
L.append("")
L.append("| model | gold AUROC | Δ gold (paired) | OOD AUROC [grouped CI] | Δ OOD (paired) |")
L.append("|---|---|---|---|---|")
for name, r in s2["models"].items():
    if "note" in r:
        L.append(f"| {name} | {f(r['gold_auroc'])} | — | — | — |")
        continue
    gg = r["gain_vs_baseline_gold_paired"]
    og = r["gain_vs_baseline_ood_paired"]
    ci = r["ood_auroc_ci95_grouped"]
    L.append(f"| {name} | {f(r['gold_auroc'])} | {f(gg.get('mean'))} "
             f"[{f(gg.get('ci95',[None,None])[0])},{f(gg.get('ci95',[None,None])[1])}] | "
             f"{f(r['ood_auroc'])} [{f(ci[0])},{f(ci[1])}] | {f(og.get('mean'))} "
             f"[{f(og.get('ci95',[None,None])[0])},{f(og.get('ci95',[None,None])[1])}] |")
L.append("")

# A2.03
L.append("## 4. Best combined set (A2.03)\n")
L.append("| combination | gold AUROC | OOD AUROC [grouped CI] | mean-domain | Δ OOD (paired) |")
L.append("|---|---|---|---|---|")
for name, r in s3["candidates"].items():
    ci = r["ood_auroc_ci95_grouped"]
    og = r.get("gain_vs_baseline_ood_paired", {})
    L.append(f"| {name} | {f(r['gold_auroc'])} | {f(r['ood_auroc'])} [{f(ci[0])},{f(ci[1])}] | "
             f"{f(s3['mean_domain_auroc'][name])} | {f(og.get('mean'))} |")
L.append("")

# gap to 95
best = max(s3["candidates"].items(), key=lambda kv: s3["mean_domain_auroc"][kv[0]])
L.append("## 5. Gap to 95%\n")
L.append(f"- Best combined configuration: **{best[0]}** — gold {f(best[1]['gold_auroc'])}, "
         f"OOD {f(best[1]['ood_auroc'])}.")
L.append(f"- Gap to 95% AUROC: in-domain **{95 - best[1]['gold_auroc']*100:.1f} pts**, "
         f"OOD **{95 - best[1]['ood_auroc']*100:.1f} pts**.")
L.append("")

L.append("## Verdict\n")
L.append("(filled from the run)\n")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT)
