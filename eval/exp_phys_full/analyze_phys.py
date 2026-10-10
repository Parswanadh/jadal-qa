#!/usr/bin/env python3
"""Q5.01 — analyse the full-corpus deterministic-physics-rule run.

Inputs
  eval/results/exp_phys_full/windows.jsonl   (Q5 rows; one per window)
  eval/results/exp_phys_full/run_meta.json   (wall, opts, determinism probe)
  /workspace/artifacts/windows_rescored.jsonl (E8.01 full per-window verdicts)
  <m/latency-fast>/eval/results/exp_latency_fast/windows.jsonl (L3 per-window)
  <m/latency-full>/eval/results/exp_latency_full/L2.01.json    (L2 partial)

Outputs
  eval/results/exp_phys_full/Q5.01.json
  docs/reports/exp_phys_full.md
  docs/experiments/exp_phys_full/images/*.png

No GPU. Reads cached run outputs only; never touches labels.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "eval", "results", "exp_phys_full")
IMGDIR = os.path.join(REPO, "docs", "experiments", "exp_phys_full", "images")
REPORT = os.path.join(REPO, "docs", "reports", "exp_phys_full.md")

WINDOWS = os.path.join(OUT, "windows.jsonl")
RUN_META = os.path.join(OUT, "run_meta.json")
E8_WINDOWS = "/workspace/artifacts/windows_rescored.jsonl"
E8_JSON = "/workspace/wt/m-latency-fast/eval/results/exp_rescore/E8.01.json"
L3_WINDOWS = "/workspace/wt/m-latency-fast/eval/results/exp_latency_fast/windows.jsonl"
L3_META = "/workspace/wt/m-latency-fast/eval/results/exp_latency_fast/run_meta_stride1.json"
L2_JSON = "/workspace/wt/m-latency-full/eval/results/exp_latency_full/L2.01.json"

STAGES = ("io_probe", "decode", "pixel_features", "descriptors", "eg2_embed",
          "phys_features", "decision", "write", "total")
HONESTY = ("These 1,546 Egocentric-10K windows have NO human labels: label_source "
           "is 'none' and every accept/reject/review verdict here is DESCRIPTIVE, "
           "not accuracy. The Q1 rule thresholds were derived on 7 IndEgo gold "
           "clips; applying them out-of-domain is a determinism transfer test, not "
           "a quality measure.")


def load_rows(path):
    rows = []
    for ln in open(path):
        try:
            r = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if "row" in r:
            rows.append(r["row"])
        elif "verdict" in r:
            rows.append(r)
    return rows


def stats(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": round(float(v.mean()), 1),
            "p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1),
            "min": round(float(v.min()), 1), "max": round(float(v.max()), 1)}


def dist_pct(counter, n):
    return {k: round(100.0 * v / max(n, 1), 2) for k, v in sorted(counter.items())}


def load_e8_verdicts():
    m = {}
    if not os.path.exists(E8_WINDOWS):
        return m
    for ln in open(E8_WINDOWS):
        try:
            r = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if r.get("type") == "header" or "verdict" not in r:
            continue
        m[r["window_id"]] = r["verdict"]
    return m


def load_verdicts(path):
    m = {}
    if not os.path.exists(path):
        return m
    for r in load_rows(path):
        m[r["window_id"]] = r["verdict"]
    return m


def confusion(a, b, order=("accept", "reject", "review")):
    """a = reference, b = candidate; returns matrix counts."""
    idx = {k: i for i, k in enumerate(order)}
    M = np.zeros((len(order), len(order)), dtype=int)
    n = 0
    for k, v in a.items():
        if k not in b:
            continue
        M[idx[v], idx[b[k]]] += 1
        n += 1
    return M, n


def kappa(M):
    M = np.asarray(M, dtype=float)
    n = M.sum()
    if n == 0:
        return None
    po = np.trace(M) / n
    pe = float((M.sum(0) * M.sum(1)).sum()) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def main():
    rows = load_rows(WINDOWS)
    meta = json.load(open(RUN_META))
    n = len(rows)
    assert n > 0, "no rows"

    vc = Counter(r["verdict"] for r in rows)
    pc = Counter(r["profile"] for r in rows)
    # every-2nd adopted policy: rows already sorted by (clip,start_s,window_id)
    sub = rows[::2]
    vc2 = Counter(r["verdict"] for r in sub)

    # per-stage
    per_stage = {}
    for st in STAGES:
        per_stage[st] = stats([r["latency_ms"].get(st) for r in rows])
    tot_mean = per_stage["total"]["mean"]
    share = {st: round(100.0 * per_stage[st].get("mean", 0.0) / tot_mean, 2)
             for st in STAGES if st != "total"}
    share["total"] = 100.0

    # rule firing
    fire = Counter()
    fire_bad_unknown = Counter()
    for r in rows:
        for rule in r.get("fired_rules", []):
            fire[rule] += 1
    rule_fire = {k: {"fire": int(v), "share_pct": round(100.0 * v / n, 2)}
                 for k, v in sorted(fire.items(), key=lambda x: -x[1])}

    # score distribution
    scores = np.array([r["score"] for r in rows], dtype=int)
    score_hist = {str(int(b)): int((scores == b).sum())
                  for b in range(int(scores.min()), int(scores.max()) + 1)}

    # determinism
    det = meta.get("determinism_probe", [])
    det_ok = [d for d in det if "receipt_match" in d]
    determinism = {
        "n_probed": len(det),
        "all_receipts_match": bool(det_ok) and all(d["receipt_match"] for d in det_ok),
        "all_features_digests_match": bool(det_ok) and all(
            d["features_digest_match"] for d in det_ok),
        "probe": det,
    }

    # comparisons
    e8 = load_e8_verdicts()
    l3 = load_verdicts(L3_WINDOWS)
    mine = {r["window_id"]: r["verdict"] for r in rows}
    Mcmp, ncmp = confusion(e8, mine) if e8 else (np.zeros((3, 3), int), 0)
    M3, n3 = confusion(l3, mine) if l3 else (np.zeros((3, 3), int), 0)
    e8c = Counter(e8.values())
    l3c = Counter(l3.values())
    # binary: is the window "rejected" or not
    bin_agree = None
    if e8:
        a = np.array([1 if e8[k] == "reject" else 0 for k in e8 if k in mine])
        b = np.array([1 if mine[k] == "reject" else 0 for k in e8 if k in mine])
        bin_agree = {"n": int(a.size),
                     "reject_agreement_pct": round(100.0 * float((a == b).mean()), 2),
                     "both_reject": int(((a == 1) & (b == 1)).sum()),
                     "e8_reject_only": int(((a == 1) & (b == 0)).sum()),
                     "q5_reject_only": int(((a == 0) & (b == 1)).sum()),
                     "neither": int(((a == 0) & (b == 0)).sum())}

    l2 = json.load(open(L2_JSON)) if os.path.exists(L2_JSON) else {}
    l3meta = json.load(open(L3_META)) if os.path.exists(L3_META) else {}

    res = {
        "id": "Q5.01",
        "track": "phys",
        "title": "Deterministic physics rule-list model: full-corpus run + atomic latency",
        "protocol": "repo E8.01 pipeline (eval/exp_rescore/run_e8.py) decode/features/"
                    "descriptors/EG2 with perf_counter per stage; decision stage "
                    "swapped to Q1's deterministic physics rule list "
                    "(eval/exp_phys/phys_rules.py + rule_table.json); no learned "
                    "weights in the decision",
        "label_source": "none (unlabelled windows; verdicts DESCRIPTIVE)",
        "honesty_note": HONESTY,
        "run": {
            "workers": meta.get("workers"), "gpu_slots": 6, "device": meta.get("device"),
            "omp_threads": meta.get("omp_threads"),
            "optimizations": meta.get("optimizations"),
            "git_sha": meta.get("git_sha"),
            "rule_table_sha256": meta.get("rule_table_sha256"),
            "reject_threshold": meta.get("reject_threshold"),
            "window_stride": meta.get("window_stride"),
            "n_errors": meta.get("n_errors"),
            "errors": meta.get("errors", []),
        },
        "corpus": {"n_windows": n, "n_manifest_windows": meta.get("n_manifest_windows"),
                   "n_clips": meta["manifest"]["n_clips"],
                   "total_hours": meta["manifest"]["total_hours"]},
        "verdict_distribution": dict(sorted(vc.items())),
        "verdict_share_pct": dist_pct(vc, n),
        "every_2nd_policy": {"n": len(sub),
                             "verdict_distribution": dict(sorted(vc2.items())),
                             "verdict_share_pct": dist_pct(vc2, len(sub))},
        "profile_distribution": dict(sorted(pc.items())),
        "rule_firing": rule_fire,
        "score_histogram": score_hist,
        "score": {"min": int(scores.min()), "max": int(scores.max()),
                  "mean": round(float(scores.mean()), 2),
                  "p50": int(np.percentile(scores, 50)),
                  "p95": int(np.percentile(scores, 95)),
                  "reject_threshold": meta.get("reject_threshold")},
        "time": {
            "wall_s": meta.get("wall_s"),
            "wall_min": round(meta.get("wall_s", 0) / 60.0, 2),
            "windows_per_s": meta.get("windows_per_s"),
            "footage_h_per_wall_h": meta.get("footage_h_per_wall_h"),
            "processed_footage_h_per_wall_h": meta.get("processed_footage_h_per_wall_h"),
            "ms_per_window_mean": per_stage["total"]["mean"],
            "ms_per_window_p50": per_stage["total"]["p50"],
            "ms_per_window_p95": per_stage["total"]["p95"],
        },
        "per_stage_ms": per_stage,
        "stage_share_pct": share,
        "determinism": determinism,
        "comparison": {
            "E8.01_reference_verdict_distribution": dict(sorted(e8c.items())) or None,
            "E8.01_reference_share_pct": dist_pct(e8c, sum(e8c.values())) if e8 else None,
            "L3_optimized_verdict_distribution": dict(sorted(l3c.items())) or None,
            "L3_optimized_share_pct": dist_pct(l3c, sum(l3c.values())) if l3 else None,
            "L3_optimized_wall_s": l3meta.get("wall_s"),
            "L3_optimized_windows_per_s": l3meta.get("windows_per_s"),
            "L2_partial_verdict_distribution": l2.get("outcomes", {}).get(
                "verdict_distribution"),
            "verdict_confusion_E8.01_rows_candidate_cols": Mcmp.tolist(),
            "verdict_confusion_order": ["accept", "reject", "review"],
            "n_compared_E8.01": ncmp,
            "verdict_agreement_E8.01_pct": round(100.0 * float(np.trace(Mcmp)) / max(ncmp, 1), 2),
            "cohen_kappa_E8.01": kappa(Mcmp),
            "verdict_confusion_L3_rows_candidate_cols": M3.tolist(),
            "n_compared_L3": n3,
            "verdict_agreement_L3_pct": round(100.0 * float(np.trace(M3)) / max(n3, 1), 2),
            "cohen_kappa_L3": kappa(M3),
            "binary_reject_vs_not_E8.01": bin_agree,
            "determinism_cost_note": "Q1 fixed IndEgo-quantile thresholds applied to "
                                     "out-of-domain Egocentric-10K: the disagreement "
                                     "with the trained logistic ensemble is the "
                                     "determinism/OOD trade-off, quantified above.",
        },
        "figures": [],
    }

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "Q5.01.json"), "w"), indent=1)

    try:
        make_figures(rows, res, e8c, l3c, vc, scores)
        res["figures"] = sorted(os.listdir(IMGDIR)) if os.path.isdir(IMGDIR) else []
        json.dump(res, open(os.path.join(OUT, "Q5.01.json"), "w"), indent=1)
    except Exception as ex:  # noqa: BLE001
        print("figure error:", ex)

    write_report(res)
    print(json.dumps({k: res[k] for k in
                      ("corpus", "verdict_distribution", "every_2nd_policy",
                       "time", "stage_share_pct", "determinism")}, indent=1))
    print("wrote", os.path.join(OUT, "Q5.01.json"))


def make_figures(rows, res, e8c, l3c, vc, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(IMGDIR, exist_ok=True)

    # 1. verdict distributions
    labels = ["accept", "reject", "review"]
    series = {
        "Q5 phys rules (1546)": [vc.get(k, 0) for k in labels],
        "Q5 every-2nd (773)": [res["every_2nd_policy"]["verdict_distribution"].get(k, 0) for k in labels],
        "E8.01 logistic (1546)": [e8c.get(k, 0) for k in labels],
        "L3 optimized logistic (1546)": [l3c.get(k, 0) for k in labels],
    }
    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1.5) * w, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("windows")
    ax.set_title("Verdict distribution: deterministic physics rules vs logistic runs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMGDIR, "verdicts.png"), dpi=130)
    plt.close(fig)

    # 2. per-stage share
    share = res["stage_share_pct"]
    st = [s for s in STAGES if s not in ("total", "write")]
    vals = [share.get(s, 0.0) for s in st]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(st, vals, color="C0")
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("share of mean per-window latency (%)")
    ax.set_title("Atomic per-stage share (Q5, 7 workers)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMGDIR, "stage_share.png"), dpi=130)
    plt.close(fig)

    # 3. score histogram
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(scores, bins=range(int(scores.min()), int(scores.max()) + 2),
            color="C3", align="left")
    ax.axvline(res["score"]["reject_threshold"], color="k", ls="--",
               label=f"reject threshold={res['score']['reject_threshold']}")
    ax.set_xlabel("integer rule score (count of crossed thresholds)")
    ax.set_ylabel("windows")
    ax.set_title("Q5 rule-score distribution over the corpus")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMGDIR, "score_hist.png"), dpi=130)
    plt.close(fig)

    # 4. rule firing
    rf = res["rule_firing"]
    names = list(rf)
    vals = [rf[k]["fire"] for k in names]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(names, vals, color="C2")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("windows where the rule fired")
    ax.set_title("Per-rule fire counts (Q5, full corpus)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMGDIR, "rule_fire.png"), dpi=130)
    plt.close(fig)


def write_report(res):
    v = res["verdict_distribution"]
    vs = res["verdict_share_pct"]
    v2 = res["every_2nd_policy"]["verdict_distribution"]
    t = res["time"]
    ps = res["per_stage_ms"]
    c = res["comparison"]
    d = res["determinism"]

    def row(st):
        s = ps.get(st, {})
        return (f"| {st} | {s.get('p50','—')} | {s.get('p95','—')} | "
                f"{s.get('mean','—')} | {res['stage_share_pct'].get(st,0.0)}% |")

    L = []
    L.append("# exp_phys_full — Q5.01 Deterministic physics rules, full corpus\n")
    L.append(f"**Branch:** `m/phys-full` · **Result:** `eval/results/exp_phys_full/Q5.01.json` · "
             f"**Rows:** `eval/results/exp_phys_full/windows.jsonl`\n")
    L.append(f"> {HONESTY}\n")
    L.append("## What ran\n")
    L.append("The L3 latency runner (`eval/exp_latency_fast/run_fast.py`) copied onto this "
             "branch with the **decision stage swapped** to Q1's deterministic physics "
             "rule list (`eval/exp_phys/phys_rules.py` + frozen `rule_table.json`, "
             f"reject_threshold={res['run']['reject_threshold']}, "
             f"rule_table_sha256={res['run']['rule_table_sha256']}). Decode, pixel-26 "
             "features, E4.01 descriptors, the router and the EG2 mean embedding are "
             "unchanged; an extra atomic stage (`phys_features`) computes the 16 physics "
             "features the rules consume. Adopted optimizations: pixel 3.33 fps, EG2 1 fps, "
             "4 s early exit, mosaic off; `--workers 7`, `JADAL_GPU_SLOTS=6`, "
             "`OMP_NUM_THREADS=1`, A40 NVDEC.\n")
    L.append(f"## Corpus\n\n{res['corpus']['n_windows']} windows / "
             f"{res['corpus']['n_clips']} clips / {res['corpus']['total_hours']} h; "
             f"{res['run']['n_errors']} decode errors.\n")
    for e in res["run"].get("errors", []):
        L.append(f"- `{e.get('window_id')}` ({e.get('clip')}): "
                 f"{e.get('error','')[:140]}\n")
    L.append("## Verdict distribution (deterministic rules)\n")
    L.append("| verdict | windows | share |\n|---|---|---|")
    for k in ("accept", "reject", "review"):
        L.append(f"| {k} | {v.get(k,0)} | {vs.get(k,0.0)}% |")
    L.append(f"\nEvery-2nd adopted policy ({res['every_2nd_policy']['n']} windows): "
             f"accept {v2.get('accept',0)} · reject {v2.get('reject',0)} · "
             f"review {v2.get('review',0)}.\n")
    L.append("## Throughput\n")
    L.append(f"- **Total wall:** {t['wall_s']} s ({t['wall_min']} min)\n"
             f"- **Windows/s:** {t['windows_per_s']}\n"
             f"- **Footage-h / wall-hour:** {t['footage_h_per_wall_h']} "
             f"(processed {t['processed_footage_h_per_wall_h']})\n"
             f"- **Per window:** mean {t['ms_per_window_mean']} ms · "
             f"p50 {t['ms_per_window_p50']} ms · p95 {t['ms_per_window_p95']} ms\n")
    L.append("## Per-stage latency (ms)\n")
    L.append("| stage | p50 | p95 | mean | share |\n|---|---|---|---|---|")
    for st in ("io_probe", "decode", "pixel_features", "descriptors", "eg2_embed",
               "phys_features", "decision", "total"):
        L.append(row(st))
    L.append("")
    L.append("## Determinism\n")
    L.append(f"Scored the first {d['n_probed']} windows twice (after the pool): "
             f"**receipt byte-identical = {d['all_receipts_match']}**, "
             f"**features_digest identical = {d['all_features_digests_match']}**. "
             "`receipt_sha256` hashes the contract line **excluding** `latency_ms` "
             "(timing is not deterministic).\n")
    L.append("## Comparison to the logistic runs\n")
    L.append(f"- E8.01 (full, unoptimized): {c['E8.01_reference_verdict_distribution']} "
             f"({c['E8.01_reference_share_pct']})\n"
             f"- L3 (optimized logistic): {c['L3_optimized_verdict_distribution']} "
             f"({c['L3_optimized_share_pct']})\n"
             f"- L2 (partial baseline): {c['L2_partial_verdict_distribution']}\n")
    L.append(f"- Verdict agreement vs E8.01 per-window: "
             f"{c['verdict_agreement_E8.01_pct']}% (n={c['n_compared_E8.01']}, "
             f"kappa={c['cohen_kappa_E8.01']}).\n"
             f"- Verdict agreement vs L3 per-window: {c['verdict_agreement_L3_pct']}% "
             f"(n={c['n_compared_L3']}, kappa={c['cohen_kappa_L3']}).\n"
             f"- Binary reject-vs-not vs E8.01: {c['binary_reject_vs_not_E8.01']}.\n")
    L.append(f"\n{c['determinism_cost_note']}\n")
    L.append("## Per-rule firing (full corpus)\n")
    L.append("| rule | windows | share |\n|---|---|---|")
    for k, vv in res["rule_firing"].items():
        L.append(f"| {k} | {vv['fire']} | {vv['share_pct']}% |")
    L.append("## Interpretation (determinism / OOD trade-off)\n")
    L.append("The deterministic rules **do not transfer** to the Egocentric-10K "
             "corpus: 99.55% of windows are REJECT (score p50 31 vs the "
             f"reject_threshold {res['score']['reject_threshold']}), and four weak/"
             "quality rules fire on 95-100% of windows (CODING_NOISE, FROZEN_STATIC "
             "100%; OVEREXPOSED 98%; NO_ACTIVE_WORK 97%; DUPLICATE 96%). Per-window "
             f"agreement with E8.01 is {c['verdict_agreement_E8.01_pct']}% "
             f"(kappa {c['cohen_kappa_E8.01']}) and with L3 "
             f"{c['verdict_agreement_L3_pct']}%. This is the price of a fully "
             "deterministic, label-free threshold table trained on 7 IndEgo clips: "
             "the physics quantiles are regime-specific, so an out-of-domain "
             "deployment needs either per-regime re-quantiling or a label-free "
             "normalisation. In-domain (Q1) the same model matches the logistic "
             "baseline (AUROC 0.822 vs 0.802).\n")
    L.append("The decision stage itself costs "
             f"{ps['decision'].get('mean','—')} ms mean "
             f"({res['stage_share_pct']['decision']}% of the window) versus "
             "1.05 ms for the logistic decision (L2): the rules are free; the "
             "window is **feature-bound** (decode 76.1%).\n")
    L.append("\n## Figures\n")
    for f in ("verdicts.png", "stage_share.png", "score_hist.png", "rule_fire.png"):
        L.append(f"![{f}](../experiments/exp_phys_full/images/{f})")
    L.append("")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
