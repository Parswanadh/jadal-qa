"""Finalise H2 true-1080p mosaic (E2.1080) -> result JSONs + console table.

Reads eval/results/exp_mosaic1080/{features.npz,raw_eval.json}.
Regen: python3 eval/exp_mosaic1080/finalize_1080.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "eval", "results", "exp_mosaic1080")

PROTOCOL = ("out-of-domain read on human-labelled Egocentric-10K windows "
            "(ood160_labels.json): stratified 5-fold CV logistic C=0.01 on "
            "L2-normalised EG2 mosaic embeddings, out-of-fold AUROC, "
            "factory-bootstrap CI B=1000 seed0 (optimistic: trains and tests "
            "on the same OOD pool; NOT a transfer proof)")
LABEL_SOURCE = ("human (Egocentric-10K windows labelled good/bad, one labeller; "
                "subset keeps ALL bad windows + even good sample)")


def main():
    raw = json.load(open(os.path.join(R, "raw_eval.json")))
    res, per, n = raw["res"], raw["per_cfg"], raw["n"]
    git = os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip()
    head = {"n": n, "label_source": LABEL_SOURCE, "protocol": PROTOCOL,
            "device": "A40", "git_sha": git,
            "command": ("bash tools/pod/gpu_run.sh python3 "
                        "eval/exp_mosaic1080/run_1080.py --limit 80"),
            "n_good": res["h1080_3x3_t224"]["n_good"],
            "n_bad": res["h1080_3x3_t224"]["n_bad"],
            "cost_totals_ms": raw["cost"],
            "decode_paths": raw["decode_paths"],
            "small_n_warning": ("n_bad=22: CIs are wide; any within-0.02-AUROC "
                                "call is a tie-by-noise, stated as such")}

    def row(tag):
        r = res[tag]
        d = dict(per[tag])
        d.update({"tag": tag, "auroc": r["auroc"], "ci95": r["ci95"],
                  "bal_acc": r["bal_acc"], "n": r["n"]})
        return d

    # ---- ladder: 1080p source, tile px down at fixed grids ----
    ladder = {t: row(t) for t in
              ["h1080_3x3_t224", "h1080_3x3_t168", "h1080_3x3_t112",
               "h1080_4x4_t168", "h1080_4x4_t112"]}
    base = ladder["h1080_3x3_t224"]["auroc"]
    for t, d in ladder.items():
        d["d_auroc_vs_baseline"] = round(d["auroc"] - base, 4)
    m9, m16 = res["mean1080_N9"]["auroc"], res["mean1080_N16"]["auroc"]
    lad = dict(head, experiment="E2.1080-LADDER",
               title="true-1080p tile-resolution ladder (timestamps top-right)",
               variant="mosaic EG2 embedding; grids 3x3/4x4, tile 224/168/112",
               results=ladder,
               mean1080_N9=res["mean1080_N9"], mean1080_N16=res["mean1080_N16"],
               gap_best_mosaic_vs_meanN16=round(
                   max(d["auroc"] for d in ladder.values()) - m16, 4))

    # ---- duel: equal canvas, 1080p-few-big vs 480p-many-small ----
    a = row("h1080_3x3_t224")
    b = row("h480_4x4_t168")
    c = row("h480_3x3_t224")
    duel = dict(head, experiment="E2.1080-DUEL",
                title="equal-canvas duel: 1080p 3x3/t224 vs 480p 4x4/t168",
                canvas_px_a=a["canvas_px"], canvas_px_b=b["canvas_px"],
                a=a, b=b, source_ablation_480_3x3_t224=c,
                d_auroc_b_minus_a=round(b["auroc"] - a["auroc"], 4),
                d_auroc_480src_minus_1080src_3x3=round(c["auroc"] - a["auroc"], 4),
                build_ms_ratio_b_over_a=round(
                    b["build_ms_per_win"] / max(0.1, a["build_ms_per_win"]), 3),
                embed_ms_ratio_b_over_a=round(
                    b["embed_ms_per_win"] / max(0.1, a["embed_ms_per_win"]), 3))

    json.dump(lad, open(os.path.join(R, "exp_mosaic1080_ladder.json"), "w"), indent=1)
    json.dump(duel, open(os.path.join(R, "exp_mosaic1080_duel.json"), "w"), indent=1)

    print("=== LADDER (1080p src; AUROC [CI] | canvas px | build+embed ms/win) ===")
    for t, d in ladder.items():
        print(f"{t:18s} {d['auroc']:.3f} {d['ci95']} d={d['d_auroc_vs_baseline']:+.3f} "
              f"| {d['canvas_px']:6d}px | {d['build_ms_per_win']:.0f}+{d['embed_ms_per_win']:.0f}ms")
    print(f"mean1080_N9 {m9:.3f} {res['mean1080_N9']['ci95']} | "
          f"mean1080_N16 {m16:.3f} {res['mean1080_N16']['ci95']}")
    print("\n=== DUEL (equal canvas ~463-465k px) ===")
    print(f"A 1080p 3x3/t224: {a['auroc']:.3f} {a['ci95']} {a['build_ms_per_win']:.0f}+{a['embed_ms_per_win']:.0f}ms")
    print(f"B 480p  4x4/t168: {b['auroc']:.3f} {b['ci95']} {b['build_ms_per_win']:.0f}+{b['embed_ms_per_win']:.0f}ms")
    print(f"d(B-A)={duel['d_auroc_b_minus_a']:+.3f}; src-ablation d(480-1080|3x3)="
          f"{duel['d_auroc_480src_minus_1080src_3x3']:+.3f}")
    print("wrote", R)


if __name__ == "__main__":
    main()
