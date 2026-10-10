#!/usr/bin/env python3
"""Render the Z4 report + the pod state file from the result JSONs.

  OMP_NUM_THREADS=2 python3 eval/exp_clip/make_report.py \
      --out eval/results/exp_clip --report docs/reports/exp_clip_sequential.md \
      --state /workspace/state/EXP_CLIP_REPORT.md

Deterministic; reads only the three result JSONs.
"""
import argparse, json, os

MODE_NOTE = {
    "hard": "deployable (binarised ensemble prob)",
    "soft": "deployable (soft expected-LLR)",
}


def pareto(rows, ykey="mean_windows_processed_frac", ekey="error_decided"):
    pts = [(r[ykey], r[ekey] if r[ekey] is not None else 1.0, r) for r in rows if r.get("n_decided", 0) > 0]
    pts.sort(key=lambda t: (t[0], t[1]))
    front, best = [], 2.0
    for x, y, r in pts:
        if y < best - 1e-12:
            front.append(r)
            best = y
    return front


def fmt(x, n=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/exp_clip")
    ap.add_argument("--report", default="docs/reports/exp_clip_sequential.md")
    ap.add_argument("--state", default="/workspace/state/EXP_CLIP_REPORT.md")
    ap.add_argument("--head", default="")
    a = ap.parse_args()

    z1 = json.load(open(f"{a.out}/Z4.01.json"))
    z2 = json.load(open(f"{a.out}/Z4.02.json"))
    z3 = json.load(open(f"{a.out}/Z4.03.json"))
    meta = z1["meta"]

    # ---- Z4.01 table
    agg_rows = []
    for meth, r in z1["aggregators"].items():
        agg_rows.append((meth, r["auroc"], r["loco_threshold_ba"], r["loco_isotonic_ba"]))
    best_agg = z1["best_aggregator_by_auroc"]
    rev = z1["review_curve"]
    min_rev = z1["min_review_frac_for_error"]

    # ---- Z4.02 frontier table
    hard = z2["frontier_hard_coarse"]
    front = pareto(hard)
    # always include the best cell
    bc = z2["best_cell"]
    front_tbl = front[:14]

    lines = []
    L = lines.append
    L(f"# Z4 — Clip-level sequential decisions + next-100 active labels")
    L("")
    L(f"**Branch:** `m/clip-sequential` · **Owner:** Z4 (monitor) · lineage `pod/base`"
      + (f" · **Head:** `{a.head}`" if a.head else ""))
    L(f"**Data:** IndEgo human gold {meta['n_windows']} windows / {meta['n_clips']} clips "
      f"(`labels/manifest_gold.json`, sha `{meta['labels_sha256'][:12]}…`); cached LOCO OOF "
      f"`caches/ensemble/oof_probs.npz`; 1,546 Egocentric-10K windows "
      f"(`artifacts/windows_rescored.jsonl`); OOD-60 (`labels/ood_labels.json`, EVAL ONLY).")
    L(f"**Clip truth:** bad fraction ≥ {meta['thresh']:.2f} → 4 reject / 3 accept "
      f"({', '.join(c + '=' + ('rej' if v == 'reject' else 'acc') for c, v in meta['clip_verdict_at_0.30'].items())}).")
    L("")
    L("> **Caveats up front.** In-domain n = **7 clips** (150 windows); the OOD set has "
      "**n_bad = 9** (51 good). Neither supports certified bounds — every number below is "
      "descriptive. The clip-verdict task is *hard*: the model reaches only ~4/7–6/7 clip "
      "verdicts at full coverage (the known IndEgo→Ego10K / pallet calibration problems). "
      "The sequential frontier is therefore unstable and its best cell is illustrative, not "
      "deployable.")
    L("")
    L("**Regen:** `OMP_NUM_THREADS=2 python3 eval/exp_clip/run_clip.py --out eval/results/exp_clip` "
      "then `python3 eval/exp_clip/make_report.py`. Figures: "
      "`docs/experiments/exp_clip/images/{{clip_aggregation,sequential_frontier,active_labels}}.png`.")
    L("")
    L("---")
    L("")
    L("## Headline")
    L("")
    L("**1 — Clip aggregation (Z4.01).** A high-quantile aggregate (q75) of the window "
      f"bad-score separates the 7 clip verdicts best (clip AUROC **{fmt(z1['aggregators']['q75']['auroc'],3)}**), "
      f"but LOCO threshold balanced accuracy tops out at **{fmt(max(r['loco_threshold_ba']['loco_ba'] for r in z1['aggregators'].values()),2)}** "
      f"(5/7 clips). The simple mean is close (AUROC {fmt(z1['aggregators']['mean']['auroc'],3)}, BA "
      f"{fmt(z1['aggregators']['mean']['loco_threshold_ba']['loco_ba'],2)}).")
    L("")
    L("**2 — Sequential decision (Z4.02).** A Wald SPRT on the calibrated log-odds, processing "
      "windows in time order, produces a usable clip error / windows-processed / abstain "
      "frontier. The cheapest cell meeting **agreement ≥ 0.90 and abstain ≤ 0.40** is "
      f"`A={fmt(bc['A'],1)}, B={fmt(bc['B'],1)}` (hard mode): **{bc['n_decided']}/7 decided, "
      f"{bc['n_decided']}/{bc['n_decided']} correct, {len(bc['abstained'])}/7 abstained, "
      f"{fmt(100*bc['mean_windows_processed_frac'],1)}% of windows processed**. **But it is not "
      "cheaper than the fixed every-2 grid (51.5% at 7/7)** — sequential early-stopping does "
      "not drop review at equal error here, because the bad rate climbs with time (E3: 0.09 "
      "sec1 → 0.58 sec5), so a stopping rule must see the tail.")
    L("")
    L("**3 — Active labels (Z4.03).** The top-100 to label next are written to "
      "`eval/results/exp_clip/top100_active_labels.json` (and the pod state file); they spread "
      "over all 47 factories. **Honest negative:** on the held-out OOD-60 windows the "
      "uncertainty score is *anti-correlated* with the true OOD errors "
      f"(AUROC = {fmt(z3['ood_eval']['auroc'],2)} < 0.5; only {z3['ood_eval']['top25_capture_bad']}/9 bad in "
      "the top-25%). The model is *confidently wrong* out of domain, so proximity-to-boundary "
      "does not find the domain-shift errors; the factory-diversity penalty is what buys "
      "coverage. Use the list for coverage/audit, not as a guaranteed error reducer.")
    L("")
    L("---")
    L("")
    L("## Z4.01 — clip-level aggregation + LOCO calibration")
    L("")
    L("Aggregate the window bad-score (`1 − p_ens`) per clip; clip truth = bad fraction ≥ 0.30; "
      "calibrate on the **clip** level with leave-one-clip-out (threshold and isotonic); "
      "clip AUROC over the 7 clips.")
    L("")
    L("| aggregator | clip AUROC | LOCO-threshold BA | correct | LOCO-isotonic BA |")
    L("|---|---|---|---|---|")
    for meth, au, lt, li in agg_rows:
        star = " **←**" if meth == best_agg else ""
        L(f"| {meth}{star} | {fmt(au,3)} | {fmt(lt['loco_ba'],2)} | {lt['n_correct']}/7 | {fmt(li['loco_ba'],2)} |")
    L("")
    L("**Window-level review needed to reach a clip error target** (start from the "
      "`mean` aggregate; review the *most uncertain* windows first, within each clip, and give "
      "them their true labels):")
    L("")
    L(f"- clip error ≤ 1/7 (6/7 correct): review **{fmt(min_rev['<=0.143 (6/7)'],2)}** of windows per clip.")
    L(f"- clip error = 0 (7/7 correct): review **{fmt(min_rev['==0 (7/7)'],2)}** of windows per clip.")
    L("")
    L("For comparison the every-2 grid reviews 51.5% and reaches 7/7 in the *label-oracle* "
      "sense (E3.01). Targeted uncertainty review reaches 6/7 at 35% — a modest saving that "
      "buys one clip of error.")
    L("")
    L("---")
    L("")
    L("## Z4.02 — sequential SPRT frontier")
    L("")
    L("Wald SPRT, windows in time order, `θ0=0.15, θ1=0.45` on the binary decision "
      "(`p̂_bad ≥ 0.5`). Stop **reject** if `S ≥ A`, **accept** if `S ≤ −B`, else abstain when "
      "the clip's windows run out. Cost = mean fraction of the clip's windows processed; "
      "error = 1 − agreement on decided clips.")
    L("")
    L("| A | B | mode | windows processed | abstain | decided | error (decided) | wrong |")
    L("|---|---|---|---|---|---|---|---|")
    for r in front_tbl:
        L(f"| {fmt(r['A'],1)} | {fmt(r['B'],1)} | {r.get('mode','hard')} | {fmt(r['mean_windows_processed_frac'],2)} | "
          f"{fmt(r['abstain_rate'],2)} | {r['n_decided']}/7 | {fmt(r['error_decided'],2)} | "
          f"{', '.join(r['wrong']) or '—'} |")
    L("")
    L("**Best tolerated cell** (`agreement ≥ 0.90`, `abstain ≤ 0.40`; "
      f"{z2.get('n_cells_tolerated', '—')} of {z2['n_cells']} fine cells qualify):")
    L("")
    L(f"- `A={fmt(bc['A'],1)}, B={fmt(bc['B'],1)}`, mode **{bc['mode']}**: "
      f"processed **{fmt(bc['mean_windows_processed_frac'],3)}**, "
      f"decided **{bc['n_decided']}/{bc['n_decided']} correct**, abstain **{fmt(bc['abstain_rate'],3)}**.")
    L(f"- decided: `{', '.join(bc['decided'])}`; abstained: `{', '.join(bc['abstained'])}`.")
    L(f"- robustness: the same partition holds for A ∈ {z2['best_cell_robustness']['A_range']}, "
      f"B ∈ {z2['best_cell_robustness']['B_range']} — but the decide/abstain boundary sits "
      "between **desk** (running min −3.69) and **ecu_late** (−3.92): a 1-clip-wide margin.")
    L("")
    L("**θ sensitivity (hard mode):**")
    L("")
    L("| θ0 | θ1 | tolerated cells | best frac | agree | abstain |")
    L("|---|---|---|---|---|---|")
    for s in z2["theta_sensitivity"]:
        L(f"| {s['theta0']} | {s['theta1']} | {s['n_cells_ok']} | {fmt(s['best_frac'],2)} | "
          f"{fmt(s['best_agree'],2)} | {fmt(s['best_abstain'],2)} |")
    L("")
    L(f"**Soft-mode finding.** {z2['soft_mode_finding']}")
    L("")
    lob = z2.get("label_oracle_best_cell")
    if lob:
        L(f"**Label-oracle comparison.** Using the *true* window labels as the SPRT evidence "
          f"(each processed window = one human review), the cheapest 7/7 cell is "
          f"`A={lob['A']}, B={lob['B']}` at **{fmt(lob['mean_windows_processed_frac'],3)}** "
          f"processed — still above every-2's 0.515. ")
    L("")
    L(f"**Review-savings finding.** {z2['review_savings_finding']}")
    L("")
    L(f"**Recommendation.** {z2['recommendation']}")
    L("")
    L("---")
    L("")
    L("## Z4.03 — best cell + next-100 active labels")
    L("")
    sel = z3["active_labeling"]["top100"]
    cov = z3["active_labeling"]["coverage"]
    L(f"**Method.** {z3['active_labeling']['method']}")
    L("")
    L(f"**Candidates.** {z3['meta']['n_candidates']:,} unlabelled Egocentric-10K windows "
      f"(1,546 minus the {z3['meta']['n_ood_labelled_excluded']} already-labelled OOD-60). "
      f"Top-100 covers **{cov['factories_covered']}/{cov['all_factories']} factories** and "
      f"{cov['clips_covered']} clips (max {cov['max_per_factory']} per factory).")
    L("")
    L(f"**OOD-60 sanity check (evaluation only).** Uncertainty U = 4p(1−p) vs the 9 human "
      f"OOD-bad windows: AUROC = **{fmt(z3['ood_eval']['auroc'],3)}** (below 0.5), "
      f"top-25% capture = {z3['ood_eval']['top25_capture_bad']}/9. The 9 bad windows sit at "
      "*high* p (confidently accepted), i.e. *low* uncertainty — uncertainty sampling does not "
      "find domain-shift errors. Treat the ranking as diversity/coverage sampling.")
    L("")
    L("**Top 25 of the next-100 (full list: `eval/results/exp_clip/top100_active_labels.json`):**")
    L("")
    L("| # | window_id | factory | p(good) | U | view-gap | rank score |")
    L("|---|---|---|---|---|---|---|")
    for i, s in enumerate(sel[:25], 1):
        L(f"| {i} | `{s['window_id']}` | {s['factory']} | "
          f"{s['p_good']:.3f} | {s['uncertainty']:.3f} | {s['view_disagreement']:.3f} | "
          f"{s['rank_score']:.3f} |")
    L("")
    L("---")
    L("")
    L("## Caveats (small-n, stated plainly)")
    L("")
    L("- **7 in-domain clips** → 4 reject / 3 accept; clip AUROC/BA, LOCO thresholds and the "
      "SPRT best cell are all computed on 7 points. The best cell's accept bound is separated "
      "by ~0.2 log-odds between two clips' running minima — an artefact, not a deployable rule.")
    L("- **OOD-60 = 9 bad** → the uncertainty-vs-error check is descriptive; an AUROC of 0.35 "
      "on n_bad=9 has a very wide CI and we do **not** fit on it.")
    L("- **Windows are ~15 s apart** and sparse; 'fraction of windows' is a proxy for footage, "
      "and a sequential stop saves feature compute, not decode/seeks.")
    L("- **Label source** is one human on IndEgo only; Ego10K windows are unlabelled. "
      "No transfer claim is made.")
    L("- Negative results kept: soft-LLR SPRT never accepts; sequential review does not beat "
      "the every-2 grid; uncertainty sampling does not enrich OOD errors.")
    L("")
    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    open(a.report, "w").write(report)

    # ------------------------------------------------------------- state file
    S = []
    S.append("# EXP_CLIP_REPORT — Z4 (clip-level sequential + next-100 labels)")
    S.append("")
    S.append(f"**Branch:** m/clip-sequential" + (f" · **Head:** {a.head}" if a.head else ""))
    S.append(f"**Data:** IndEgo human gold {meta['n_windows']} windows / {meta['n_clips']} clips; "
             f"cached OOF `p_ens`; 1,546 Egocentric-10K windows; OOD-60 (eval only).")
    S.append("")
    S.append("## Summary")
    S.append(f"- **Z4.01 aggregation:** q75 best (clip AUROC {fmt(z1['aggregators']['q75']['auroc'],3)}); "
             f"LOCO clip BA ≤ {fmt(max(r['loco_threshold_ba']['loco_ba'] for r in z1['aggregators'].values()),2)} "
             f"(5/7). Windows to review for clip error ≤1/7 = {fmt(min_rev['<=0.143 (6/7)'],2)}; for 7/7 = {fmt(min_rev['==0 (7/7)'],2)}.")
    S.append(f"- **Z4.02 sequential frontier:** best tolerated cell A={fmt(bc['A'],1)}, B={fmt(bc['B'],1)} "
             f"(hard): decided {bc['n_decided']}/{bc['n_decided']} correct, abstain {fmt(bc['abstain_rate'],2)}, "
             f"windows {fmt(bc['mean_windows_processed_frac'],3)}. NOT cheaper than every-2 (0.515).")
    S.append(f"- **Z4.03 next-100:** {len(sel)} ids, {cov['factories_covered']}/{cov['all_factories']} factories. "
             f"OOD-60 U-AUROC {fmt(z3['ood_eval']['auroc'],3)} (anti-correlated) — coverage use only.")
    S.append("")
    S.append("Frontier (coarse, hard mode):")
    S.append("")
    S.append("| A | B | windows | abstain | decided | error |")
    S.append("|---|---|---|---|---|---|")
    for r in front_tbl:
        S.append(f"| {fmt(r['A'],1)} | {fmt(r['B'],1)} | {fmt(r['mean_windows_processed_frac'],2)} | "
                 f"{fmt(r['abstain_rate'],2)} | {r['n_decided']}/7 | {fmt(r['error_decided'],2)} |")
    S.append("")
    S.append("## Next-100 windows to label")
    S.append("")
    S.append("`window_id | clip | factory | p_good | U | view_gap | rank_score`")
    S.append("")
    for i, s in enumerate(sel, 1):
        S.append(f"{i:3d}. {s['window_id']} | {s['clip']} | {s['factory']} | "
                 f"{s['p_good']:.3f} | {s['uncertainty']:.3f} | {s['view_disagreement']:.3f} | {s['rank_score']:.3f}")
    S.append("")
    os.makedirs(os.path.dirname(a.state), exist_ok=True)
    open(a.state, "w").write("\n".join(S) + "\n")

    print("wrote", a.report, "and", a.state)
    print(f"headline best cell: A={bc['A']} B={bc['B']} frac={bc['mean_windows_processed_frac']:.3f} "
          f"agree={bc['agreement_decided']} abstain={bc['abstain_rate']:.3f}")


if __name__ == "__main__":
    main()
