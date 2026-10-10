#!/usr/bin/env python3
"""Generate docs/reports/exp_acc_fusion.md from the N3.01/02/03 JSONs."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "eval", "results", "exp_acc")
OUT = os.path.join(ROOT, "docs", "reports", "exp_acc_fusion.md")

s1 = json.load(open(os.path.join(RES, "N3.01.json")))
s2 = json.load(open(os.path.join(RES, "N3.02.json")))
s3 = json.load(open(os.path.join(RES, "N3.03.json")))


def f(v):
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


L = []
L.append("# N3 — Fusion + routing: deterministic rules × adapted model × frozen ensemble\n")
L.append("**Pod:** jadal-qa · **Branch:** `m/acc-fusion` · **PR:** #159 (base `pod/base`)\n")
L.append("**Question.** Fuse and route the available scorers so accuracy goes up and review goes down — "
         "the quality-vs-semantic split.\n")
L.append("**Device:** CPU (numpy/sklearn; cached features only, no GPU). "
         "**Git sha:** `%s`.\n" % s1.get("git_sha", "?"))
L.append("**Results:** `eval/results/exp_acc/N3.01.json` (split), `N3.02.json` (routing), "
         "`N3.03.json` (fusion). **Figure:** `docs/experiments/exp_acc/images/n3_fusion_routing.png`.\n")

# ---- data
L.append("## Data & protocol\n")
L.append("| set | n | good | bad | bad reasons | protocol |")
L.append("|---|---|---|---|---|---|")
for dom, tag, proto in (("indego_loco", "IndEgo gold", "leave-one-clip-out (7 clips)"),
                        ("ood160", "OOD-160 Egocentric-10K", "frozen scores; factory-grouped CI")):
    D = s1["domains"][dom]
    rc = ", ".join(f"{k}:{v}" for k, v in sorted(D["bad_reason_counts"].items()))
    L.append(f"| {tag} | {D['n']} | {D['n_good']} | {D['n_bad']} | {rc} | {proto} |")
L.append("")
L.append("**Scorers** (all frozen, nothing refit on OOD): **rules** = deterministic physics rule table "
         "(`tools/phys`, vendored from PR #144), fold-local thresholds on gold, frozen table on OOD; "
         "**frozen** = ensemble_v1 two-view logistic (gold OOF `p_ens`, OOD `usable_prob`); "
         "**adapted** = Z3 6 transferable features, per-domain z-score, IndEgo-signed mean "
         "(reproduces `m/adapt160` recipe (a): 0.684 in-domain / 0.759 OOD).\n")
L.append("**Sub-problems.** quality = TOO_DARK, BLUR, OVEREXPOSED, FROZEN_STATIC, DUPLICATE, "
         "UNDECODABLE. semantic = NO_ACTIVE_WORK, HANDS_NOT_VISIBLE. "
         "AUROC below = bad-detection of {all good} vs {bad windows whose reason is in the category}.\n")

# ---- N3.01
L.append("## 1. Quality-vs-semantic split (N3.01)\n")
L.append("| domain | scorer | all AUROC [grouped 95% CI] | quality AUROC (n) | semantic AUROC (n) |")
L.append("|---|---|---|---|---|")
for dom, tag in (("indego_loco", "IndEgo LOCO"), ("ood160", "OOD-160")):
    D = s1["domains"][dom]["scorers"]
    for sc in ("rules", "rules_quality", "rules_semantic", "frozen", "adapted"):
        r = D[sc]
        ci = r.get("auroc_all_ci95_grouped")
        cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
        L.append(f"| {tag} | {sc} | {f(r['auroc_all'])} {cis} | "
                 f"{f(r['auroc_quality'])} ({r['n_quality_bad']}) | "
                 f"{f(r['auroc_semantic'])} ({r['n_semantic_bad']}) |")
L.append("")
L.append("**What the deterministic rules actually do (native ACCEPT/REJECT/REVIEW):**\n")
L.append("| domain | group | n | REJECT | REVIEW | ACCEPT |")
L.append("|---|---|---|---|---|---|")
for dom, tag in (("indego_loco", "IndEgo"), ("ood160", "OOD-160")):
    nd = s1["domains"][dom]["rules_native_decisions"]
    for grp in ("quality_bad", "semantic_bad", "good"):
        g = nd[grp]
        if g.get("n", 0) == 0:
            L.append(f"| {tag} | {grp} | 0 | — | — | — |")
        else:
            L.append(f"| {tag} | {grp} | {g['n']} | {g['reject']} | {g['review']} | {g['accept']} |")
L.append("")
L.append("**Reading.** In-domain the quality rules are near-perfect (AUROC 0.957; 11/13 quality-bad "
         "REJECT, 0 ACCEPT) while semantic is the hard part (rules 0.775, frozen 0.821). OOD contains "
         "**no quality-bad windows** (all 22 bad are semantic/other), so *quality transfer is "
         "untestable here*; and the deterministic rules invert on the OOD quality-only score (0.312) "
         "while firing on 93/138 good OOD windows. The semantic OOD sub-problem is solved only by the "
         "adapted model (0.749). So the split explains *where each scorer wins*, not a clean "
         "'quality transfers / semantic does not' law: the OOD failures are semantic, and the "
         "transferable win comes from the adapted capture-telemetry model, not from the semantic rules.\n")

# ---- N3.02
L.append("## 2. Routing (N3.02)\n")
L.append("Review proxy: balanced-Platt calibrated P(good) (grouped CV), REVIEW iff P∈[0.4,0.6]; "
         "accept iff P≥0.5. AUROC is on the raw rank score (calibration is an operating-point proxy only). "
         "`route_aspect_wide` = adapted iff width/height ≥ 1.3 else rules. "
         "Note this proxy is *not* the rules' native REVIEW: the deterministic rule table reviews "
         "60% of in-domain and 36% of OOD windows (N3.01); the calibrated band is a model-comparable "
         "alternative.\n")
L.append("| policy | in-domain AUROC | in-domain review | in-domain BA | OOD AUROC [CI] | OOD review | OOD BA | mean AUROC |")
L.append("|---|---|---|---|---|---|---|---|")
order = ["rules", "frozen", "adapted", "route_fps_ge25", "route_profile_factory",
         "route_aspect_wide", "route_rules_review_to_adapted", "route_quality_desc"]
summ = s2.get("summary_mean_domain_auroc", {})
for n in order:
    gi = s2["domains"]["indego_loco"]["scorers"][n]
    go = s2["domains"]["ood160"]["scorers"][n]
    ci = go.get("auroc_ci95_grouped")
    cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
    L.append(f"| {n} | {f(gi['auroc'])} | {f(gi['review_share'])} | {f(gi['balacc_band_as_accept'])} | "
             f"{f(go['auroc'])} {cis} | {f(go['review_share'])} | {f(go['balacc_band_as_accept'])} | "
             f"{f(summ.get(n))} |")
L.append("")
L.append("**Reading.** No single scorer is good in both regimes (rules 0.822→0.427; adapted 0.684→0.759; "
         "frozen 0.834→0.566). The E4 regime descriptors (`fps≥25`, profile `factory_1080p30`) catch only "
         "the 30 fps OOD batch — the OOD label windows are all 854×480, so width/fps do **not** separate "
         "the regimes and those routers sit between the two scorers. Aspect ratio (IndEgo is 480×480, "
         "Egocentric-10K is 16:9) is the only descriptor that separates them: "
         "`route_aspect_wide` = 0.822 in-domain / 0.759 OOD (mean 0.791), the best mean-domain AUROC of "
         "any single scorer or router, at 24%/26% review. The content-based quality gate "
         "(`route_quality_desc`) is a **negative result**: routing OOD quality-looking windows to the "
         "rules drops OOD AUROC to 0.680.\n")

# ---- N3.03
L.append("## 3. Fusion (N3.03)\n")
L.append("Rank-average of per-domain rank01 good-scores (Y3 recipe). Review proxy as above.\n")
L.append("| combo | in-domain AUROC | in-domain review | in-domain BA | OOD AUROC [CI] | OOD review | OOD BA | mean AUROC |")
L.append("|---|---|---|---|---|---|---|---|")
csumm = s3.get("summary_mean_domain_auroc", {})
for c in ("frozen", "adapted", "rules", "rank_adapted+frozen", "rank_adapted+frozen+rules"):
    gi = s3["domains"]["indego_loco"]["combos"][c]
    go = s3["domains"]["ood160"]["combos"][c]
    ci = go.get("auroc_ci95_grouped")
    cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
    L.append(f"| {c} | {f(gi['auroc'])} | {f(gi['review_share'])} | {f(gi['balacc_band_as_accept'])} | "
             f"{f(go['auroc'])} {cis} | {f(go['review_share'])} | {f(go['balacc_band_as_accept'])} | "
             f"{f(csumm.get(c))} |")
L.append("")
gf = s3["domains"]["ood160"]["combos"]["rank_adapted+frozen"]
for key, lab in (("gain_vs_frozen_paired", "vs frozen"), ("gain_vs_adapted_paired", "vs adapted")):
    g = gf.get(key)
    if g:
        L.append(f"- OOD paired factory-grouped AUROC gain of `rank_adapted+frozen` {lab}: "
                 f"{g['mean']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] "
                 f"({g['frac_gt0']*100:.1f}% of resamples > 0).")
L.append("")
L.append("**Reading.** Fusion is a **compromise, not a win**. In-domain it is dragged down by the weaker "
         "adapted score (frozen 0.834 → fused 0.784); OOD it is dragged down by the near-chance frozen "
         "score (adapted 0.759 → fused 0.686). Its mean-domain AUROC (0.735) does beat every single "
         "scorer (adapted 0.722, frozen 0.700) but is below the routed system (0.791). Fusion also does "
         "**not** cut the ambiguous fraction: OOD review rises 0.31 (adapted) → 0.53 (fused) → 0.86 "
         "(+rules).\n")

# ---- joint
L.append("## 4. Joint point: fused + routed, review% vs OOD error\n")
L.append("| OOD configuration | AUROC [grouped CI] | route share | REVIEW | BA (band→accept) | err on decided |")
L.append("|---|---|---|---|---|---|")
j = s3["joint_point_ood"]
for k in ("adapted_only", "quality_gate_rules_else_fused", "aspect_wide_fused_else_rules"):
    r = j[k]
    ci = r["ood_auroc_ci95_grouped"]
    rs = ", ".join(f"{a}:{b:.2f}" for a, b in r["route_share"].items())
    L.append(f"| {r['system']} | {f(r['ood_auroc'])} [{ci[0]:.2f}, {ci[1]:.2f}] | "
             f"{rs} | {f(r['review_share'])} | {f(r['balacc_band_as_accept'])} | "
             f"{f(r['err_decided_only'])} |")
L.append("")
L.append("**Best configuration.** For OOD the best scorer is **adapted alone**: AUROC 0.759 "
         "[0.597, 0.886], REVIEW 0.256, BA 0.697, err-on-decided 0.219. Fusing or quality-gating the "
         "rules into it only adds ambiguity (0.53–0.68 REVIEW) and lowers AUROC (0.63–0.69). The best "
         "*system* across the domain split is the **deterministic aspect-ratio router**: rules in-domain "
         "(0.822, REVIEW 0.24) + adapted OOD (0.759, REVIEW 0.26) — mean AUROC 0.791.\n")
L.append("### Honesty caveats\n")
L.append("- **Aspect ratio is a dataset shortcut, not a semantic detector.** IndEgo is 480×480 and the "
         "OOD label windows are 854×480; a square OOD source would be misrouted to the rules. The E4 "
         "descriptors (fps/profile/quality) do not separate the regimes, so a deployable router needs a "
         "better in-domain detector than the descriptors currently cached.\n")
L.append("- **OOD quality transfer is untestable** with the current labels: 0/22 OOD bad windows are "
         "quality failures. The 'quality transfers' half of the hypothesis cannot be confirmed or "
         "refuted OOD.\n")
L.append("- **Small OOD bad count** (n_bad=22 over 47 factories) → wide grouped CIs; the adapted OOD "
         "AUROC CI is [0.60, 0.89].\n")
L.append("- **Review proxy uses labels** (balanced Platt calibration, grouped CV) as a measurement; it "
         "is not a deployable review gate. The adapted score's calibration does not transfer (m/adapt160), "
         "so the OOD REVIEW number is descriptive.\n")
L.append("- Fusion used cached, frozen scorers only; no new inference, no GPU, no label edits.\n")
L.append("\n## Reproduce\n")
L.append("```bash\nOMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc_fusion.py\n"
         "python3 eval/exp_acc/make_figure.py\npython3 eval/exp_acc/make_report.py\n```\n")
L.append("Cached binary inputs (`eval/results/exp_acc/inputs/ood_full.npz` from `m/phys-ready` "
         "`eval/results/exp_phys/inputs/`, and `z3_160.npz` from `m/adapt160` "
         "`eval/results/exp_adapt160/inputs/`) are `*.npz`-gitignored; the E4 descriptor `.jsonl` "
         "inputs are committed here so the router is reproducible. `tools/phys` is vendored from "
         "PR #144 (`m/phys-ready`) pending merge.\n")

with open(OUT, "w") as fh:
    fh.write("\n".join(L))
print("wrote", OUT)
