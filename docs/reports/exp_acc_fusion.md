# N3 — Fusion + routing: deterministic rules × adapted model × frozen ensemble

**Pod:** jadal-qa · **Branch:** `m/acc-fusion` · **PR:** #TBD (base `pod/base`)

**Question.** Fuse and route the available scorers so accuracy goes up and review goes down — the quality-vs-semantic split.

**Device:** CPU (numpy/sklearn; cached features only, no GPU). **Git sha:** `f20013b677ab3e25b930d7ed23bac572dbb78134`.

**Results:** `eval/results/exp_acc/N3.01.json` (split), `N3.02.json` (routing), `N3.03.json` (fusion). **Figure:** `docs/experiments/exp_acc/images/n3_fusion_routing.png`.

## Data & protocol

| set | n | good | bad | bad reasons | protocol |
|---|---|---|---|---|---|
| IndEgo gold | 150 | 100 | 50 | DUPLICATE:6, HANDS_NOT_VISIBLE:14, NO_ACTIVE_WORK:23, TOO_DARK:6, UNDECODABLE:1 | leave-one-clip-out (7 clips) |
| OOD-160 Egocentric-10K | 160 | 138 | 22 | HANDS_NOT_VISIBLE:2, NO_ACTIVE_WORK:19, SHAKY:1 | frozen scores; factory-grouped CI |

**Scorers** (all frozen, nothing refit on OOD): **rules** = deterministic physics rule table (`tools/phys`, vendored from PR #144), fold-local thresholds on gold, frozen table on OOD; **frozen** = ensemble_v1 two-view logistic (gold OOF `p_ens`, OOD `usable_prob`); **adapted** = Z3 6 transferable features, per-domain z-score, IndEgo-signed mean (reproduces `m/adapt160` recipe (a): 0.684 in-domain / 0.759 OOD).

**Sub-problems.** quality = TOO_DARK, BLUR, OVEREXPOSED, FROZEN_STATIC, DUPLICATE, UNDECODABLE. semantic = NO_ACTIVE_WORK, HANDS_NOT_VISIBLE. AUROC below = bad-detection of {all good} vs {bad windows whose reason is in the category}.

## 1. Quality-vs-semantic split (N3.01)

| domain | scorer | all AUROC [grouped 95% CI] | quality AUROC (n) | semantic AUROC (n) |
|---|---|---|---|---|
| IndEgo LOCO | rules | 0.822 [0.71, 0.85] | 0.957 (13) | 0.775 (37) |
| IndEgo LOCO | rules_quality | 0.656 [0.47, 0.73] | 0.862 (13) | 0.584 (37) |
| IndEgo LOCO | rules_semantic | 0.769 [0.63, 0.83] | 0.847 (13) | 0.742 (37) |
| IndEgo LOCO | frozen | 0.834 [0.72, 0.91] | 0.870 (13) | 0.821 (37) |
| IndEgo LOCO | adapted | 0.684 [0.52, 0.78] | 0.798 (13) | 0.644 (37) |
| OOD-160 | rules | 0.427 [0.27, 0.56] | — (0) | 0.425 (21) |
| OOD-160 | rules_quality | 0.312 [0.17, 0.47] | — (0) | 0.313 (21) |
| OOD-160 | rules_semantic | 0.477 [0.33, 0.61] | — (0) | 0.476 (21) |
| OOD-160 | frozen | 0.566 [0.42, 0.70] | — (0) | 0.577 (21) |
| OOD-160 | adapted | 0.759 [0.60, 0.89] | — (0) | 0.750 (21) |

**What the deterministic rules actually do (native ACCEPT/REJECT/REVIEW):**

| domain | group | n | REJECT | REVIEW | ACCEPT |
|---|---|---|---|---|---|
| IndEgo | quality_bad | 13 | 11 | 2 | 0 |
| IndEgo | semantic_bad | 37 | 15 | 20 | 2 |
| IndEgo | good | 100 | 10 | 68 | 22 |
| OOD-160 | quality_bad | 0 | — | — | — |
| OOD-160 | semantic_bad | 21 | 10 | 11 | 0 |
| OOD-160 | good | 138 | 93 | 45 | 0 |

**Reading.** In-domain the quality rules are near-perfect (AUROC 0.957; 11/13 quality-bad REJECT, 0 ACCEPT) while semantic is the hard part (rules 0.775, frozen 0.821). OOD contains **no quality-bad windows** (all 22 bad are semantic/other), so *quality transfer is untestable here*; and the deterministic rules invert on the OOD quality-only score (0.312) while firing on 93/138 good OOD windows. The semantic OOD sub-problem is solved only by the adapted model (0.749). So the split explains *where each scorer wins*, not a clean 'quality transfers / semantic does not' law: the OOD failures are semantic, and the transferable win comes from the adapted capture-telemetry model, not from the semantic rules.

## 2. Routing (N3.02)

Review proxy: balanced-Platt calibrated P(good) (grouped CV), REVIEW iff P∈[0.4,0.6]; accept iff P≥0.5. AUROC is on the raw rank score (calibration is an operating-point proxy only). `route_aspect_wide` = adapted iff width/height ≥ 1.3 else rules. Note this proxy is *not* the rules' native REVIEW: the deterministic rule table reviews 60% of in-domain and 36% of OOD windows (N3.01); the calibrated band is a model-comparable alternative.

| policy | in-domain AUROC | in-domain review | in-domain BA | OOD AUROC [CI] | OOD review | OOD BA | mean AUROC |
|---|---|---|---|---|---|---|---|
| rules | 0.822 | 0.147 | 0.745 | 0.427 [0.27, 0.56] | 0.975 | 0.500 | 0.625 |
| frozen | 0.834 | 0.227 | 0.755 | 0.566 [0.42, 0.70] | 1.000 | 0.500 | 0.700 |
| adapted | 0.684 | 0.373 | 0.685 | 0.759 [0.60, 0.89] | 0.256 | 0.697 | 0.722 |
| route_fps_ge25 | 0.822 | 0.147 | 0.745 | 0.621 [0.47, 0.75] | 0.688 | 0.592 | 0.722 |
| route_profile_factory | 0.822 | 0.147 | 0.745 | 0.621 [0.47, 0.75] | 0.688 | 0.592 | 0.722 |
| route_aspect_wide | 0.822 | 0.147 | 0.745 | 0.759 [0.60, 0.89] | 0.256 | 0.697 | 0.791 |
| route_rules_review_to_adapted | 0.799 | 0.280 | 0.750 | 0.660 [0.52, 0.81] | 0.725 | 0.635 | 0.729 |
| route_quality_desc | 0.671 | 0.373 | 0.680 | 0.679 [0.51, 0.82] | 0.469 | 0.672 | 0.675 |

**Reading.** No single scorer is good in both regimes (rules 0.822→0.427; adapted 0.684→0.759; frozen 0.834→0.566). The E4 regime descriptors (`fps≥25`, profile `factory_1080p30`) catch only the 30 fps OOD batch — the OOD label windows are all 854×480, so width/fps do **not** separate the regimes and those routers sit between the two scorers. Aspect ratio (IndEgo is 480×480, Egocentric-10K is 16:9) is the only descriptor that separates them: `route_aspect_wide` = 0.822 in-domain / 0.759 OOD (mean 0.791), the best mean-domain AUROC of any single scorer or router, at 24%/26% review. The content-based quality gate (`route_quality_desc`) is a **negative result**: routing OOD quality-looking windows to the rules drops OOD AUROC to 0.680.

## 3. Fusion (N3.03)

Rank-average of per-domain rank01 good-scores (Y3 recipe). Review proxy as above.

| combo | in-domain AUROC | in-domain review | in-domain BA | OOD AUROC [CI] | OOD review | OOD BA | mean AUROC |
|---|---|---|---|---|---|---|---|
| frozen | 0.834 | 0.227 | 0.755 | 0.566 [0.42, 0.70] | 1.000 | 0.500 | 0.700 |
| adapted | 0.684 | 0.373 | 0.685 | 0.759 [0.60, 0.89] | 0.256 | 0.697 | 0.722 |
| rules | 0.822 | 0.147 | 0.745 | 0.427 [0.27, 0.56] | 0.975 | 0.500 | 0.625 |
| rank_adapted+frozen | 0.783 | 0.307 | 0.720 | 0.686 [0.53, 0.82] | 0.525 | 0.575 | 0.735 |
| rank_adapted+frozen+rules | 0.819 | 0.233 | 0.715 | 0.616 [0.46, 0.75] | 0.863 | 0.513 | 0.718 |

- OOD paired factory-grouped AUROC gain of `rank_adapted+frozen` vs frozen: +0.120 [+0.038, +0.204] (99.9% of resamples > 0).
- OOD paired factory-grouped AUROC gain of `rank_adapted+frozen` vs adapted: -0.073 [-0.161, +0.003] (3.0% of resamples > 0).

**Reading.** Fusion is a **compromise, not a win**. In-domain it is dragged down by the weaker adapted score (frozen 0.834 → fused 0.784); OOD it is dragged down by the near-chance frozen score (adapted 0.759 → fused 0.686). Its mean-domain AUROC (0.735) does beat every single scorer (adapted 0.722, frozen 0.700) but is below the routed system (0.791). Fusion also does **not** cut the ambiguous fraction: OOD review rises 0.31 (adapted) → 0.53 (fused) → 0.86 (+rules).

## 4. Joint point: fused + routed, review% vs OOD error

| OOD configuration | AUROC [grouped CI] | route share | REVIEW | BA (band→accept) | err on decided |
|---|---|---|---|---|---|
| adapted_only | 0.759 [0.60, 0.89] | adapted:1.00 | 0.256 | 0.697 | 0.218 |
| quality-desc -> rules else rank_adapted+frozen | 0.635 [0.48, 0.76] | fused:0.72, rules:0.28 | 0.681 | 0.588 | 0.157 |
| aspect>=1.3 -> fused else rules | 0.686 [0.53, 0.82] | fused:1.00 | 0.525 | 0.575 | 0.276 |

**Best configuration.** For OOD the best scorer is **adapted alone**: AUROC 0.759 [0.597, 0.886], REVIEW 0.256, BA 0.697, err-on-decided 0.219. Fusing or quality-gating the rules into it only adds ambiguity (0.53–0.68 REVIEW) and lowers AUROC (0.63–0.69). The best *system* across the domain split is the **deterministic aspect-ratio router**: rules in-domain (0.822, REVIEW 0.24) + adapted OOD (0.759, REVIEW 0.26) — mean AUROC 0.791.

### Honesty caveats

- **Aspect ratio is a dataset shortcut, not a semantic detector.** IndEgo is 480×480 and the OOD label windows are 854×480; a square OOD source would be misrouted to the rules. The E4 descriptors (fps/profile/quality) do not separate the regimes, so a deployable router needs a better in-domain detector than the descriptors currently cached.

- **OOD quality transfer is untestable** with the current labels: 0/22 OOD bad windows are quality failures. The 'quality transfers' half of the hypothesis cannot be confirmed or refuted OOD.

- **Small OOD bad count** (n_bad=22 over 47 factories) → wide grouped CIs; the adapted OOD AUROC CI is [0.60, 0.89].

- **Review proxy uses labels** (balanced Platt calibration, grouped CV) as a measurement; it is not a deployable review gate. The adapted score's calibration does not transfer (m/adapt160), so the OOD REVIEW number is descriptive.

- Fusion used cached, frozen scorers only; no new inference, no GPU, no label edits.


## Reproduce

```bash
OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc_fusion.py
python3 eval/exp_acc/make_figure.py
python3 eval/exp_acc/make_report.py
```

Cached binary inputs (`eval/results/exp_acc/inputs/ood_full.npz` from `m/phys-ready` `eval/results/exp_phys/inputs/`, and `z3_160.npz` from `m/adapt160` `eval/results/exp_adapt160/inputs/`) are `*.npz`-gitignored; the E4 descriptor `.jsonl` inputs are committed here so the router is reproducible. `tools/phys` is vendored from PR #144 (`m/phys-ready`) pending merge.
