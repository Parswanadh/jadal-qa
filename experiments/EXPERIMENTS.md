# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E5.00 | E5 | done | maybe | Frozen ensemble OOD prior (usable_prob AUROC on 60 human OOD windows) | 60 Egocentric-10K human-labelled windows (51 good / 9 bad, s / 480p 10s transcodes of 1080p/30fps factory clips; ~1 window/factory |
| E5.01 | E5 | done | maybe | In-domain EG2 frame heads (10k AI FACTORY frames) pooled to IndEgo human windows | Egocentric-10K-Evaluation 10k AI-labelled FACTORY frames (ha / head frames: 1080p/30fps factory stills; IndEgo eval: 480p ~10fps H.264, mixed lighting/sharpness |
| E5.02 | E5 | done | maybe | Few-shot adaptation on human OOD labels (factory-grouped CV) | 60 Egocentric-10K human-labelled windows (51 good / 9 bad, s / 480p 10s transcodes of 1080p/30fps factory clips; ~47 factories, ~1 window/factory |
| E5.03 | E5 | done | maybe | Per-feature signed AUROC IndEgo gold vs Egocentric-10K OOD (the what-transfers table) | 150 IndEgo human gold windows (manifest_gold.json) + 60 Egoc / IndEgo 480p ~10fps vs Egocentric-10K 1080p/30fps-source, both decoded 160x90; IndEgo native 10fps vs OOD subsampled to 10fps effective |
| E5.04 | E5 | done | inconclusive | Frozen ensemble re-score on 160 OOD windows (labels2 pending -> inconclusive) | 160 Egocentric-10K windows (60 in caches/ensemble/ood_scores / 480p 10s transcodes of 1080p/30fps factory clips |

## E5.00 — Frozen ensemble OOD prior (usable_prob AUROC on 60 human OOD windows)
- **Why:** The E5 report quotes the frozen-ensemble out-of-domain AUROC 0.415 [0.210,0.647], but that number lived only in docs/reports/ensemble_views.md on main / PR #79, not in this branch or in a committed results JSON. Reproduce it from the frozen bundle scores so every E5 number traces to a committed file.
- **Hypothesis:** The frozen IndEgo ensemble scores ~0.415 AUROC on the 60 human-labelled Egocentric-10K windows (exploratory, n_bad=9); reproducing it from the frozen scores gives the same headline.
- **Data / label source:** 60 Egocentric-10K human-labelled windows (51 good / 9 bad, single labeller, ood_labels.json) scored by frozen ensemble_v1 (bundle caches/ensemble/ood_scores.json); human labels
- **Regime:** 480p 10s transcodes of 1080p/30fps factory clips; ~1 window/factory
- **Method:** load frozen usable_prob; AUROC vs human labels; window-bootstrap 95% CI (optimistic)
- **Owner / started / finished:** experimenter / 2026-10-10 16:12 / 2026-10-10 16:20
- **Verdict:** maybe — Frozen ensemble_v1 usable_prob vs 60 human OOD windows (51 good/9 bad, exploratory): AUROC 0.4150, window-bootstrap CI [0.201,0.656] (optimistic); balacc@0.5 0.471. Reproduces the 0.415 headline previously cited from docs/reports/ensemble_views.md (main/PR #79, NOT on this branch) from $JADAL_BUNDLE/caches/ensemble/ood_scores.json.
- **Decision:** Cite E5.00.json for the frozen-OOD prior in the E5 report; keep the PR-#79 note for the earlier CI [0.210,0.647].
- **Cost:** CPU <1 s; 60 frozen scores; no GPU, no encode
- **Results:** eval/results/exp_domain/E5.00.json

## E5.01 — In-domain EG2 frame heads (10k AI FACTORY frames) pooled to IndEgo human windows
- **Why:** All IndEgo-trained models failed out-of-domain on Egocentric-10K (0.4-0.53 AUROC). Before adapting, measure how much window-level signal a head trained on in-domain (factory, AI-labelled) frames carries on the 150 human-labelled IndEgo windows.
- **Hypothesis:** Frame heads trained on the 10k AI-labelled FACTORY frames separate IndEgo usable/bad windows when pooled (LOCO AUROC > 0.5), but this is model-vs-model agreement, NOT ground-truth accuracy.
- **Data / label source:** Egocentric-10K-Evaluation 10k AI-labelled FACTORY frames (hand_count, active_labor; AI, gemini-2.5-flash, NOT human) + 150 IndEgo human gold windows (manifest_gold.json, single labeller)
- **Regime:** head frames: 1080p/30fps factory stills; IndEgo eval: 480p ~10fps H.264, mixed lighting/sharpness
- **Method:** EG2 frame embeddings -> logistic heads (5-fold CV, C=1) -> pool per-window (mean/max/p25/p75/frac>0.5/var) -> LOCO over 7 clips + clip-bootstrap CI
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 16:20
- **Verdict:** maybe — Pooled AI-frame heads on 150 IndEgo human windows (LOCO over 7 clips, clip-bootstrap CI): work-prob mean 0.813 [0.695,0.864]; hands-prob mean 0.818 [0.685,0.893]; both 0.826 [0.717,0.881]; 12-feat LOCO logistic 0.789 [0.686,0.862]. Below the 0.85 in-domain EG2-window-embedding baseline. Frame-level 0.911/0.957 is AI-vs-AI (model-vs-model, NOT ground-truth); the window-level 0.789-0.826 IS scored against the human IndEgo gold labels.
- **Decision:** Keep pooled AI-frame heads as a cheap prior only; they do not beat the in-domain EG2 embedding baseline. Do not claim transfer/ground truth.
- **Cost:** CPU ~2 min; 10k AI frames + 150 windows; no GPU encode
- **Results:** eval/results/exp_domain/E5.01.json

## E5.02 — Few-shot adaptation on human OOD labels (factory-grouped CV)
- **Why:** If the frozen IndEgo models do not transfer (0.415 OOD AUROC), the next question is whether a handful of human OOD labels can recover usable signal, and how many labels are needed.
- **Hypothesis:** With n_bad=9 and factory-grouped folds the OOD few-shot logistic is unstable and largely indistinguishable from chance; simple recipes will not beat chance reliably.
- **Data / label source:** 60 Egocentric-10K human-labelled windows (51 good / 9 bad, single labeller, ood_labels.json) + frozen caches: EG2 OOD frame embeddings (60x20x768) and 26 per-pixel features per window; labels2 NOT delivered
- **Regime:** 480p 10s transcodes of 1080p/30fps factory clips; ~47 factories, ~1 window/factory
- **Method:** grouped/leave-one-factory-out CV (StratifiedGroupKFold); recipes: logistic on 26 per-pixel features, logistic on EG2 window means, two-view probability average; report smallest n_bad in any train fold
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 16:20
- **Verdict:** maybe — 60 OOD windows (51 good / 9 bad, 47 factories), factory-grouped CV: pixel-26 0.756 [0.575,0.902] balacc 0.634; EG2-window-mean 0.546 [0.350,0.736] balacc 0.520; two-view avg 0.767 [0.596,0.904] balacc 0.614. Learning curve (pixel-26, C=1 balanced): ~12 -> 0.709 (std 0.10, 17 repeats, min_n_bad_train 1), ~24 -> 0.715 (20), ~36 -> 0.703 (20), ~45 -> 0.691 (19). Smallest n_bad in any training fold = 7. Estimates unstable (n_bad=9); grouped folds ~1 window/factory so CV is optimistic.
- **Decision:** A ~12-window OOD adaptation set can recover ~0.7 AUROC on THIS set (exploratory); needs the 100-window batch + labels2 before any rate is quoted.
- **Cost:** CPU ; 60 windows; 26 features / 768 EG2 means
- **Results:** eval/results/exp_domain/E5.02.json

## E5.03 — Per-feature signed AUROC IndEgo gold vs Egocentric-10K OOD (the what-transfers table)
- **Why:** The domain gap is per-feature; a table of which of the 26 per-pixel features keep their sign from IndEgo to Egocentric-10K tells us what transfers and what flips.
- **Hypothesis:** Sharpness/light/duplicate features keep their sign; motion and temporal-band features flip between IndEgo (480p ~10fps) and Egocentric-10K (1080p/30fps source).
- **Data / label source:** 150 IndEgo human gold windows (manifest_gold.json) + 60 Egocentric-10K human-labelled windows (51 good / 9 bad, ood_labels.json)
- **Regime:** IndEgo 480p ~10fps vs Egocentric-10K 1080p/30fps-source, both decoded 160x90; IndEgo native 10fps vs OOD subsampled to 10fps effective
- **Method:** signed AUROC per feature (label1=usable); clip-bootstrap CI (IndEgo, 7 clips) and window-bootstrap CI (OOD, optimistic given ~1 window/factory); classify stable/flip/chance
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 16:20
- **Verdict:** maybe — 26 per-pixel features, signed AUROC IndEgo(gold,n=150,clip-CI) vs OOD(n=60,n_bad=9,window-CI,10fps-effective): 8 stable, 14 flip, 4 chance. Rule: chance iff BOTH |AUROC-0.5|<0.05, else stable/flip by sign. Only feature-sign stability was measured: the sharpness/spatial-spectrum trio (radial_slope 0.633/0.645, hf_ratio 0.618/0.693, dct_energy 0.631/0.702) is sign-stable but still weak out-of-domain (AUROC 0.64-0.70). Illumination does not hold as a group (illum_g flips; illum_r is stable only at chance level). Native-30 class stored per row: 3/26 disagree with the 10fps class (temp_b3, flicker_idx chance->flip; dark_frac chance->stable).
- **Decision:** Use the sign-stable features only as weak sign priors for regime routing; do not reuse motion/temporal-band or illumination signs across domains.
- **Cost:** CPU ; 150+60 windows x 26 features
- **Results:** eval/results/exp_domain/E5.03.json

## E5.04 — Frozen ensemble re-score on 160 OOD windows (labels2 pending -> inconclusive)
- **Why:** The 100-window second batch exists as mp4s with frozen scores cached, but the HUMAN labels ood_labels2.json are not delivered. Script the confirmatory eval to run automatically when the file appears.
- **Hypothesis:** With labels2 absent the confirmatory OOD eval stays inconclusive; only descriptive score distributions over the 160 windows can be reported (no accuracy).
- **Data / label source:** 160 Egocentric-10K windows (60 in caches/ensemble/ood_scores.json, 100 in ood_scores2.json) scored by frozen ensemble_v1; ood_labels2.json MISSING (n human labels = 0)
- **Regime:** 480p 10s transcodes of 1080p/30fps factory clips
- **Method:** load frozen scores (or re-score via jadal.ensemble.scorer when missing); when ood_labels2.json appears, merge + eval_ood-style window-bootstrap AUROC; until then descriptive distributions only
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 16:20
- **Verdict:** inconclusive — ood_labels2.json NOT delivered: the 160 scored windows have NO labels; result JSON records labels_missing=true, n=0, ci95=n/a (no accuracy computed). Descriptive only: 160 windows scored with frozen ensemble_v1, usable_prob mean 0.560 (median 0.460), accept 71 / review 35 / reject 54; batch1(60) mean 0.496, batch2(100) mean 0.598. Config: frozen sha 4da1068a recorded by the scorer; local raw sha d62a95a differs only by pod path remap; canonical sha f43654d7 matches and is asserted (observed shas stored in E5.04.json).
- **Decision:** Watcher eval/exp_domain/watch_e5_04.sh re-scores and runs the confirmatory window-bootstrap eval automatically when ood_labels2.json appears.
- **Cost:** 0 new GPU scoring (frozen scores cached); descriptive only
- **Results:** eval/results/exp_domain/E5.04.json
