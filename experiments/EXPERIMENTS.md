# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from the union of the pod worktrees' `experiments/registry.jsonl` files (append-only; the latest line per id wins). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E0.01 | E0 | done | maybe | Label persistence and position bias on IndEgo gold | IndEgo human gold, 150 windows, 7 clips (human labels) / 480p/10fps IndEgo |
| E1.01 | E1 | done | adopt | IndEgo pixel-feature fps sweep {10,5,3.33,2,1} | IndEgo gold 150 windows human labels 100good/50bad / 480p 10fps H264 in-domain |
| E1.02 | E1 | done | maybe | EG2 window-embedding fps subsample {2,1,0.5} + 5/10fps re-embed | IndEgo gold 150 windows human labels / 480p 10fps in-domain |
| E1.03 | E1 | running | - | OOD 1080p30 source fps sweep {30,15,10,5,3,2,1} | OOD labels.json 60 windows human 51good/9bad (+labels2 if pr / 1080p 30fps HEVC out-of-domain |
| E1.04 | E1 | running | - | Frame-selection strategies at fixed 2fps budget | IndEgo gold 150 human labels / 480p 10fps in-domain |
| E3.01 | E3 | done | adopt | Sparse probe vs full coverage on IndEgo clips (label oracle + OOF scores) | IndEgo human gold 150 windows/7 clips (human labels) + EG2/2 / 480p/10fps IndEgo assembly+logistics |
| E3.02 | E3 | done | drop | Position bias on OOD Egocentric-10K labelled windows | OOD labels.json 60 windows (51 good/9 bad, human) + queue so / 1080p/30fps Egocentric-10K factory |
| E3.03 | E3 | done | adopt | Early exit within a window: accuracy vs observed seconds (EG2 prefixes) | IndEgo human gold 150 windows; EG2 21-frame/2fps embeddings  / 480p/10fps IndEgo |
| E3.04 | E3 | done | adopt | Compute accounting: video-hours processed per footage-hour per policy | Egocentric-10K manifest 1546 windows/4.404 h (unlabelled) +  / 1080p/30fps Egocentric-10K factory |
| E3.05 | E3 | done | adopt | Sampling-policy decision rule per data regime | Synthesis of E3.01-E3.04 results JSONs / IndEgo + Egocentric-10K |
| E4.01 | E4 | running | - | Regime descriptors: IndEgo vs Egocentric-10K domain-shift diagnosis | 150 IndEgo gold windows (human) + 60 OOD labelled (human) +  / mixed: 480p/10fps H264 vs 1080p/30fps HEVC |
| E4.02 | E4 | running | - | Regime-specific vs global models within IndEgo | 150 IndEgo gold windows, human labels, leave-one-clip-out / terciles of sharpness/lighting/motion |
| E4.03 | E4 | running | - | Domain gap control: common-spec renormalisation | 60 OOD labelled windows, human labels; frozen ensemble basel / 480p/10fps transcode vs 1080p/30fps source |
| E4.04 | E4 | running | - | Profiles product layer: schema + examples + deterministic router | synthetic + E4.01 descriptor ranges / all |

## E0.01 — Label persistence and position bias on IndEgo gold
- **Why:** Test whether sparse window sampling or skipping clip starts is justified
- **Hypothesis:** Neighbouring windows share a verdict; early windows may be irrelevant
- **Data / label source:** IndEgo human gold, 150 windows, 7 clips (human labels)
- **Regime:** 480p/10fps IndEgo
- **Method:** nearest-neighbour label agreement; bad rate by position
- **Owner / started / finished:** orchestrator / 2026-10-10 19:17 / 2026-10-10 19:17
- **Verdict:** maybe — Neighbour agreement 0.73 vs 0.56 chance (n=143 pairs, ~15 s apart): sparse sampling plausible. Bad rate in first 120 s = 0.19 vs 0.42 later: skipping the start is NOT supported on IndEgo (the start is the cleaner part); must be tested per data type.
- **Decision:** Test change-point-driven sampling (probe sparsely, densify on change), not a fixed 'skip first N minutes' rule
- **Cost:** none (labels only)
- **Results:** 

## E1.01 — IndEgo pixel-feature fps sweep {10,5,3.33,2,1}
- **Why:** Builder: 30fps wastes compute, test 1-10fps; pixel features use all 100 frames and are decode/feature bound
- **Hypothesis:** AUROC stays within 0.02 of full down to 2fps; temporal-band features degrade first
- **Data / label source:** IndEgo gold 150 windows human labels 100good/50bad
- **Regime:** 480p 10fps H264 in-domain
- **Method:** decode once native 10fps, deterministic index subsample, cpu_window_features with true fps, LOCO logistic C=0.3 + clip-bootstrap CI
- **Owner / started / finished:** EXP-FPS / 2026-10-10 19:25 / 2026-10-10 19:32
- **Verdict:** adopt — pixel-view LOCO AUROC n=150: 10fps 0.809 [0.69,0.89], 5fps 0.793, 3.33fps 0.807, 2fps 0.787, 1fps 0.774; 3.33fps within 0.02 of full at 34 vs 100 frames
- **Decision:** pixel view: 3.33fps meets 0.02 rule; 2fps borderline (-0.022). Combine with EG2 + OOD before final recommendation
- **Cost:** decode 154ms/window + features 238/119/80/47/26ms at 10/5/3.33/2/1fps CPU
- **Results:** eval/results/exp_fps/e1_01.json

## E1.02 — EG2 window-embedding fps subsample {2,1,0.5} + 5/10fps re-embed
- **Why:** EG2 view already 2fps; test if fewer frames keep 0.85 and whether denser helps
- **Hypothesis:** 2fps near-optimal; 0.5fps drops >0.02
- **Data / label source:** IndEgo gold 150 windows human labels
- **Regime:** 480p 10fps in-domain
- **Method:** cached segments.npz subsample + GPU-lock re-embed for 5/10fps, LOCO C=0.01
- **Owner / started / finished:** EXP-FPS / 2026-10-10 19:25 / 2026-10-10 19:33
- **Verdict:** maybe — EG2 LOCO n=150: 2fps 0.848 [0.73,0.91], 1fps 0.860, 0.5fps 0.870 (CIs overlap; fewer frames NOT worse); two-view prob-avg 0.852-0.854
- **Decision:** EG2 needs no denser sampling; 5/10fps re-embed (E1.02b) only if GPU time remains; proceed to OOD + selection
- **Cost:** 0 GPU (cached embeds); eval CPU seconds
- **Results:** eval/results/exp_fps/e1_02.json

## E1.03 — OOD 1080p30 source fps sweep {30,15,10,5,3,2,1}
- **Why:** OOD regime is 1080p30 HEVC; fps answer may differ per regime; frozen models fail transfer so also label-free stability
- **Hypothesis:** frozen IndEgo model stays 0.4-0.53 at all fps; score rank-correlation across fps high
- **Data / label source:** OOD labels.json 60 windows human 51good/9bad (+labels2 if present)
- **Regime:** 1080p 30fps HEVC out-of-domain
- **Method:** decode original source_path at each fps, same features, frozen-model AUROC + rank stability + per-feature AUROC
- **Owner / started / finished:** EXP-FPS / 2026-10-10 19:25 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 

## E1.04 — Frame-selection strategies at fixed 2fps budget
- **Why:** uniform may waste frames on static stretches; motion-adaptive/keyframe could win at same cost
- **Hypothesis:** motion-adaptive >= uniform; keyframe-only worse (I-frames arbitrary); random control in between
- **Data / label source:** IndEgo gold 150 human labels
- **Regime:** 480p 10fps in-domain
- **Method:** 20-frame budget: uniform vs motion-adaptive vs keyframe-only vs random seed0, same LOCO protocol
- **Owner / started / finished:** EXP-FPS / 2026-10-10 19:25 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 

## E3.01 — Sparse probe vs full coverage on IndEgo clips (label oracle + OOF scores)
- **Why:** Builder wants sparse sampling / skip stretches; E0.01 showed verdicts persist (~15 s) but skip-the-start fails on IndEgo. Need measured error-vs-cost for probe-every-k, skip-first-M, random, change-point-driven policies.
- **Hypothesis:** Probing every 2nd-3rd window estimates clip bad-fraction within a few points at ~1/2-1/3 cost; skip-first-M biases the estimate; CUSUM-densified probing beats fixed-k at equal cost.
- **Data / label source:** IndEgo human gold 150 windows/7 clips (human labels) + EG2/26-feat LOCO OOF probs oof_probs.npz
- **Regime:** 480p/10fps IndEgo assembly+logistics
- **Method:** Per-clip subsampling simulation on gold-window order; label-oracle + OOF-score estimates; clip bad-fraction MAE, accept/reject @30% agreement, fraction processed, first-bad-stretch delay; LOCO for tuned CUSUM threshold
- **Owner / started / finished:** EXP-TEMPORAL / 2026-10-10 19:24 / 2026-10-10 19:27
- **Verdict:** adopt — every-2: clip-frac err 0.049 [0.019,0.087] (n=150/7 clips, label oracle), 7/7 verdicts @30%, delay 0.3 windows at 51% cost; every-3/4 err 0.117/0.124; skip-first-M err 0.088->0.418; random matches error but delay 8.9 windows; CUSUM-default never fires (=every-4), LOCO errM 0.214 ~ grid. Model err 0.167 at full coverage dominates.
- **Decision:** Adopt every-2nd-window grid for IndEgo-like data; drop CUSUM-as-sampler and skip-first-M; random rejected on latency
- **Cost:** simulated: 0.51 footage @every-2 (CPU-only sim, no GPU)
- **Results:** /workspace/exp-temporal/eval/results/exp_temporal/E3.01_policies.json

## E3.02 — Position bias on OOD Egocentric-10K labelled windows
- **Why:** Skip-the-start rule must be justified per data type; IndEgo says start is cleaner. Test on OOD before any skip rule ships for factory footage.
- **Hypothesis:** Position in the 20-min Egocentric-10K clip does not predict bad windows (as on IndEgo).
- **Data / label source:** OOD labels.json 60 windows (51 good/9 bad, human) + queue source_start_s + manifest durations; labels2 if present
- **Regime:** 1080p/30fps Egocentric-10K factory
- **Method:** Logistic bad~position (raw + normalised) and Mann-Whitney bad vs good positions
- **Owner / started / finished:** EXP-TEMPORAL / 2026-10-10 19:24 / 2026-10-10 19:27
- **Verdict:** drop — No position effect on OOD: MW bad-vs-good p=0.42 raw / 0.47 normalised (n=60, n_bad=9); medians 70s vs 90s. Skip-the-start NOT justified for Ego10K either, but n_bad=9 is thin.
- **Decision:** No skip rule for either regime; re-run position test when labels2 reaches n_bad~24
- **Cost:** none (labels+manifest arithmetic)
- **Results:** /workspace/exp-temporal/eval/results/exp_temporal/E3.02_ood_position.json

## E3.03 — Early exit within a window: accuracy vs observed seconds (EG2 prefixes)
- **Why:** If 2-4 s of a 10 s window suffices, window cost drops 2-5x multiplicatively with sparse probing.
- **Hypothesis:** AUROC saturates well before 10 s (diminishing returns after ~4-6 s).
- **Data / label source:** IndEgo human gold 150 windows; EG2 21-frame/2fps embeddings segments.npz
- **Regime:** 480p/10fps IndEgo
- **Method:** LOCO logistic (StandardScaler, C=0.01 balanced) on L2-normalised mean of first 4/8/12/16/21 frames; AUROC + accuracy vs seconds
- **Owner / started / finished:** EXP-TEMPORAL / 2026-10-10 19:24 / 2026-10-10 19:27
- **Verdict:** adopt — EG2-prefix LOCO AUROC (n=150): 2s 0.804, 4s 0.860, 6s 0.869, 8s 0.857, 10.5s 0.848. Saturates at 4-6s; 4s within 0.01 of peak.
- **Decision:** Adopt observe-first-4s early exit (IndEgo); never below 4s; OOD curve needs a working OOD scorer
- **Cost:** 2.5x window-cost cut (4s/10s); CPU LOCO seconds
- **Results:** /workspace/exp-temporal/eval/results/exp_temporal/E3.03_early_exit.json

## E3.04 — Compute accounting: video-hours processed per footage-hour per policy
- **Why:** E7/dashboard needs hours-processed-per-hour and speed-up per sampling policy on the 4.4 h corpus.
- **Hypothesis:** Sparse probing + early exit combine multiplicatively to <1/4 footage processing.
- **Data / label source:** Egocentric-10K manifest 1546 windows/4.404 h (unlabelled) + policy fractions from E3.01/E3.03
- **Regime:** 1080p/30fps Egocentric-10K factory
- **Method:** Arithmetic: fraction-windows-processed x within-window-seconds-fraction -> processed hours per footage hour + speed-up
- **Owner / started / finished:** EXP-TEMPORAL / 2026-10-10 19:24 / 2026-10-10 19:27
- **Verdict:** adopt — 4.404h corpus: every-2 -> 2.27h (x1.9); every-2 + 4s exit -> ~0.9h (x4.9); every-3 + 4s -> x7.2 (decode-bound upper bound).
- **Decision:** Feed x4.9 adopted / x7.2 budget numbers to E7 dashboard
- **Cost:** arithmetic, no compute
- **Results:** /workspace/exp-temporal/eval/results/exp_temporal/E3.04_compute.json

## E3.05 — Sampling-policy decision rule per data regime
- **Why:** Track must output one adopted policy: fewest windows within tolerance on clip-level error, per regime, with thin-evidence caveats.
- **Hypothesis:** Different regimes need different policies (IndEgo: sparse probe ok, no skip; Ego10K: unknown, n_bad=9).
- **Data / label source:** Synthesis of E3.01-E3.04 results JSONs
- **Regime:** IndEgo + Egocentric-10K
- **Method:** Tolerance-based selection on clip-level error; per-regime statement; evidence-thinness audit
- **Owner / started / finished:** EXP-TEMPORAL / 2026-10-10 19:24 / 2026-10-10 19:27
- **Verdict:** adopt — Adopted IndEgo: every-2nd + 4s exit (~0.20 footage, x5). Dropped: CUSUM-sampler, skip-first-M, random grids. OOD: no certified policy (transfer fails; n_bad=9). Sampling not the bottleneck (model err 0.167 full coverage). Thin: 7 clips, sparse gold, IndEgo-only exit curve.
- **Decision:** Freeze E3 policy; unblock on E5 scorer for OOD; re-test position at n_bad~24
- **Cost:** synthesis only
- **Results:** /workspace/exp-temporal/eval/results/exp_temporal/E3.05_decision.json

## E4.01 — Regime descriptors: IndEgo vs Egocentric-10K domain-shift diagnosis
- **Why:** Every IndEgo-trained scorer failed on real Egocentric-10K windows (0.4-0.53); need to show which deterministic descriptors separate the datasets
- **Hypothesis:** fps, resolution, sharpness/noise and ego-motion descriptors separate IndEgo from Egocentric-10K
- **Data / label source:** 150 IndEgo gold windows (human) + 60 OOD labelled (human) + ~300 stratified Ego10K sample (unlabelled)
- **Regime:** mixed: 480p/10fps H264 vs 1080p/30fps HEVC
- **Method:** deterministic per-window descriptors (ffprobe + sampled-frame stats, no GPU)
- **Owner / started / finished:** exp-regimes / 2026-10-10 19:24 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 

## E4.02 — Regime-specific vs global models within IndEgo
- **Why:** Decide whether blurry/low-light/walking splits deserve their own models or thresholds
- **Hypothesis:** Regime-specific models beat global only where regimes have both classes
- **Data / label source:** 150 IndEgo gold windows, human labels, leave-one-clip-out
- **Regime:** terciles of sharpness/lighting/motion
- **Method:** logistic on 26 feats and EG2 embedding, global vs regime-specific, nested LOCO
- **Owner / started / finished:** exp-regimes / 2026-10-10 19:24 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 

## E4.03 — Domain gap control: common-spec renormalisation
- **Why:** Test whether matching input spec (480p/10fps) fixes OOD transfer
- **Hypothesis:** Re-normalising 1080p30 to 480p10 shrinks descriptor gap but may not fix score distribution
- **Data / label source:** 60 OOD labelled windows, human labels; frozen ensemble baseline ood_scores.json
- **Regime:** 480p/10fps transcode vs 1080p/30fps source
- **Method:** descriptor gap before/after + frozen-ensemble OOD score comparison
- **Owner / started / finished:** exp-regimes / 2026-10-10 19:24 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 

## E4.04 — Profiles product layer: schema + examples + deterministic router
- **Why:** Builder requires customisable product adapting to whatever data is used
- **Hypothesis:** Rule-based router on descriptors selects profile with margin, falls back safely
- **Data / label source:** synthetic + E4.01 descriptor ranges
- **Regime:** all
- **Method:** profiles/schema.json, 4 example profiles, jadal/regimes/router.py, unit tests
- **Owner / started / finished:** exp-regimes / 2026-10-10 19:24 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 
