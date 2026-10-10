# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E0.01 | E0 | done | maybe | Label persistence and position bias on IndEgo gold | IndEgo human gold, 150 windows, 7 clips (human labels) / 480p/10fps IndEgo |
| E1.01 | E1 | done | adopt | IndEgo pixel-feature fps sweep {10,5,3.33,2,1} | IndEgo gold 150 windows human labels 100good/50bad / 480p 10fps H264 in-domain |
| E1.02 | E1 | done | maybe | EG2 window-embedding fps subsample {2,1,0.5} + 5/10fps re-embed | IndEgo gold 150 windows human labels / 480p 10fps in-domain |
| E1.03 | E1 | running | - | OOD 1080p30 source fps sweep {30,15,10,5,3,2,1} | OOD labels.json 60 windows human 51good/9bad (+labels2 if pr / 1080p 30fps HEVC out-of-domain |
| E1.04 | E1 | running | - | Frame-selection strategies at fixed 2fps budget | IndEgo gold 150 human labels / 480p 10fps in-domain |
| E2.01 | E2 | done | drop | Mosaic embedding (adopt candidate) | IndEgo gold 150 human windows (LOCO, 7 clips) + OOD 60 human / 480p@10fps IndEgo, 854x480@30fps Egocentric-10K; 10-s windows |
| E2.02 | E2 | done | adopt | Mean-of-frames control (matched frames) | same windows/folds as E2.01; label_source human / same as E2.01 |
| E2.03 | E2 | done | drop | Shuffled tile order | same windows/folds; label_source human / same as E2.01 |
| E2.04 | E2 | done | drop | Mosaic without timestamps | same windows/folds; label_source human / same as E2.01 |
| E2.05 | E2 | done | maybe | Single middle frame | same windows/folds; label_source human / same as E2.01 |
| E2.06 | E2 | done | maybe | Motion mosaic (|I_t - I_{t-1}| tiles) | same windows/folds; label_source human / same as E2.01 |
| E2.07 | E2 | done | drop | Pooled per-tile statistics control (no layout) | same windows/folds; label_source human / same as E2.01 |
| E3.01 | E3 | done | adopt | Sparse probe vs full coverage on IndEgo clips (label oracle + OOF scores) | IndEgo human gold 150 windows/7 clips (human labels) + EG2/2 / 480p/10fps IndEgo assembly+logistics |
| E3.02 | E3 | done | drop | Position bias on OOD Egocentric-10K labelled windows | OOD labels.json 60 windows (51 good/9 bad, human) + queue so / 1080p/30fps Egocentric-10K factory |
| E3.03 | E3 | done | adopt | Early exit within a window: accuracy vs observed seconds (EG2 prefixes) | IndEgo human gold 150 windows; EG2 21-frame/2fps embeddings  / 480p/10fps IndEgo |
| E3.04 | E3 | done | adopt | Compute accounting: video-hours processed per footage-hour per policy | Egocentric-10K manifest 1546 windows/4.404 h (unlabelled) +  / 1080p/30fps Egocentric-10K factory |
| E3.05 | E3 | done | adopt | Sampling-policy decision rule per data regime | Synthesis of E3.01-E3.04 results JSONs / IndEgo + Egocentric-10K |
| E4.01 | E4 | running | - | Regime descriptors: IndEgo vs Egocentric-10K domain-shift diagnosis | 150 IndEgo gold windows (human) + 60 OOD labelled (human) +  / mixed: 480p/10fps H264 vs 1080p/30fps HEVC |
| E4.02 | E4 | running | - | Regime-specific vs global models within IndEgo | 150 IndEgo gold windows, human labels, leave-one-clip-out / terciles of sharpness/lighting/motion |
| E4.03 | E4 | running | - | Domain gap control: common-spec renormalisation | 60 OOD labelled windows, human labels; frozen ensemble basel / 480p/10fps transcode vs 1080p/30fps source |
| E4.04 | E4 | running | - | Profiles product layer: schema + examples + deterministic router | synthetic + E4.01 descriptor ranges / all |
| E5.01 | E5 | done | maybe | In-domain EG2 frame heads (10k AI FACTORY frames) pooled to IndEgo human windows | Egocentric-10K-Evaluation 10k AI-labelled FACTORY frames (ha / head frames: 1080p/30fps factory stills; IndEgo eval: 480p ~10fps H.264, mixed lighting/sharpness |
| E5.02 | E5 | done | maybe | Few-shot adaptation on human OOD labels (factory-grouped CV) | 60 Egocentric-10K human-labelled windows (51 good / 9 bad, s / 480p 10s transcodes of 1080p/30fps factory clips; ~47 factories, ~1 window/factory |
| E5.03 | E5 | done | maybe | Per-feature signed AUROC IndEgo gold vs Egocentric-10K OOD (the what-transfers table) | 150 IndEgo human gold windows (manifest_gold.json) + 60 Egoc / IndEgo 480p ~10fps vs Egocentric-10K 1080p/30fps-source, both decoded 160x90; IndEgo native 10fps vs OOD subsampled to 10fps effective |
| E5.04 | E5 | done | inconclusive | Frozen ensemble re-score on 160 OOD windows (labels2 pending -> inconclusive) | 160 Egocentric-10K windows (60 in caches/ensemble/ood_scores / 480p 10s transcodes of 1080p/30fps factory clips |
| E6.01 | E6 | done | drop | Motion cue conditioning on ego-motion magnitude (terciles + Spearman partial) | IndEgo human gold, 150 windows / 7 clips (one labeller); cue / 480p/10fps IndEgo mp4_480, ~100 frames per 10 s window |
| E6.02 | E6 | done | drop | Per-clip sign calibration from unlabelled windows only (LOCO) | IndEgo human gold, 150 windows / 7 clips (one labeller) for  / 480p/10fps IndEgo mp4_480 |
| E6.03 | E6 | done | drop | Motion cue out-of-domain on 60 human-labelled Egocentric-10K windows | Egocentric-10K queues/label_queue_random.json 480p/30fps tra / 480p/30fps transcode of 1080p/30fps HEVC factory clips, stride 3 -> ~10 fps effective |
| E6.04 | E6 | done | drop | Motion cue conditioning robustness (quantiles vs fixed bins) + logistic controls | IndEgo human gold, 150 windows / 7 clips (one labeller), sam / 480p/10fps IndEgo mp4_480 |
| E7.01 | E7 | done | adopt | Cost model: hours of video per wall-hour on the pod A40, per profile | n/a (cost model): measured timings on the pod A40 from this  / 480p/10fps H.264 IndEgo + 1080p/30fps HEVC Egocentric-10K |
| E8.01 | E8 | running | - | E8.01 re-score 1,546 Egocentric-10K windows with the frozen two-view recipe + profile router | queues/manifest_scale.json: 1546 windows / 58 clips / 4.404  / factory Egocentric-10K, 1080p, 30 fps, HEVC, handheld egocentric manufacturing |
| E9.01 |  | done | adopt |  |  /  |
| LB1 | B | done | adopt | Independent verification of pod baselines, E1.01-E1.02 fps sweeps and E3.01-E3.05 temporal policy | IndEgo human gold 150 windows / 7 clips (human labels, READ- / 480p/10fps IndEgo in-domain LOCO (+ OOD position test, labels only) |
| LB2 | B | done | adopt | PoC integration: adopted every-2nd window grid + 4s early exit + EG2 2fps + 26-feature pixel logistic (descriptive verdicts) |  / CPU only (GPU unavailable in the container); cached features; deterministic (fixed seeds) |
| LB4 | B | pending | inconclusive | Stage frozen pre-registered OOD run (one command, sha256-guarded) | labels/ood_labels2.json (100 human-labelled Egocentric-10K w /  |
| LB5 | B | done | maybe | Branch triage & merge queue |  /  |

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

## E2.01 — Mosaic embedding (adopt candidate)
- **Why:** Test if ONE tiled timestamped image can replace N sequential frame evaluations, same accuracy at <=1/4 cost.
- **Hypothesis:** Mosaic embedding is within 0.02 AUROC of mean-of-frames at N frames with 1/N the embedding passes.
- **Data / label source:** IndEgo gold 150 human windows (LOCO, 7 clips) + OOD 60 human windows (51 good/9 bad); label_source human
- **Regime:** 480p@10fps IndEgo, 854x480@30fps Egocentric-10K; 10-s windows
- **Method:** EmbeddingGemma-2 image embedding of mosaic, logistic C=0.01 LOCO, clip-bootstrap CI; N in {4,6,9,12,16} tile 224
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** drop — IndEgo LOCO (n=150, 7 clips) mosaic AUROC 0.757-0.842 over N=4..100, always below matched mean-of-frames (0.822-0.860); deltas -0.001..-0.214 (only N30 within 0.02 but OOD 0.155). OOD (n=60, n_bad=9): mosaic 0.155-0.638 vs mean 0.420-0.627, inconsistent. Fails the within-0.02-on-BOTH-regimes bar -> NOT adopted as model input; adopted ONLY as the human-evidence artefact (dashboard why-view). 95% CIs clip-bootstrap, optimistic.
- **Decision:** Do not use mosaic as model input; ship mosaic PNG+sidecar as the reviewer 'why' artefact.
- **Cost:** 1 embed pass/window vs N for mean-of-frames (<=1/4 for N>=4); 224px tiles; build 1.6s/window, embed 1 pass; A40 VRAM peak 3967 MB
- **Results:** eval/results/exp_mosaic/E2.01.json

## E2.02 — Mean-of-frames control (matched frames)
- **Why:** Control: is the mosaic better/worse than the plain average of the SAME N frame embeddings; matched folds.
- **Hypothesis:** Mean-of-frames is the accuracy baseline the mosaic must match within 0.02 AUROC.
- **Data / label source:** same windows/folds as E2.01; label_source human
- **Regime:** same as E2.01
- **Method:** mean of per-frame EmbeddingGemma-2 embeddings (same decoded frames), logistic LOCO
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** adopt — mean-of-frames control on the SAME decoded frames: IndEgo LOCO 0.822-0.860 over N=4..100 (N9 0.852 [0.75,0.91]; N16 0.860 [0.76,0.92]) reproduces the cached EG2 window-embedding reference 0.848 [0.73,0.91]. OOD 0.420-0.627. Confirms the incumbent model-input baseline under the matched folds. n=150 (one labeller) + 60 OOD (n_bad=9, optimistic).
- **Decision:** Keep mean-of-frames EG2 as the model-input baseline; no change.
- **Cost:** N embed passes/window (N frames decoded/mosaic); build ms included; A40
- **Results:** eval/results/exp_mosaic/E2.02.json

## E2.03 — Shuffled tile order
- **Why:** Does temporal/spatial layout carry signal, or is it only the bag of tiles?
- **Hypothesis:** Shuffling tile placement costs little if content dominates; a drop means layout matters.
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** mosaic N=9 tile 224 with deterministic placement permutation (seed 1), same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** drop — shuffled tile placement IndEgo 0.801 [0.64,0.89] vs ordered mosaic N9 0.785 [0.64,0.86]: delta +0.016, inside the CI; OOD 0.423. Tile order does not carry load-bearing signal -> no reason to shuffle.
- **Decision:** Drop shuffled-order input.
- **Cost:** 1 embed pass/window; A40
- **Results:** eval/results/exp_mosaic/E2.03.json

## E2.04 — Mosaic without timestamps
- **Why:** Do the burned-in timestamps add signal beyond the pixels?
- **Hypothesis:** Removing timestamps does not change AUROC much if content dominates.
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** mosaic N=9 tile 224 timestamps=False, same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** drop — no-timestamp mosaic IndEgo 0.810 [0.68,0.88] vs with-timestamps 0.785: delta +0.025 inside CI; OOD 0.444. Burned-in timestamps do not add signal beyond pixels.
- **Decision:** Drop the timestamp-helps hypothesis; keep timestamps only for the human-evidence view.
- **Cost:** 1 embed pass/window; A40
- **Results:** eval/results/exp_mosaic/E2.04.json

## E2.05 — Single middle frame
- **Why:** Lower bound: one frame instead of N (cheapest input).
- **Hypothesis:** Single frame is clearly worse than N-tile mosaic and mean-of-frames.
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** 1x1 mosaic of the middle frame (t=5s), same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** maybe — single middle frame (1x1, t=5s) IndEgo 0.784 [0.69,0.85] ~= mosaic N9 0.785 [0.64,0.86] (1/9 the embedding cost) but 0.068 below mean-of-frames N9 0.852; OOD 0.514. Cheapest input but below the 0.02 bar to the baseline.
- **Decision:** Keep as a cost floor / dashboard thumbnail; not a model input.
- **Cost:** 1 embed pass/window; A40
- **Results:** eval/results/exp_mosaic/E2.05.json

## E2.06 — Motion mosaic (|I_t - I_{t-1}| tiles)
- **Why:** Does a motion-energy mosaic carry the usable-window cue (part may be head stability)?
- **Hypothesis:** Motion mosaic differs from intensity mosaic; may invert (good windows have less residual motion).
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** N=9 tiles of abs consecutive-frame difference, same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** maybe — motion mosaic |I_t-I_{t-1}| N=9 IndEgo 0.863 [0.78,0.91] is the best in-domain variant (mean-of-frames N9 0.852) but OOD 0.401 (fails transfer, like all IndEgo-trained models). Motion/stability cue again looks inverted/regime-specific; part may be head stability, not hand work.
- **Decision:** Maybe for in-domain regime profiles only; never claimed to transfer to Egocentric-10K.
- **Cost:** 1 embed pass/window; A40
- **Results:** eval/results/exp_mosaic/E2.06.json

## E2.07 — Pooled per-tile statistics control (no layout)
- **Why:** Control for E2.01/E2.03: does tiling layout/timestamps help beyond pooled per-tile stats?
- **Hypothesis:** concat(mean,std) of N per-tile embeddings is close to the mosaic (layout not load-bearing).
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** concat(mean,std) of N=9 per-tile EmbeddingGemma-2 embeddings, same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** drop — pooled per-tile stats concat(mean,std) of N=9 tile embeddings IndEgo 0.826 [0.69,0.88], OOD 0.586. Control confirms layout/timestamp are not load-bearing and it does not beat mean-of-frames 0.852.
- **Decision:** Drop as a method; record as the no-layout control.
- **Cost:** 9 embed passes/window; A40
- **Results:** eval/results/exp_mosaic/E2.07.json

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

## E5.01 — In-domain EG2 frame heads (10k AI FACTORY frames) pooled to IndEgo human windows
- **Why:** All IndEgo-trained models failed out-of-domain on Egocentric-10K (0.4-0.53 AUROC). Before adapting, measure how much window-level signal a head trained on in-domain (factory, AI-labelled) frames carries on the 150 human-labelled IndEgo windows.
- **Hypothesis:** Frame heads trained on the 10k AI-labelled FACTORY frames separate IndEgo usable/bad windows when pooled (LOCO AUROC > 0.5), but this is model-vs-model agreement, NOT ground-truth accuracy.
- **Data / label source:** Egocentric-10K-Evaluation 10k AI-labelled FACTORY frames (hand_count, active_labor; AI, gemini-2.5-flash, NOT human) + 150 IndEgo human gold windows (manifest_gold.json, single labeller)
- **Regime:** head frames: 1080p/30fps factory stills; IndEgo eval: 480p ~10fps H.264, mixed lighting/sharpness
- **Method:** EG2 frame embeddings -> logistic heads (5-fold CV, C=1) -> pool per-window (mean/max/p25/p75/frac>0.5/var) -> LOCO over 7 clips + clip-bootstrap CI
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 15:40
- **Verdict:** maybe — Pooled AI-frame heads on 150 IndEgo human windows (LOCO over 7 clips, clip-bootstrap CI): work-prob mean 0.813 [0.695,0.864]; hands-prob mean 0.818 [0.685,0.893]; both 0.826 [0.717,0.881]; 12-feat LOCO logistic 0.789 [0.686,0.862]. Below the 0.85 in-domain EG2-window-embedding baseline. Model-vs-model (AI head labels), NOT accuracy.
- **Decision:** Keep pooled AI-frame heads as a cheap prior only; they do not beat the in-domain EG2 embedding baseline. Do not claim transfer/ground truth.
- **Cost:** CPU ~2 min; 10k AI frames + 150 windows; no GPU encode
- **Results:** eval/results/exp_domain/E5.01.json

## E5.02 — Few-shot adaptation on human OOD labels (factory-grouped CV)
- **Why:** If the frozen IndEgo models do not transfer (0.415 OOD AUROC), the next question is whether a handful of human OOD labels can recover usable signal, and how many labels are needed.
- **Hypothesis:** With n_bad=9 and factory-grouped folds the OOD few-shot logistic is unstable and largely indistinguishable from chance; simple recipes will not beat chance reliably.
- **Data / label source:** 60 Egocentric-10K human-labelled windows (51 good / 9 bad, single labeller, ood_labels.json) + frozen caches: EG2 OOD frame embeddings (60x20x768) and 26 per-pixel features per window; labels2 NOT delivered
- **Regime:** 480p 10s transcodes of 1080p/30fps factory clips; ~47 factories, ~1 window/factory
- **Method:** grouped/leave-one-factory-out CV (StratifiedGroupKFold); recipes: logistic on 26 per-pixel features, logistic on EG2 window means, two-view probability average; report smallest n_bad in any train fold
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 15:40
- **Verdict:** maybe — 60 OOD windows (51 good / 9 bad, 47 factories), factory-grouped CV: pixel-26 0.756 [0.575,0.902]; EG2-window-mean 0.546 [0.350,0.736]; two-view avg 0.767 [0.596,0.904]. Learning curve: ~12 train windows already ~0.71 (std 0.10). Smallest n_bad in any training fold = 7. Estimates unstable (n_bad=9); grouped folds ~1 window/factory so CV is optimistic.
- **Decision:** A ~12-window OOD adaptation set can recover ~0.7 AUROC on THIS set (exploratory); needs the 100-window batch + labels2 before any rate is quoted.
- **Cost:** CPU ; 60 windows; 26 features / 768 EG2 means
- **Results:** eval/results/exp_domain/E5.02.json

## E5.03 — Per-feature signed AUROC IndEgo gold vs Egocentric-10K OOD (the what-transfers table)
- **Why:** The domain gap is per-feature; a table of which of the 26 per-pixel features keep their sign from IndEgo to Egocentric-10K tells us what transfers and what flips.
- **Hypothesis:** Sharpness/light/duplicate features keep their sign; motion and temporal-band features flip between IndEgo (480p ~10fps) and Egocentric-10K (1080p/30fps source).
- **Data / label source:** 150 IndEgo human gold windows (manifest_gold.json) + 60 Egocentric-10K human-labelled windows (51 good / 9 bad, ood_labels.json)
- **Regime:** IndEgo 480p ~10fps vs Egocentric-10K 1080p/30fps-source, both decoded 160x90; IndEgo native 10fps vs OOD subsampled to 10fps effective
- **Method:** signed AUROC per feature (label1=usable); clip-bootstrap CI (IndEgo, 7 clips) and window-bootstrap CI (OOD, optimistic given ~1 window/factory); classify stable/flip/chance
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 15:40
- **Verdict:** maybe — 26 per-pixel features, signed AUROC IndEgo(gold,n=150,clip-CI) vs OOD(n=60,n_bad=9,window-CI): 8 sign-stable (radial_slope 0.633/0.645, hf_ratio 0.618/0.693, dct_energy 0.631/0.702, luma_p05, global_shift, illum_r, illum_b, zncc_self); 14 flip (incl. motion_energy 0.308->0.721, temp_b0 0.534->0.240, skin_frac 0.667->0.257); 4 chance. Sharpness/light are the transferable cues.
- **Decision:** Use the stable sharpness/light features for regime routing; do not reuse motion/temporal-band signs across domains.
- **Cost:** CPU ; 150+60 windows x 26 features
- **Results:** eval/results/exp_domain/E5.03.json

## E5.04 — Frozen ensemble re-score on 160 OOD windows (labels2 pending -> inconclusive)
- **Why:** The 100-window second batch exists as mp4s with frozen scores cached, but the HUMAN labels ood_labels2.json are not delivered. Script the confirmatory eval to run automatically when the file appears.
- **Hypothesis:** With labels2 absent the confirmatory OOD eval stays inconclusive; only descriptive score distributions over the 160 windows can be reported (no accuracy).
- **Data / label source:** 160 Egocentric-10K windows (60 in caches/ensemble/ood_scores.json, 100 in ood_scores2.json) scored by frozen ensemble_v1; ood_labels2.json MISSING (n human labels = 0)
- **Regime:** 480p 10s transcodes of 1080p/30fps factory clips
- **Method:** load frozen scores (or re-score via jadal.ensemble.scorer when missing); when ood_labels2.json appears, merge + eval_ood-style window-bootstrap AUROC; until then descriptive distributions only
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 15:40
- **Verdict:** inconclusive — ood_labels2.json NOT delivered: result JSON records labels_missing=true, n=0, ci95=n/a. Descriptive only: 160 windows scored with frozen ensemble_v1, usable_prob mean 0.560 (median 0.460), accept 71 / review 35 / reject 54; batch1(60) mean 0.496, batch2(100) mean 0.598. No accuracy computed.
- **Decision:** Watcher eval/exp_domain/watch_e5_04.sh re-scores and runs the confirmatory window-bootstrap eval automatically when ood_labels2.json appears.
- **Cost:** 0 new GPU scoring (frozen scores cached); descriptive only
- **Results:** eval/results/exp_domain/E5.04.json

## E6.01 — Motion cue conditioning on ego-motion magnitude (terciles + Spearman partial)
- **Why:** The residual-motion cue is documented inverted (signed AUROC ~0.20). If it is only 'the wearer is walking', it is an ego-motion magnitude proxy and cannot separate usable windows within a motion level.
- **Hypothesis:** The residual-motion cue survives within ego-motion terciles and its Spearman partial correlation with the label controlling for shift_px stays substantially negative.
- **Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller); cue recomputed from video, ego-motion from caches/perpixel/series_gold.npz shift_px
- **Regime:** 480p/10fps IndEgo mp4_480, ~100 frames per 10 s window
- **Method:** per-window mean residual motion after translation-only phase-correlation compensation at 160x90; ego-motion = mean shift_px; within-tercile AUROC + Spearman partial correlation; clip-bootstrap CI (2000)
- **Owner / started / finished:** experimenter / 2026-10-10 15:21 / 2026-10-10 15:48
- **Verdict:** drop — n=150 IndEgo human windows (100 good/50 bad, 7 clips, one labeller). Strongest motion scalar = ego-motion magnitude (series_gold shift_px mean), oriented AUROC 0.805 [0.675,0.893] (unoriented 0.195). Residual-motion cue (translation-compensated, 160x90) oriented AUROC 0.692; Spearman partial correlation with the label controlling for ego-motion = -0.102 [-0.347,+0.072] (CI includes 0). Low-motion tercile is 49 good/1 bad, so the cue is largely 'the wearer is not walking'.
- **Decision:** Do not use the residual-motion cue as an independent usable-window signal; use ego-motion magnitude as a regime/stability descriptor in profiles, not as evidence of active work.
- **Cost:** CPU-only: 100 frames/window, 150 windows in 98.6 s (3 workers) = 657 ms/window, 0 MB VRAM; bootstraps 2000 draws; device A40 (GPU idle).
- **Results:** eval/results/exp_motion_cue/E6.01.json

## E6.02 — Per-clip sign calibration from unlabelled windows only (LOCO)
- **Why:** pallet is a per-clip sign inversion (raw 0.27) that puts some clips at chance. If the sign can be fixed at inference from the clip's own unlabelled windows, the cue becomes usable without labels.
- **Hypothesis:** A label-free per-clip sign/magnitude calibration (majority-prior sign or within-clip |z|) raises the pooled LOCO AUROC above the raw oriented cue.
- **Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller) for SCORING only; calibration uses only the clip's own unlabelled cue values
- **Regime:** 480p/10fps IndEgo mp4_480
- **Method:** leave-one-clip-out direction (6 training clips), per-clip label-free recalibration on the held-out clip's own window values (majority-prior sign; within-clip -|z|); pooled AUROC + clip-bootstrap CI
- **Owner / started / finished:** experimenter / 2026-10-10 15:21 / 2026-10-10 15:48
- **Verdict:** drop — n=150 IndEgo human windows, 7 clips, one labeller. LOCO with per-clip sign from the clip's OWN unlabelled windows: raw pooled 0.692 [0.477,0.839]; within-clip -|z| 0.606 [0.525,0.708]; majority-prior sign 0.610 [0.449,0.767]. Mean per-clip AUROC raw 0.664 [0.486,0.809] vs majority-prior 0.691 [0.535,0.808] (CIs overlap); pallet 0.36->0.64 and organise 0.31->0.69 repaired but desk 0.73->0.27 broken. Oracle per-clip sign (uses test labels, UPPER BOUND only) mean 0.757.
- **Decision:** The pallet inversion cannot be fixed at inference without labels: the label-free sign rule trades one broken clip for two repaired and lowers the pooled AUROC. No per-clip sign calibration in the product.
- **Cost:** no extra extraction (cached cue); 2000-draw clip bootstraps x 4 variants x 4 rules; device A40 (CPU).
- **Results:** eval/results/exp_motion_cue/E6.02.json

## E6.03 — Motion cue out-of-domain on 60 human-labelled Egocentric-10K windows
- **Why:** Test whether the inverted motion cue even points the right way outside IndEgo; the PoC cascade was at chance out of domain (0.529).
- **Hypothesis:** The IndEgo-frozen cue direction is at or near chance out of domain.
- **Data / label source:** Egocentric-10K queues/label_queue_random.json 480p/30fps transcodes, 60 windows human-labelled (51 good / 9 bad, one labeller; n_bad=9 -> EXPLORATORY)
- **Regime:** 480p/30fps transcode of 1080p/30fps HEVC factory clips, stride 3 -> ~10 fps effective
- **Method:** same cue extraction (160x90, translation-only compensation); direction frozen from IndEgo; pooled AUROC + window-bootstrap CI (optimistic)
- **Owner / started / finished:** experimenter / 2026-10-10 15:21 / 2026-10-10 15:48
- **Verdict:** drop — n=60 human-labelled Egocentric-10K windows (51 good/9 bad, one labeller; EXPLORATORY). Cue on the labelled 480p/30 fps transcode (stride 3 ~10 fps). IndEgo-frozen direction: resid_mean AUROC 0.281 [0.145,0.434] (window bootstrap, optimistic) - worse than chance; Spearman cue vs label +0.271, opposite sign to in-domain (-0.314). Post-hoc best orientation 0.719 (not usable).
- **Decision:** Do not apply the IndEgo motion-cue direction out of domain. Per-source sign calibration is not applicable (one labelled window per source).
- **Cost:** 60 windows, 95.6 s wall (3 workers) = 1593 ms/window, 0 MB VRAM; device A40 (CPU-only extraction).
- **Results:** eval/results/exp_motion_cue/E6.03.json

## E6.04 — Motion cue conditioning robustness (quantiles vs fixed bins) + logistic controls
- **Why:** The E6.01 conclusion must not depend on one arbitrary stratification; a vanishing effect must be logged as a negative result.
- **Hypothesis:** Within-stratum AUROC and the partial correlation are stable across quantile and fixed-bin strata, and cue+ego logistic does not beat ego alone.
- **Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller), same cue/ego vectors as E6.01
- **Regime:** 480p/10fps IndEgo mp4_480
- **Method:** same strata under 4 rules (terciles, quintiles, fixed 5/15 px, fixed 1/5/20 px) x 4 cue variants; LOCO logistic controls cue-only vs ego-only vs cue+ego
- **Owner / started / finished:** experimenter / 2026-10-10 15:21 / 2026-10-10 15:48
- **Verdict:** drop — n=150 IndEgo human windows, 7 clips. Within-stratum oriented AUROC of the residual cue under 4 stratification rules x 4 cue variants: interpretable strata (>=3 bad) span 0.503-0.756 (terciles 0.517/0.682; quintiles 0.503-0.756; fixed 1/5/20 px 0.569-0.806; fixed 5/15 px only 1 interpretable stratum). LOCO logistic control: cue only 0.660, ego-motion magnitude only 0.798, cue+ego 0.793.
- **Decision:** Negative result logged: the within-motion-level effect is weak and stratum-dependent, and the cue adds nothing on top of ego-motion magnitude (-0.005). Keep the cue only as a global stability descriptor.
- **Cost:** no extra extraction; 4 schemes x 4 variants x 2000-draw bootstraps; device A40 (CPU).
- **Results:** eval/results/exp_motion_cue/E6.04.json

## E7.01 — Cost model: hours of video per wall-hour on the pod A40, per profile
- **Why:** The product needs one honest cost table (fps x window sparsity x early exit x mosaic) computed only from measured numbers on THIS pod, so the demo/dashboard does not quote laptop or guessed throughput.
- **Hypothesis:** Sparse probing (every-2nd) and a 4 s early exit multiply, and with NVDEC decode the A40 stays feature/embedding-bound; hours-of-video per wall-hour should rise >= 4x vs the full-sampling profile.
- **Data / label source:** n/a (cost model): measured timings on the pod A40 from this run + source JSONs E1.01, E1.02, E1.04, E2.01, E3.04; profiles/*.json (5 profiles). No labels used.
- **Regime:** 480p/10fps H.264 IndEgo + 1080p/30fps HEVC Egocentric-10K
- **Method:** timed decode (CPU ffmpeg vs NVDEC cuvid), pixel26 CPU features, EmbeddingGemma-2 batch-1 image embeddings on real mosaic PNGs; arithmetic cost model per profile with stated assumptions
- **Owner / started / finished:** experimenter / 2026-10-10 15:39 / 2026-10-10 15:59
- **Verdict:** adopt — Cost table on A40 (contended pod): per-profile 0.51-15.22 video-h/wall-h; every-2nd+4s exit pixel26 565 ms/window -> 35.4 vs 9.1 full sampling (x3.88 wall-clock, mosaic-only x4.14); embedding-bound at batch 1 (183 ms/image, 47-94% of the window), decode-bound only for factory_1080p30 CPU (46%); NVDEC 7.37 vs 27.3 ms/frame on 1080p HEVC but slower on 480p (CI95 = window variation only, machine shared).
- **Decision:** Use this table as the pod cost reference; keep NVDEC(1080p HEVC)/CPU(480p) separate; re-time on an idle pod and measure a pipelined multi-stream run + batch-8 eg2 feature path; policy verdicts stay with E3.01/E3.03/E3.05 (no new accuracy here; 3.88x < the 4x cost-only bar).
- **Cost:** timed measurement ~9 min wall (decode 4 configs x 6 windows, features 12 windows, embedder 32 images, mosaic build 6 windows), peak 1.5 GB VRAM
- **Results:** eval/results/exp_cost/E7.01.json

## E8.01 — E8.01 re-score 1,546 Egocentric-10K windows with the frozen two-view recipe + profile router
- **Why:** B2/B4 PoC needs one contract JSONL over the whole 4.404 h scale corpus: per-window verdict, evidence op-threshold, features digest and content-hashed receipt, produced by the frozen in-domain recipe and the deterministic router
- **Hypothesis:** The frozen ensemble_v1 (EG2+pixel, equal-weight) + router produce a deterministic, reproducible contract file for all 1,546 windows; the router routes every 1920x1080/30 window to factory_1080p30; verdict shares are descriptive only (IndEgo-trained models did not transfer out-of-domain, 0.4-0.53 AUROC)
- **Data / label source:** queues/manifest_scale.json: 1546 windows / 58 clips / 4.404 h, 1080p30 HEVC; label_source none (no human labels)
- **Regime:** factory Egocentric-10K, 1080p, 30 fps, HEVC, handheld egocentric manufacturing
- **Method:** per window: ffmpeg/NVDEC decode -> (a) 160x90 rgb24 pixel26 features at native fps, (b) 512-side frames every 0.5 s for EmbeddingGemma-2 mean embedding, (c) 8 native frames for E4.01 regime descriptors; frozen logistic heads + equal-weight probability average; deterministic profile router choose_profile(descriptors); contract.jsonl with receipt = sha256(canonical JSON)
- **Owner / started / finished:** experimenter-e8-rescore / 2026-10-10 15:42 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 

## E9.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 15:23
- **Verdict:** adopt — Reproduces both documented baselines on n=150 IndEgo human windows (100 good/50 bad, 7 clips): 26 per-pixel features LOCO logistic C=0.3 -> AUROC 0.8100 (clip-bootstrap 95% CI 0.693-0.887); EG2 window embedding LOCO logistic C=0.01 -> 0.8480 (0.734-0.909). OOD (n=60, n_bad=9, window-bootstrap, optimistic): frozen IndEgo pixel26 model does NOT transfer, 0.4314 at fps=10 (E1.03 protocol); the features.json-trained variant saturates (all p=1.0, reported as 0.5).
- **Decision:** Use eval/standard/run_all.py as the single comparison number for PR-triggered review; exec tier runs it in a throwaway clone behind the run-tests label.
- **Cost:** CPU sklearn, no GPU, ~14-36 s, no network
- **Results:** eval/results/standard/run_all.json

## LB1 — Independent verification of pod baselines, E1.01-E1.02 fps sweeps and E3.01-E3.05 temporal policy
- **Why:** The adopted settings (every-2nd window, 4 s early exit, EG2 2 fps view, 26-feature logistic) drive the PoC and the submission; a second, independent re-run is the cheapest defence against a silent protocol bug.
- **Hypothesis:** Every published number reproduces within the published precision when re-run from pod/base with the cached features and read-only gold labels.
- **Data / label source:** IndEgo human gold 150 windows / 7 clips (human labels, READ-ONLY); caches/gpu/features.json; caches/frames10k/segments.npz; caches/ensemble/oof_probs.npz; OOD labels n=60 n_bad=9
- **Regime:** 480p/10fps IndEgo in-domain LOCO (+ OOD position test, labels only)
- **Method:** Re-run pinned scripts from pod/base; E1.02 additionally re-implemented from scratch; LOCO + clip-bootstrap 95% CI; exact-match tolerance at published precision
- **Owner / started / finished:** lead-b /  / —
- **Verdict:** adopt — 20/20 core claims reproduced EXACTLY from pod/base: pixel26 0.810, EG2 0.8480 [0.734,0.909], prob-avg 0.853, rank-avg 0.856; E1.01 10/5/3.33/2/1 fps 0.8086/0.7932/0.8068/0.7866/0.7736; E1.02 2/1/0.5 fps 0.8480/0.8596/0.8702 (plus an independent from-scratch re-implementation with identical point estimates); two-view 0.8524/0.8538; E3.01 every-2 clip-error 0.0486 [0.0189,0.0872], 7/7 verdicts, delay 0.286 at 0.515 of windows; E3.02 MW p=0.419/0.468 (n=60, n_bad=9); E3.03 2/4/6/8/10.5 s = 0.8038/0.8596/0.8694/0.8574/0.8480; E3.04 x1.94; E3.05 strict-pass ['every-2']. TWO FINDINGS FAILED VERIFICATION: (1) 14/316 SHA256SUMS.txt entries mismatch on disk, including the frozen scorer caches/ensemble/ensemble_v1.json (recorded 4da1068a..., on disk cf7fd7d3...) and labels/manifest_gold.json - cause is post-checksum path remapping (the file contains the pod path /workspace/jadal_pod_bundle/data/eg2); no canonical-JSON variant matches either; LB4 recovered the byte-exact frozen file from git blob 0ede3b83d93ce23dc5b68cd20fb7de0af703ad27. (2) OOD ATTRIBUTION FIXED: the published 0.529 is the PoC CASCADE and reproduces exactly (origin/main:eval/results/poc/ood_check_60.json -> 0.529 [0.316,0.728], 21/51 good rejected, 5/9 bad, bal.acc 0.513 vs published 0.51); the FROZEN ensemble_v1 is a different figure, 0.415 [0.207,0.630] with 26/51 good rejected and 3/9 bad (p_eg2 0.586, p_pixel 0.342). My first pass conflated the two scorers. No drift involved (shipped OOD file is byte-identical to origin/main, sha256 300c1eda...). Both scorers are at chance with heavy false rejects, so the finding the submission rests on is confirmed twice. 29 attempts recorded by verifier PR #100. Also: the frozen artifact hash IS recoverable and verifies (git blob 0ede3b83d93ce23dc5b68cd20fb7de0af703ad27 -> sha256 4da1068a..., independently re-checked by LB1). ADDITIONAL LIVE HAZARD: jadal.ensemble.scorer.load_config() raises ValueError on the DEFAULT config path (config sha256 mismatch d62a95a4... != recorded 4da1068a...) on pod/base and every branch tested, because the shipped config is the path-remapped file while the .sha256 still records the frozen hash; meanwhile the LB2 PoC loader (jadal/poc_stream/policy.py load_frozen_ensemble) does NOT check the sha256 at all and loads the remapped artifact silently. Strict loader fails on a recoverable situation, lenient loader succeeds on the wrong artifact. Fix: point the default path at the recovered hash-verified configs/frozen/ensemble_v1.prereg.json (LB4, PR #97) and make the PoC loader fail closed. LB2 C-sensitivity independently verified (PR #98): fused 4s AUROC 0.8736 at C_eg2=0.003 vs 0.8734 at published C_eg2=0.01, paired delta -0.0002 CI [-0.0133,+0.0095] => C choice is immaterial; do not frame it as an improvement over 0.848 (that is a full-window inner-OOF number).
- **Decision:** Adopt the verified in-domain stack (every-2nd window, 4 s exit, EG2 2 fps, 26-feature logistic). Quote the two OOD numbers separately and never merge them: 0.529 = PoC cascade, 0.415 = frozen ensemble_v1; both at chance. GPU feature-extraction half and the delivered OOD-labels half remain unverified (no CUDA in this container).
- **Cost:** CPU-only, ~6 min wall (2 shards E1.01 + cached-feature LOCO); 0 GPU
- **Results:** docs/reports/pod_verification.md + eval/results/pod_verify/ (baseline_pixel26_loco.json, baseline_ensemble_views.json, e1_01_fps_sweep.json, e1_02_eg2.json, ood_and_nested.json, bundle_checksum_audit.json, exp_temporal/E3.0{1..5}_*.json) + eval/results/pod_verify/ood_and_nested.json + bundle_checksum_audit.json

## LB2 — PoC integration: adopted every-2nd window grid + 4s early exit + EG2 2fps + 26-feature pixel logistic (descriptive verdicts)
- **Why:** E3.01/E3.05 adopted an IndEgo working policy (every-2nd window, 4s exit) but the PoC scorer path did not apply it; E1.01/E1.02 views were only measured standalone. This integrates the adopted settings in one policy module and regenerates per-window verdicts from the cached features, with LOCO numbers and DESCRIPTIVE labels only.
- **Hypothesis:** Applying the adopted settings to the cached scorer path reproduces the adopted view quality at 4s prefixes (EG2 ~0.86, pixel ~0.80, fused >= both) and yields a deterministic per-window accept/reject/uncertain stream with no OOD or accuracy claim.
- **Data / label source:** 
- **Regime:** CPU only (GPU unavailable in the container); cached features; deterministic (fixed seeds)
- **Method:** leave-one-clip-out over the 7 IndEgo clips: both logistic views re-fitted per fold at the 4s prefix (frozen ensemble_v1 C_eg2=0.003, C_px=0.3), equal-weight probability fusion, frozen verdict rule (>0.5 accept, <0.4 reject, else uncertain), every-2nd window grid per clip (range(0,n,2)), clip-bootstrap 95% CI (2000 resamples, seed 7)
- **Owner / started / finished:** lead-b /  / —
- **Verdict:** adopt — n=150 human gold IndEgo windows, LOCO, clip-bootstrap 95% CI. Fused 4s: AUROC 0.874 [0.742,0.942], balanced acc 0.740; EG2 2fps 4s 0.860 [0.715,0.944]; pixel26 4s 0.797 [0.658,0.860]. Full-window comparison: fused 0.849 [0.721,0.929], EG2 0.842, pixel 0.804 (4s not worse; CIs overlap, difference not established). Verdicts over 150 windows: accept 108 / reject 27 / uncertain 15. Every-2nd grid samples 77/150 windows (51.3%). Clip verdicts (sampled bad fraction >= 0.30): 5/7 agree with label-oracle truth (misses desk 0.20 vs 0.45 and pallet2 0.00 vs 0.67, n_sampled=3); E3.01's 7/7 is label-oracle, this is model-based. Cost per E3.04: every-2 x1.94; every-2 + 4s exit x4.86 (~0.907h per 4.404h corpus, 0.2059 of footage). OOD 60 windows (51 good/9 bad) included DESCRIPTIVE ONLY from frozen full-window scores (accept 17 / reject 29 / uncertain 14); no cached features exist to re-score them at 4s and NO OOD/accuracy claim is made.
- **Decision:** Integrate the adopted settings into jadal/poc_stream/policy.py (single source of constants) and eval/poc_integrate/run_integration.py; regenerate window verdicts and summary; keep OOD rows descriptive-only. E3.05 status 'working policy on IndEgo, not validated' is quoted verbatim in results and docs.
- **Cost:** CPU-only, cached features; policy cost ratios from E3.04 (x1.94 every-2; x4.86 every-2+4s exit) reported, not re-measured
- **Results:** ['eval/results/poc_integrated/window_verdicts.jsonl', 'eval/results/poc_integrated/summary.json']

## LB4 — Stage frozen pre-registered OOD run (one command, sha256-guarded)
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** labels/ood_labels2.json (100 human-labelled Egocentric-10K windows, second batch) - NOT yet delivered; videos present in jadal_pod_bundle/data/ego10k_scale/label_windows2
- **Regime:** n/a
- **Method:** tools/pod/run_ood_frozen.sh + eval/exp_ood/run_ood_frozen.py; guards: exactly 100 labelled windows AND frozen artifact raw sha256 == recorded pre-registered hash 4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31 (byte-exact copy recovered from git blob 0ede3b83 at freeze commit b21e3af, committed as configs/frozen/ensemble_v1.prereg.json); metrics: AUROC + clip-bootstrap 95% CI, balanced accuracy at the pre-registered threshold 0.5, accept/reject counts; no tuning
- **Owner / started / finished:** lead-b /  / —
- **Verdict:** inconclusive — STAGED, NOT RUN: labels/ood_labels2.json is absent, so eval/results/ood_frozen/ood_run.json is status PENDING (exit 3) and nothing was scored. Post-freeze pod copies of the artifact fail the check (repo config d62a95a4..., bundle cache cf7fd7d3...) and are recorded as observations only - the recorded hash is never substituted. OOD claims remain n=60 / n_bad=9 (first batch, 51 good / 9 bad) and no claim was updated. One command when labels land: bash tools/pod/run_ood_frozen.sh
- **Decision:** 
- **Cost:** 
- **Results:** eval/results/ood_frozen/ood_run.json

## LB5 — Branch triage & merge queue
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:** lead-b /  / 2026-10-10T16:04:59Z
- **Verdict:** maybe — Input handover/BRANCH_TRIAGE_REPORT.md missing (absent on disk and in every ref), reconstructed from git+PR state: 28 open PRs, 0/28 ancestors of main, 28/28 branch-only (1337 branch-only files), 18 gh-mergeable / 10 conflicting, only CI check is CodeRabbit (main has no .github/workflows, no test CI); 9 conflicting PRs share one mechanical keep-main resolution set. Merge gate on origin/main 65f37dc: 493 passed, 0 failed, 3 skipped, 4 collection errors (fastapi+imagehash missing in the pod env) in 23.5 s.
- **Decision:** 
- **Cost:** 
- **Results:** docs/reports/branch_triage.md
| E8.01 | E8 | done | maybe | E8.01 re-score 1,546 Egocentric-10K windows with the frozen two-view recipe + profile router | queues/manifest_scale.json: 1546 windows / 58 clips / 4.404  / factory Egocentric-10K, 1080p, 30 fps, HEVC, handheld egocentric manufacturing |
| L3.01 | L3 | done | maybe | L3.01 full-corpus pipeline at full resources (7 workers) + adopted optimizations | queues/manifest_scale.json: 1546 windows / 58 clips / 4.404  / factory Egocentric-10K, 1080p, 30 fps, HEVC |
- **Owner / started / finished:** experimenter-e8-rescore / 2026-10-10 15:42 / 2026-10-10 16:27
- **Verdict:** maybe — n=1546/1546 windows (58 clips, 4.404 h) scored, 0 decode failures, label_source none, protocol descriptive re-score + 6-window decode-path control. Verdicts (descriptive, not accuracy): accept 296 (19.15%), reject 83 (5.37%), review 1167 (75.49%); accept share 95% CI [0.136,0.254] by clip bootstrap over 58 clips (window Wilson [0.173,0.212] is anti-conservative). Profile: factory_1080p30 1540, default 6 (UNSEEN_REGIME: 3 extreme ego-motion shift_mean 245-642, 3 overexposed luma_mean 0.66-0.67). Cost: 0.683 windows/s, 5761 ms/window mean (p50 5195, p95 9371), wall 2263 s with 4 workers on a SHARED A40 (load ~15); decode 2354 ms, pixel26 1154 ms, descriptors 1055 ms, EG2 embed 1196 ms. 1546/1546 receipts + features_digest re-verified. Recipe ensemble_v1 (0.8534 full-OOF in-domain AUROC, 150 windows/7 clips, one labeller) did NOT transfer out-of-domain (0.4-0.53 AUROC, n_bad=9), so the accept/reject share is NOT a quality measure.
- **Decision:** Ship /workspace/artifacts/windows_rescored.jsonl as the B2 PoC contract (not committed, artifacts/ ignored); keep the frozen recipe; do not read accept/reject shares as quality; next: E5 domain adaptation on the same corpus, and re-run this runner if the frozen decode path (bit-identical) is needed for the final demo receipts.
- **Cost:** 4 workers x ~2263 s wall (37.7 min) on a shared A40; 1546 windows x (300 pixel frames @160x90 + 20 EG2 frames @512x288 + 8 native descriptor frames) = 1 ffmpeg/NVDEC pass and 1 EG2 forward per window; 0.68 windows/s, 5.76 s/window mean; no extra storage (pipes only); E3.04 every-2nd+4s policy deliberately NOT applied (see report).
- **Results:** eval/results/exp_rescore/E8.01.json
## L3.01 — L3.01 full-corpus pipeline at full resources (7 workers) + adopted optimizations
- **Why:** The frozen L2.01 baseline ran deliberately slow (4 workers, 300 pixel-frames/window, 2 fps EG2, mosaic on). Measure the same corpus with the pod's adopted optimizations and full cgroup resources to quantify the speedup.
- **Hypothesis:** E1.01 3.33 fps pixel + E1.02 1 fps EG2 + E3.03 4 s early exit + E2 skip mosaic cut feature/EG2/mosaic cost several-fold; E3.01 every-2nd halves windows; throughput rises well above the 0.84 win/s baseline while the verdict distribution stays close.
- **Data / label source:** queues/manifest_scale.json: 1546 windows / 58 clips / 4.404 h, 1080p30 HEVC; label_source none (descriptive verdicts)
- **Regime:** factory Egocentric-10K, 1080p, 30 fps, HEVC
- **Method:** same harness as L2.01 (repo E8.01 pipeline, perf_counter per stage); adopted optimizations as CLI/env flags; scoring code unchanged; stride-1 full-corpus run + stride-2 every-2nd run; A40, 7 workers, JADAL_GPU_SLOTS=6
- **Owner / started / finished:** l3-latency / 2026-10-10 18:41 / 2026-10-10 19:20
- **Verdict:** maybe — Full corpus 1546 windows/4.404 h at 7 workers + JADAL_GPU_SLOTS=6 with adopted opts (E1.01 pixel 3.33 fps, E1.02 EG2 1 fps, E3.03 4 s exit, E2 skip mosaic). Wall 801 s, 1.93 win/s, 19.8 footage-h/wall-h = 2.30x baseline throughput (0.84 win/s). Clean 1-worker per-window total p50 1483 ms = 2.99x baseline; per-stage p50 speedups pixel 13.2x, EG2 6.6x, descriptors 2.1x, mosaic skipped. BUT the frozen-head verdict distribution is NOT close: score mean +0.20 (0.486->0.687), accept 296->876, agreement 54% — driven by pixel-fps (+0.15) and early-exit (+0.12), EG2 neutral; needs head refit/recalibration. Decode is the bottleneck (81% of per-window) and 7 workers serialise on the single A40 NVDEC (per-window decode 2780 ms at 7 workers vs 943 ms at 1).
- **Decision:** Adopt the resource/feature optimizations for throughput; do NOT ship sparse sampling with the frozen head until it is refit or the profile thresholds recalibrated. Every-2nd (E3.01) gives 2x footage coverage on top (39.6 footage-h/wall-h derived; 32.9 measured, partly contended).
- **Cost:** 7 workers x 801 s + 482 s (stride-2) + 1-worker probes on a shared A40; 1546 windows x (13 pixel frames @160x90 + 4 EG2 frames @512x288 + 3 native frames)
- **Results:** eval/results/exp_latency_fast/L3.01.json
