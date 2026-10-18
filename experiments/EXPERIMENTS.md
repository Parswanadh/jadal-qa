# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E0.01 | E0 | done | maybe | Label persistence and position bias on IndEgo gold | IndEgo human gold, 150 windows, 7 clips (human labels) / 480p/10fps IndEgo |
| E1.01 | E1 | done | adopt | IndEgo pixel-feature fps sweep {10,5,3.33,2,1} | IndEgo gold 150 windows human labels 100good/50bad / 480p 10fps H264 in-domain |
| E1.02 | E1 | done | maybe | EG2 window-embedding fps subsample {2,1,0.5} + 5/10fps re-embed | IndEgo gold 150 windows human labels / 480p 10fps in-domain |
| E1.02b | E1 | done | drop | Denser EG2 sampling (5/10 fps) vs the 2 fps IndEgo view | IndEgo human gold 150 windows / 7 clips (100 good / 50 bad,  / 480p/10fps H.264 IndEgo in-domain, LOCO over 7 clips |
| E1.03 | E1 | running | - | OOD 1080p30 source fps sweep {30,15,10,5,3,2,1} | OOD labels.json 60 windows human 51good/9bad (+labels2 if pr / 1080p 30fps HEVC out-of-domain |
| E1.04 | E1 | running | - | Frame-selection strategies at fixed 2fps budget | IndEgo gold 150 human labels / 480p 10fps in-domain |
| E1.05 | E1 | done | maybe | Cost / Pareto re-bench: clean minima for the E1 fps and selection arms | cost only, no labels: 1 IndEgo window (labels/manifest_gold. / in-domain IndEgo 480p 10fps H.264 + out-of-domain Egocentric-10K original 1080p 30fps HEVC, decode CPU cv2 160x90 |
| E1.06 | E1 | running | - | fps x resolution Pareto cross (IndEgo gold, pixel-26) | IndEgo gold manifest_gold.json, human labels, n=150 (100 goo / 480p~10fps H.264 source; re-encoded grid fps {1,2,3.33,5} x res {480,360,240}px CRF23 |
| E2.01 | E2 | done | drop | Mosaic embedding (adopt candidate) | IndEgo gold 150 human windows (LOCO, 7 clips) + OOD 60 human / 480p@10fps IndEgo, 854x480@30fps Egocentric-10K; 10-s windows |
| E2.02 | E2 | done | adopt | Mean-of-frames control (matched frames) | same windows/folds as E2.01; label_source human / same as E2.01 |
| E2.03 | E2 | done | drop | Shuffled tile order | same windows/folds; label_source human / same as E2.01 |
| E2.04 | E2 | done | drop | Mosaic without timestamps | same windows/folds; label_source human / same as E2.01 |
| E2.05 | E2 | done | maybe | Single middle frame | same windows/folds; label_source human / same as E2.01 |
| E2.06 | E2 | done | maybe | Motion mosaic (|I_t - I_{t-1}| tiles) | same windows/folds; label_source human / same as E2.01 |
| E2.07 | E2 | done | drop | Pooled per-tile statistics control (no layout) | same windows/folds; label_source human / same as E2.01 |
| E2.1080 | E2 | running | - | True-1080p mosaic tile-resolution ladder + equal-canvas duel | Egocentric-10K human-labelled OOD windows ood160_labels.json / 1080p/30fps HEVC factory egocentric + 480p/30fps transcodes; mixed lighting |
| E3.01 | E3 | done | adopt | Sparse probe vs full coverage on IndEgo clips (label oracle + OOF scores) | IndEgo human gold 150 windows/7 clips (human labels) + EG2/2 / 480p/10fps IndEgo assembly+logistics |
| E3.02 | E3 | done | drop | Position bias on OOD Egocentric-10K labelled windows | OOD labels.json 60 windows (51 good/9 bad, human) + queue so / 1080p/30fps Egocentric-10K factory |
| E3.03 | E3 | done | adopt | Early exit within a window: accuracy vs observed seconds (EG2 prefixes) | IndEgo human gold 150 windows; EG2 21-frame/2fps embeddings  / 480p/10fps IndEgo |
| E3.04 | E3 | done | adopt | Compute accounting: video-hours processed per footage-hour per policy | Egocentric-10K manifest 1546 windows/4.404 h (unlabelled) +  / 1080p/30fps Egocentric-10K factory |
| E3.05 | E3 | done | adopt | Sampling-policy decision rule per data regime | Synthesis of E3.01-E3.04 results JSONs / IndEgo + Egocentric-10K |
| E4.01 | E4 | running | - | Regime descriptors: IndEgo vs Egocentric-10K domain-shift diagnosis | 150 IndEgo gold windows (human) + 60 OOD labelled (human) +  / mixed: 480p/10fps H264 vs 1080p/30fps HEVC |
| E4.02 | E4 | running | - | Regime-specific vs global models within IndEgo | 150 IndEgo gold windows, human labels, leave-one-clip-out / terciles of sharpness/lighting/motion |
| E4.03 | E4 | running | - | Domain gap control: common-spec renormalisation | 60 OOD labelled windows, human labels; frozen ensemble basel / 480p/10fps transcode vs 1080p/30fps source |
| E4.04 | E4 | running | - | Profiles product layer: schema + examples + deterministic router | synthetic + E4.01 descriptor ranges / all |
| E5.00 | E5 | done | maybe | Frozen ensemble OOD prior (usable_prob AUROC on 60 human OOD windows) | 60 Egocentric-10K human-labelled windows (51 good / 9 bad, s / 480p 10s transcodes of 1080p/30fps factory clips; ~1 window/factory |
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
| GRID.00 | GRID | done | adopt | Baseline gate: 26-feature logistic LOCO on IndEgo-150 must reproduce AUROC 0.810 | IndEgo 150 windows (100 good / 50 bad), ONE human labeller;  / 480p, ~10 fps, IndEgo assembly+logistics, 7 clips |
| GRID.V01 | GRID.V | done | adopt | Reading check: does the model read the sheet's tiles and timestamps correctly? | 20 context sheets (OOD subset seed 0), one read_tiles call e / 12-tile 4x3 426px sheets, 10 s windows, CPU sheet build + API call |
| GRID.V02 | GRID.V | running | - | Quality verdict from a 12-tile sheet on OOD-160, factory-grouped | OOD-160: 160 human-labelled Egocentric-10K windows, 138 good / One 10 s window = one 12-tile sheet; frozen quality prompt; factory-grouped bootstrap CI; paired comparison vs the frozen shipped head |
| GRID.V03 | GRID.V | done | inconclusive | Ablations: timestamps, tile order, single frame, tile count, reasoning step, grey null | The V03 stratified 20 windows (10 bad / 10 good, frozen ids  / Six arms x 20 windows: no timestamps, shuffled order, one middle frame, 6 tiles, 24 tiles, JSON-only prompt; plus a flat grey null sheet |
| GRID.V04 | GRID.V | running | - | Run-to-run variance of the model on the same sheets | The V04 40-window subset (frozen ids in PREREG), 3 runs each / Same frozen prompt and sheets across runs; unanimous share and Cohen/Fleiss kappa over usable and over task_progress |
| GRID.V05 | GRID.V | done | drop | Task level on IndEgo: 12-tile sheets vs video on the same 30 s slices, then L2 | 4 annotated IndEgo clips (ecu_full, ecu_late, desk, organise / One slice_l1 call per 30 s slice as a sheet and as video via the existing analyze_video path; then ONE l2_text call over the six slice answers per clip |
| GRID.V06 | GRID.V | running | - | Role in the funnel: review-band resolver and veto on high-confidence accepts | IndEgo-150 and OOD-160 with frozen model outputs from V02 pl / Frozen prompt and model id; (a) resolver for the band, (b) veto on windows the shipped head accepts at p>=0.75; error at 100 % coverage and the resulting review load |
| LB1 | B | done | adopt | Independent verification of pod baselines, E1.01-E1.02 fps sweeps and E3.01-E3.05 temporal policy | IndEgo human gold 150 windows / 7 clips (human labels, READ- / 480p/10fps IndEgo in-domain LOCO (+ OOD position test, labels only) |
| LB2 | B | done | adopt | PoC integration: adopted every-2nd window grid + 4s early exit + EG2 2fps + 26-feature pixel logistic (descriptive verdicts) |  / CPU only (GPU unavailable in the container); cached features; deterministic (fixed seeds) |
| LB4 | B | pending | inconclusive | Stage frozen pre-registered OOD run (one command, sha256-guarded) | labels/ood_labels2.json (100 human-labelled Egocentric-10K w /  |
| LB5 | B | done | maybe | Branch triage & merge queue |  /  |
| MR.01 |  | done | adopt |  |  /  |
| MR.02 |  | done | drop |  |  /  |
| MR.03 |  | done | maybe |  |  /  |
| O1.01 | O | done | maybe | Frozen PoC out-of-domain evaluation on the 160 human labels | Egocentric-10K 160 windows, 47 factories, 138 good / 22 bad  /  |
| O2.01 |  | done | drop |  |  /  |
| O2.02 |  | done | maybe |  |  /  |
| O2.03 |  | done | maybe |  |  /  |
| O3.01 | O3 | done | adopt-with-caveats | Honest review-vs-OOD-error frontier on the 160-window pre-registered OOD label set | 160 human-labelled Egocentric-10K windows (138 good / 22 bad / Egocentric-10K factory 1920x1080/30 |
| O4.01 |  | done | adopt |  |  /  |
| O4.02 |  | done | adopt |  |  /  |
| RES.01 |  | done | adopt |  |  /  |
| RES.02 |  | done | adopt |  |  /  |
| RES.03 |  | done | adopt |  |  /  |
| RES.04 |  | done | maybe |  |  /  |
| SKIP.01 |  | done | maybe |  |  /  |
| SKIP.02 |  | done | drop |  |  /  |
| SKIP.03 |  | done | adopt |  |  /  |
| Y1.01 |  | done | adopt |  |  /  |
| Y2.01 |  | done | adopt |  |  /  |
| Y2.02 |  | done | adopt |  |  /  |
| Y3.01 |  | done | adopt |  |  /  |
| Y3.02 |  | done | drop |  |  /  |
| Z1.01 |  | done | maybe |  |  /  |
| Z1.02 |  | done | drop |  |  /  |
| Z1.03 |  | done | maybe |  |  /  |
| Z2.01 |  | done | adopt |  |  /  |
| Z2.02 |  | done | adopt |  |  /  |
| Z2.03 |  | done | drop |  |  /  |
| Z3.01 |  | done | maybe |  |  /  |
| Z3.02 |  | done | no |  |  /  |
| Z3.03 |  | done | yes |  |  /  |
| Z4.01 |  | done | maybe |  |  /  |
| Z4.02 |  | done | drop |  |  /  |
| Z4.03 |  | done | maybe |  |  /  |

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

## E1.02b — Denser EG2 sampling (5/10 fps) vs the 2 fps IndEgo view
- **Why:** E1.02 left the denser arm untested ('5/10fps re-embed only if GPU time remains'); the product ships the 2 fps EG2 view, so the cost/benefit of 2.5x-5x denser sampling had to be settled before the freeze.
- **Hypothesis:** A denser window mean (50/100 frames instead of 20) raises the LOCO AUROC of the EG2 view by more than the 0.005 practical-significance bar.
- **Data / label source:** IndEgo human gold 150 windows / 7 clips (100 good / 50 bad, one labeller, read-only labels/manifest_gold.json, labels_sha256 43ab44e5...); published 2 fps cache caches/frames10k/segments.npz
- **Regime:** 480p/10fps H.264 IndEgo in-domain, LOCO over 7 clips
- **Method:** one GPU decode+embed pass at native 10 fps (100 frames/window, 15000 frames, A40, EG2 revision 914f7f89..., bf16, max_soft_tokens 140); 5 fps and 2 fps are exact row subsamples ::2/::5 of that pass; LOCO logistic C=0.01 + StandardScaler on the train fold; clip-bootstrap B=2000 seed 0 with ONE shared resample set, so CIs and deltas are paired; deltas reported vs the published 2 fps cache AND vs a same-pass 2 fps control (noise floor).
- **Owner / started / finished:** lead-b / 2026-10-10 17:16 / 2026-10-10 17:38
- **Verdict:** drop — n=150 IndEgo human windows (100 good/50 bad, 7 clips, one labeller), device A40. EG2 LOCO AUROC: 2 fps published 0.8480 [0.733,0.910]; 2 fps same-pass control 0.8468 [0.733,0.909]; 5 fps 0.8486 [0.743,0.910]; 10 fps 0.8478 [0.739,0.910]. Paired clip-bootstrap deltas (B=2000, shared resamples) against the published 2 fps cache: 5 fps +0.0006 [-0.0051,+0.0140], 10 fps -0.0002 [-0.0082,+0.0127]; density-only vs the same-pass 2 fps control: 5 fps +0.0018 [-0.0045,+0.0151], 10 fps +0.0010 [-0.0073,+0.0127]; noise floor (same frames re-embedded) -0.0012 [-0.0048,+0.0049]. NO difference is established at any density and the point estimates sit at the noise floor, so denser sampling is dropped on cost-effectiveness: 2.5x (5 fps) and 5.0x (10 fps) the embedding work per window (0.85 s -> 2.12 s -> 4.23 s measured on the A40). Decode validated pixel-identical to the published seek decode (6/6 windows), per-frame cosine 0.9947.
- **Decision:** Keep the 2 fps EG2 view in the funnel; do not re-run the denser arm. Read the verdict as 'no gain established at 2.5x-5x cost', NOT as 'denser is worse' (the CIs exclude only gains above ~+0.015).
- **Cost:** GPU A40 slot 1, 15000 EG2 frames: 78.5 s decode + 556.6 s encode (26.9 frames/s, batch 16), VRAM peak 2.17 GB; scoring CPU ~40 s; embeddings kept out of git at /workspace/cache_e1_02b/dense10.npz (17.9 MB).
- **Results:** eval/results/exp_fps/e1_02b.json

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

## E1.05 — Cost / Pareto re-bench: clean minima for the E1 fps and selection arms
- **Why:** The PR #88 review (F5/F6) found the committed Pareto artefact had no single-command regeneration (run_all.sh omitted --benches) and its cost_note quoted a shard-run multiplier that does not match the committed JSONs; the pod load also changed since the first bench, so re-measure the clean minima and let the script record the per-regime/stage ratio in the JSON instead of a hand-typed range.
- **Hypothesis:** Re-benching keeps the same Pareto front (IndEgo 10/3.33/2/1 fps + uniform20, OOD 1-2 fps) and the shard-run means stay slower than the clean minima, by a regime/stage-dependent factor of roughly 2-9x.
- **Data / label source:** cost only, no labels: 1 IndEgo window (labels/manifest_gold.json, human) + 1 Egocentric-10K labelled window (labels/ood_labels.json source_path, human); accuracy rows read from E1.01/E1.03/E1.04.json (n=150 in-domain LOCO, n=60 out-of-domain, 51 good/9 bad)
- **Regime:** in-domain IndEgo 480p 10fps H.264 + out-of-domain Egocentric-10K original 1080p 30fps HEVC, decode CPU cv2 160x90
- **Method:** eval/exp_fps/pareto.py --reps 3 --benches 3: min over 3 reps x 3 bench runs of decode + cpu_window_features per fps setting, OMP_NUM_THREADS=1, cv2 threads=1; Pareto front over (AUROC, total ms/window) per regime; clean_vs_shard_ratio computed in-script from E1.01/E1.03/E1.04.json
- **Owner / started / finished:** experimenter / 2026-10-10 16:20 / 2026-10-10 16:20
- **Verdict:** maybe — cost re-bench, no labels (n=1 IndEgo window + 1 Egocentric-10K 1080p window), 3 reps x 3 bench runs, min-of-9, OMP_NUM_THREADS=1 cv2 threads=1, host load 17.6: clean minima in-domain 480p decode 112.0 ms/window (1.1198 ms/frame) + features 288.9/166.7/102.2/53.7/32.4 ms/window at 10/5/3.33/2/1 fps; OOD 1080p HEVC decode 2900.4 ms/window (9.6679 ms/frame) + features 843.0/391.8/246.5/128.7/74.9/55.5/32.4 ms/window at 30/15/10/5/3/2/1 fps. Pareto fronts unchanged (in-domain 10/3.33/2/1 fps + uniform20; OOD 1-2 fps). The shard-run means in E1.01/E1.03/E1.04.json are 2.24-8.01x these clean minima, regime/stage-dependent (OOD decode 2.24x, in-domain decode 5.08x, features 3.11-5.75x). Spread over the 3 bench runs: in-domain decode 1.12-1.84 ms/frame, OOD decode 9.67-11.54 ms/frame.
- **Decision:** Cite pareto.json for every clean cost number in the E1 docs. Regenerate with python3 eval/exp_fps/pareto.py --reps 3 --benches 3 (run_all.sh now does); read clean_vs_shard_ratio instead of quoting a hand-typed shard/clean multiplier.
- **Cost:** about 80 s wall, 1 core, no GPU; CPU decode cv2 160x90
- **Results:** eval/results/exp_fps/pareto.json

## E1.06 — fps x resolution Pareto cross (IndEgo gold, pixel-26)
- **Why:** Find cheapest (fps,res) cell within 0.02 AUROC of 480p/10fps baseline; prior E1.01 (fps) and X1/RES (res/CRF) tested one axis each, interaction (low fps + low res) unknown
- **Hypothesis:** A mid cell (e.g. 360p/3.33fps or 240p/5fps) keeps AUROC within 0.02 of 0.809 at a fraction of bytes; very-low+very-low (240p/1fps) may interact downward
- **Data / label source:** IndEgo gold manifest_gold.json, human labels, n=150 (100 good/50 bad), 7 clips
- **Regime:** 480p~10fps H.264 source; re-encoded grid fps {1,2,3.33,5} x res {480,360,240}px CRF23
- **Method:** re-encode each gold window at (fps,res) libx264 veryfast CRF23 -> decode 160x90 -> 26 cpu per-pixel features -> LOCO logistic C=0.3 balanced + StandardScaler; CI=clip-bootstrap B=1000 seed0
- **Owner / started / finished:** H3 / 2026-10-10 17:43 / —
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

## E2.1080 — True-1080p mosaic tile-resolution ladder + equal-canvas duel
- **Why:** User claim: 480p-or-smaller tiles with more of them process faster without accuracy loss; X3 tested this on 480p gold clips only, need TRUE 1080p source
- **Hypothesis:** At equal canvas pixels, 1080p-source 3x3/t224 ties 480p-source 4x4/t168 within 0.02 AUROC; smaller tiles cut canvas px and ms
- **Data / label source:** Egocentric-10K human-labelled OOD windows ood160_labels.json (160; 138 good/22 bad), 1080p HEVC source + 480p H264 transcodes
- **Regime:** 1080p/30fps HEVC factory egocentric + 480p/30fps transcodes; mixed lighting
- **Method:** mosaic EG2 embeddings, stratified 5-fold CV logistic C=0.01, factory-bootstrap CI
- **Owner / started / finished:** H2 / 2026-10-10 17:43 / —
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

## GRID.00 — Baseline gate: 26-feature logistic LOCO on IndEgo-150 must reproduce AUROC 0.810
- **Why:** Pre-registered gate from the DSH brief: no GRID experiment starts until the documented baseline is reproduced on this machine from the pod-bundle cache, proving data paths, features and folds are the same as the pod.
- **Hypothesis:** Pooled leave-one-clip-out AUROC on the 150 human-labelled IndEgo windows = 0.810 +/- 0.01; semantic-bad and quality-bad breakdown as documented (pallet/pallet2 at chance).
- **Data / label source:** IndEgo 150 windows (100 good / 50 bad), ONE human labeller; features from pod bundle cache caches/gpu/features.json (26 per-pixel features); labels labels/manifest_gold.json
- **Regime:** 480p, ~10 fps, IndEgo assembly+logistics, 7 clips
- **Method:** tools/port/baseline_logistic_loco.py unchanged: StandardScaler + LogisticRegression(C=0.3, class_weight=balanced), leave-one-clip-out over 7 clips, pooled AUROC; clip-bootstrap 95% CI
- **Owner / started / finished:** dsh / 2026-10-18 06:30 / 2026-10-18 06:32
- **Verdict:** adopt — PASS. 26-number logistic LOCO AUROC 0.810 [0.683,0.892] clip-bootstrap (window-bootstrap 0.722-0.889, optimistic), n=150 (100 good/50 bad, ONE human labeller), 7 clips. good-vs-semantic-bad 0.797, good-vs-quality-bad 0.846. pallet/pallet2 at chance (0.50) as documented. Gate expected 0.810 +/-0.01: observed 0.810.
- **Decision:** Gate passed: GRID.01-09 and AUTO.00+ may start. Cached pod-bundle 26 features reproduce exactly on this laptop; use jadal-qa-wt/handover/jadal_pod_bundle/pod_bundle as JADAL_BUNDLE.
- **Cost:** 0 GPU, ~2 s CPU
- **Results:** eval/results/grid/GRID.00.json

## GRID.V01 — Reading check: does the model read the sheet's tiles and timestamps correctly?
- **Why:** The whole track depends on the model actually reading the labels; if it does not, every later verdict is uninterpretable.
- **Hypothesis:** Per-tile text agrees with the tile on a human spot check of 20 windows; a timestamp-reading question returns the exact label.
- **Data / label source:** 20 context sheets (OOD subset seed 0), one read_tiles call each + timestamp spot check
- **Regime:** 12-tile 4x3 426px sheets, 10 s windows, CPU sheet build + API call
- **Method:** One read_tiles call per window; the builder spot-checks the tiles; exact-match rate on the 'tile 7' timestamp question.
- **Owner / started / finished:** dsh / 2026-10-18 08:29 / 2026-10-18 09:04
- **Verdict:** adopt — Reading check PASSES at ceiling: 20/20 pre-registered windows, 240/240 sheet tiles returned a line (coverage 1.000) and 240/240 matched our own sidecar on BOTH tile number and timestamp (exact match 1.000 per sheet tile and per returned line), 0 failed calls, 0 unparsed, 0 out-of-range numbers, 0 right-number/wrong-timestamp lines; mean ~12 s/call; 20 image calls (PREREG cap 20). Independently re-derived by eval/exp_vlm/verify_v01.py (own regex, own sidecar re-read, no shared code): identical 240/240. Re-parsed with both the strict and the tolerant (post-fix) parse_tile_lines: identical, because the count is at ceiling. Negative control: a contentless flat-grey sheet through read_tiles produced 12 'uniform gray blank tile' lines and wrote every timestamp as --:--.- rather than inventing one.
- **Decision:** ADOPT the sheet FORMAT as readable (scope: readability only). This is a reading measurement and supports NO quality claim: a model that reads perfectly can still judge badly (that is V02/V03). Builder's human spot check of the 20 sheets is OUTSTANDING; every PNG sha256 and raw reply is in the results JSON for it.
- **Cost:** 20 image API calls (V01 PREREG cap 20); ~12 s/call; 0 text calls; sheets built on CPU
- **Results:** eval/results/exp_vlm/GRID.V01.json

## GRID.V02 — Quality verdict from a 12-tile sheet on OOD-160, factory-grouped
- **Why:** The mosaic has never been shown to a vision-language model. Awign's description ('12 frames in one image') is this configuration. The key question is whether it catches NO_ACTIVE_WORK, which the pixel head misses.
- **Hypothesis:** The model separates good from bad factory windows better than the frozen shipped head (AUROC 0.588) with a paired factory-clustered interval excluding zero, and catches a useful share of the 19 NO_ACTIVE_WORK windows.
- **Data / label source:** OOD-160: 160 human-labelled Egocentric-10K windows, 138 good / 22 bad (19 NO_ACTIVE_WORK), 47 factories, ONE labeller
- **Regime:** One 10 s window = one 12-tile sheet; frozen quality prompt; factory-grouped bootstrap CI; paired comparison vs the frozen shipped head
- **Method:** AUROC, recall of bad, false-reject on good, balanced accuracy, confusion by reason code, usable=true share, paired factory-clustered difference vs the shipped head.
- **Owner / started / finished:** dsh / 2026-10-18 08:29 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 

## GRID.V03 — Ablations: timestamps, tile order, single frame, tile count, reasoning step, grey null
- **Why:** V02 could be explained by any single frame, by the labels, or by the reasoning step rather than by the multi-frame sheet.
- **Hypothesis:** Removing timestamps, shuffling tile order, or dropping to one frame changes the verdict materially; the grey null sheet is not read as usable.
- **Data / label source:** The V03 stratified 20 windows (10 bad / 10 good, frozen ids in PREREG)
- **Regime:** Six arms x 20 windows: no timestamps, shuffled order, one middle frame, 6 tiles, 24 tiles, JSON-only prompt; plus a flat grey null sheet
- **Method:** Per-arm AUROC and paired differences against the V02 primary arm on the same windows; the null must read cannot_tell or usable=false.
- **Owner / started / finished:** dsh / 2026-10-18 08:29 / 2026-10-18 09:04
- **Verdict:** inconclusive — Six arms x 20 stratified windows (10 bad / 10 good, all 10 bad NO_ACTIVE_WORK, 14 factories) + 2 contentless-sheet controls = 122 calls, exactly the cap. NEGATIVE CONTROL PASSES: grey sheet through quality -> usable=false, HANDS_NOT_VISIBLE, cannot_tell ('All twelve tiles are blank gray...'); grey sheet through read_tiles -> 12 lines, all timestamps '--:--.-' (no hallucinated times). So the model reads the image. Arm AUROC (usable; binary so equal to balanced accuracy at 0.5): c_single_middle_frame 0.600 [0.379,0.800]; d24 0.650 [0.470,0.846]; a_no_timestamps 0.700 [0.500,0.896]; e_json_only (REFERENCE) 0.700 [0.500,0.896]; d6 0.750 [0.542,0.944]; b_shuffled 0.800 [0.586,1.000]. Every paired factory-clustered difference vs the reference INCLUDES ZERO (max +0.100 [0.000,+0.278]); the critical 12-frame vs single-frame comparison is -0.100 [-0.250,+0.111] and is additionally confounded (arm c uses the quality prompt, the reference uses quality_json_only). NOTHING IS ESTABLISHED. Read against V-A's V04 noise floor (per-run AUROC spread 0.0714, Fleiss kappa 0.52), no difference <=0.10 on 20 windows is interpretable. Notable: arms a and e returned identical verdicts on 20/20 windows (identical usable AND task_progress) on genuinely different sheets (0/20 shared PNG sha256), so within this subset neither the timestamps nor the reasoning step changes the verdict.
- **Decision:** INCONCLUSIVE, not a drop: the design cannot resolve arm differences at this budget. The sheet variants are not shown to matter, and the single-frame question is NOT answered. To read this table, repeat each arm several times per cell (V04-style) before comparing; do not quote any arm ordering from this run.
- **Cost:** 122 image API calls (cap 122 = 120 arms + 2 controls); sub-stream ledger total 142 incl. V01's 20
- **Results:** eval/results/exp_vlm/GRID.V03.json

## GRID.V04 — Run-to-run variance of the model on the same sheets
- **Why:** The model is probabilistic; without a variance number, no VLM result can be compared with a deterministic head.
- **Hypothesis:** Unanimous share and kappa over the 3 runs are high enough to treat a single run as a measurement, or they are not and the track reports a ranker only.
- **Data / label source:** The V04 40-window subset (frozen ids in PREREG), 3 runs each
- **Regime:** Same frozen prompt and sheets across runs; unanimous share and Cohen/Fleiss kappa over usable and over task_progress
- **Method:** Unanimous share, kappa with CI, and the per-run AUROC spread.
- **Owner / started / finished:** dsh / 2026-10-18 08:29 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 

## GRID.V05 — Task level on IndEgo: 12-tile sheets vs video on the same 30 s slices, then L2
- **Why:** L1 (is relevant work progressing) is answerable from video alone; L2 (was the task completed) needs the builder's definition. IndEgo's 4 annotated clips are free ground truth.
- **Hypothesis:** Sheet L1 agrees with the step annotations at least as well as the video path, and the text-only L2 summariser beats the majority class.
- **Data / label source:** 4 annotated IndEgo clips (ecu_full, ecu_late, desk, organise), 30 s slices, clip-absolute tile timestamps; L1 labels derived from the VIA step annotations; L2 from the builder's working definition (pick from one place, put in another, accurately)
- **Regime:** One slice_l1 call per 30 s slice as a sheet and as video via the existing analyze_video path; then ONE l2_text call over the six slice answers per clip
- **Method:** L1 agreement vs annotations and vs the always-advancing baseline, kappa, seconds per call, bytes sent, image-sheet vs video; L2 accuracy vs the majority class. Reported as 'builder's definition, not Awign's'.
- **Owner / started / finished:** dsh / 2026-10-18 08:29 / 2026-10-18 08:49
- **Verdict:** drop — IndEgo 4 annotated clips, 6 evenly spaced 30 s slices each (24 slices, 2 of them no_relevant_work), 12-tile sheets with clip-absolute timestamps vs the same slices as VIDEO. L1 (is relevant work progressing): sheet agreement 0.7083 vs an always-advancing baseline of 0.9167, kappa -0.1351, gap recall 0/2 -- WORSE than a constant; video agreement 0.7917, also below the constant, and it also missed both gap slices. Reading is not the problem: V01 showed 240/240 exact tile+timestamp reads. L2 (was the task completed, BUILDER's definition): the free truth is 4/4 completed = ZERO variance, so no accuracy can be computed; the model answered not_completed once and cannot_tell three times, disagreeing with the annotations on all four clips. Cost: 24 image + 4 text + 24 video calls; sheet 10.04 s/call, video 9.63 s/call, video median 1.06 MB.
- **Decision:** DROP the VLM as an L1/L2 task judge on this evidence: on IndEgo it does not beat the always-advancing constant, and the L2 truth is too weak (zero variance) to support any claim. The 12-tile sheet and 30 s of video cost the same (~10 s) and perform the same, so the sheet is not a cheaper substitute for video on the task question either. Builder's definition of completion used for L2, NOT Awign's; Awign must confirm what counts as inaccurate.
- **Cost:** 24 image + 4 text + 24 video API calls
- **Results:** eval/results/exp_vlm/GRID.V05.json

## GRID.V06 — Role in the funnel: review-band resolver and veto on high-confidence accepts
- **Why:** A model that is only a ranker is still useful if it resolves the review band or vetoes confident accepts.
- **Hypothesis:** As a resolver it lowers error at 100 % coverage versus AUTO.00's forced 0.5; as a veto it catches false accepts without destroying the accept share.
- **Data / label source:** IndEgo-150 and OOD-160 with frozen model outputs from V02 plus fresh calls on the IndEgo review band
- **Regime:** Frozen prompt and model id; (a) resolver for the band, (b) veto on windows the shipped head accepts at p>=0.75; error at 100 % coverage and the resulting review load
- **Method:** 100 %-coverage balanced accuracy and review share for each role, with clip- and factory-grouped intervals, against AUTO.00's forced decision.
- **Owner / started / finished:** dsh / 2026-10-18 08:29 / —
- **Verdict:** pending — 
- **Decision:** 
- **Cost:** 
- **Results:** 

## LB1 — Independent verification of pod baselines, E1.01-E1.02 fps sweeps and E3.01-E3.05 temporal policy
- **Why:** The adopted settings (every-2nd window, 4 s early exit, EG2 2 fps view, 26-feature logistic) drive the PoC and the submission; a second, independent re-run is the cheapest defence against a silent protocol bug.
- **Hypothesis:** Every published number reproduces within the published precision when re-run from pod/base with the cached features and read-only gold labels.
- **Data / label source:** IndEgo human gold 150 windows / 7 clips (human labels, READ-ONLY); caches/gpu/features.json; caches/frames10k/segments.npz; caches/ensemble/oof_probs.npz; OOD labels n=60 n_bad=9
- **Regime:** 480p/10fps IndEgo in-domain LOCO (+ OOD position test, labels only)
- **Method:** Re-run pinned scripts from pod/base; E1.02 additionally re-implemented from scratch; LOCO + clip-bootstrap 95% CI; exact-match tolerance at published precision
- **Owner / started / finished:** lead-b /  / —
- **Verdict:** adopt — 20/20 core claims reproduced EXACTLY from pod/base: pixel26 0.810, EG2 0.8480 [0.734,0.909], prob-avg 0.853, rank-avg 0.856; E1.01 10/5/3.33/2/1 fps 0.8086/0.7932/0.8068/0.7866/0.7736; E1.02 2/1/0.5 fps 0.8480/0.8596/0.8702 (plus an independent from-scratch re-implementation with identical point estimates); two-view 0.8524/0.8538; E3.01 every-2 clip-error 0.0486 [0.0189,0.0872], 7/7 verdicts, delay 0.286 at 0.515 of windows; E3.02 MW p=0.419/0.468 (n=60, n_bad=9); E3.03 2/4/6/8/10.5 s = 0.8038/0.8596/0.8694/0.8574/0.8480; E3.04 x1.94; E3.05 strict-pass ['every-2']. TWO FINDINGS FAILED VERIFICATION: (1) 14/316 SHA256SUMS.txt entries mismatch on disk, including the frozen scorer caches/ensemble/ensemble_v1.json (recorded 4da1068a..., on disk cf7fd7d3...) and labels/manifest_gold.json - cause is post-checksum path remapping (the file contains the pod path /workspace/jadal_pod_bundle/data/eg2); no canonical-JSON variant matches either; LB4 recovered the byte-exact frozen file from git blob 0ede3b83d93ce23dc5b68cd20fb7de0af703ad27. (2) OOD ATTRIBUTION FIXED: the published 0.529 is the PoC CASCADE and reproduces exactly (origin/main:eval/results/poc/ood_check_60.json -> 0.529 [0.316,0.728], 21/51 good rejected, 5/9 bad, bal.acc 0.513 vs published 0.51); the FROZEN ensemble_v1 is a different figure, 0.415 [0.207,0.630] with 26/51 good rejected and 3/9 bad (p_eg2 0.586, p_pixel 0.342). My first pass conflated the two scorers. No drift involved (shipped OOD file is byte-identical to origin/main, sha256 300c1eda...). Both scorers are at chance with heavy false rejects, so the finding the submission rests on is confirmed twice. 29 attempts recorded by verifier PR #100. Also: the frozen artifact hash IS recoverable and verifies (git blob 0ede3b83d93ce23dc5b68cd20fb7de0af703ad27 -> sha256 4da1068a..., independently re-checked by LB1). ADDITIONAL LIVE HAZARD: jadal.ensemble.scorer.load_config() raises ValueError on the DEFAULT config path (config sha256 mismatch d62a95a4... != recorded 4da1068a...) on pod/base and every branch tested, because the shipped config is the path-remapped file while the .sha256 still records the frozen hash; meanwhile the LB2 PoC loader (jadal/poc_stream/policy.py load_frozen_ensemble) does NOT check the sha256 at all and loads the remapped artifact silently. Strict loader fails on a recoverable situation, lenient loader succeeds on the wrong artifact. Fix: point the default path at the recovered hash-verified configs/frozen/ensemble_v1.prereg.json (LB4, PR #97) and make the PoC loader fail closed. LB2 C-sensitivity independently verified (PR #98): fused 4s AUROC 0.8736 at C_eg2=0.003 vs 0.8734 at published C_eg2=0.01, paired delta -0.0002 CI [-0.0133,+0.0095] => C choice is immaterial; do not frame it as an improvement over 0.848 (that is a full-window inner-OOF number). POST-GPU-UNLOCK ADDITIONS (the GPU was never broken - see docs/reports/pod_verification.md §9: the driver was loaded and CUDA 12.8 installed, but the execution sandbox denied /dev/nvidia* with EACCES, which surfaced as NVML 'Unknown Error'; once device access was granted the A40 came up immediately). (a) FEATURE-EXTRACTION HALF NOW VERIFIED: all 150 gold windows re-derived from source video reproduce caches/gpu/features.json to max|delta| 7.11e-15 (mean per-window max 1.81e-15). (b) NESTED AUROCs RECOMPUTED FROM RAW PROBABILITIES (oof_probs.npz holds the NESTED OOF probs): EG2 0.8458, pixel 0.7916, prob-avg 0.8336, rank-avg 0.8440 - exactly the published nested figures, so the earlier caveat that they could not be recomputed from scratch is RETRACTED. (c) E1.02b (denser EG2 at 5/10 fps) and the EG2 embedding-cache reproduction are running as separate tracks. (d) RECEIPT + DETERMINISM CLAIMS VERIFIED: receipt hash is stable across calls and processes, the timestamp and receipt_id are excluded/derived correctly, tampering any of the 5 body fields is detected, key order is irrelevant (sorted-key canonical JSON), there is no signature field inside the receipt, and the CPU feature path gives an identical quantised digest over 10 repeated runs (max spread 0.0, cross-process match). (e) GPU KERNEL PARITY: the GPU feature kernels agree with the verified CPU reference on 22/26 features to <1e-6, but 7 features move >1e-3 on this device (rppg_pulse up to 0.57 abs, radial_slope 0.24, dup_hash_mean 0.14, illum_* ~0.008, pca_top1_share 0.003). Decision impact measured over all 150 windows: LOCO AUROC 0.7998 from GPU features vs 0.8100 from CPU features, paired clip-bootstrap delta -0.0093 [-0.0304,+0.0149] => the difference is NOT established at n=150. So the fast GPU path is usable without a demonstrated accuracy cost, but it is NOT the same artifact as the published 0.810 and any number must say which path produced it.
- **Decision:** Adopt the verified in-domain stack (every-2nd window, 4 s exit, EG2 2 fps, 26-feature logistic). Quote the two OOD numbers separately and never merge them: 0.529 = PoC cascade, 0.415 = frozen ensemble_v1; both at chance. GPU feature-extraction half and the delivered OOD-labels half remain unverified (no CUDA in this container). Feature-extraction half verified; nested numbers verified from raw probs. Remaining unverified: the E1.02b dense-sampling question and the EG2 embedding-extraction half (both in flight).
- **Cost:** CPU-only, ~6 min wall (2 shards E1.01 + cached-feature LOCO); 0 GPU
- **Results:** docs/reports/pod_verification.md + eval/results/pod_verify/ (baseline_pixel26_loco.json, baseline_ensemble_views.json, e1_01_fps_sweep.json, e1_02_eg2.json, ood_and_nested.json, bundle_checksum_audit.json, exp_temporal/E3.0{1..5}_*.json) + eval/results/pod_verify/ood_and_nested.json + bundle_checksum_audit.json + feature_reproduction_all150.json + oof_crosscheck.json + feature_ablation.json + receipts_determinism.json + gpu_kernel_parity.json + gpu_features_decision_impact.json + manifest_restore_event.json

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

## MR.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:50
- **Verdict:** adopt — Tiles 112-160px at 3x3 match/better 224px (0.803-0.804 vs 0.785) at -36% ms/win and -74% canvas px; smallest within 0.02 of best = 112px. Still below mean-of-frames N9 0.851. Adopt small tiles for a mosaic pipeline.
- **Decision:** adopt 112-160px tiles at 3x3
- **Cost:** 43.7 ms/win, 118k canvas px @112px vs 68.5 ms/win, 462k px @224px
- **Results:** eval/results/exp_mosaic_res/MR.01.json

## MR.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:50
- **Verdict:** drop — At matched frame count mosaic loses to mean-of-frames at EVERY density (-0.048 N9, -0.063 N16, -0.113 N25, -0.081 N36). Best density 3x3 (0.803). E2 verdict stands; density does not recover the gap.
- **Decision:** keep as evidence artefact only
- **Cost:** 43.7 ms/win @3x3/112 to 107.4 ms/win @6x6/112
- **Results:** eval/results/exp_mosaic_res/MR.02.json

## MR.03 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:50
- **Verdict:** maybe — Equal canvas (~462-465k px): 16x168px (0.845) beats 9x224px (0.785) by +0.060 AUROC at -2.4 ms/win -> supports 'more smaller tiles, no loss'. But mosaic still within 0.016 of mean-of-frames N16 (0.861) and only ties EG2 baseline 0.848. Timestamps help +0.021; order irrelevant.
- **Decision:** adopt geometry only if a mosaic input is used; mean-of-frames still preferred
- **Cost:** 66.1 ms/win, 465k canvas px
- **Results:** eval/results/exp_mosaic_res/MR.03.json

## O1.01 — Frozen PoC out-of-domain evaluation on the 160 human labels
- **Why:** Measure the real OOD error rate of the frozen ensemble_v1 against the newly delivered 160 human labels (pre-registered test)
- **Hypothesis:** The IndEgo-trained frozen ensemble transfers weakly to Egocentric-10K
- **Data / label source:** Egocentric-10K 160 windows, 47 factories, 138 good / 22 bad (labels/ood160_labels.json); frozen scores from caches/ensemble/ood_scores{,2}.json
- **Regime:** n/a
- **Method:** frozen two-view ensemble_v1 as-is (no fitting/re-thresholding); AUROC + balanced accuracy with window bootstrap (optimistic) and factory-grouped bootstrap (honest)
- **Owner / started / finished:** O1 /  / 2026-10-10 17:34
- **Verdict:** maybe — Frozen ensemble OOD AUROC 0.566 (window CI 0.433-0.693; factory-grouped 0.414-0.697), up from a noise-level 0.415 at n=60. Shipped operating point: accept 71 / review 35 / reject 54; BA 0.387 review=miss (0.546 scalar@0.5); it hard-rejects 47/138 good windows (34%) and rejects only 7/22 bad. Weak positive ranking signal, not separable from chance; unusable as a gate.
- **Decision:** OOD claims: replace 'at chance/inverted 0.415' with 'weak positive 0.566, not significant, bad operating point'; cascade 0.600 carried over from handover with no grouped CI.
- **Cost:** CPU only, caches (no GPU decode)
- **Results:** eval/results/exp_ood160/O1.01.json

## O2.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:40
- **Verdict:** drop — 160 OOD windows (138/22, 47 factories), frozen ensemble_v1 usable_prob: AUROC 0.566, factory-grouped CI [0.445,0.674] (includes 0.5), window-bootstrap CI [0.437,0.694], bal.acc@0.5 0.546 [0.460,0.635], acc 0.481. Batch caveat: q1(old 60) 0.415 vs q2(new 100) 0.657. Frozen ensemble is NOT a reliable standalone OOD scorer.
- **Decision:** Update E5.04 inconclusive -> drop. Do not use the frozen ensemble as an OOD gate; adapt (O2.02).
- **Cost:** CPU; 160 cached frozen scores; 0 new GPU scoring
- **Results:** eval/results/exp_fewshot/O2.01.json

## O2.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:40
- **Verdict:** maybe — Factory-grouped curve. pixel-26 logistic: k=10 0.544/err0.489, 20 0.609/0.452, 40 0.650/0.415, 80 0.701/0.389, 160 grouped-CV 0.689 [0.510,0.847]/err0.395. Frozen-view arms stay at chance (platt 0.49-0.56, two-view 0.47-0.52). Best label-efficient recipe = logistic on 26 pixel features; ~40-80 labels matches the 160-label level. Frozen 0-label baseline 0.566/0.454.
- **Decision:** Adapt with the 26-pixel logistic; spend labels on pixel features, not on ensembling frozen views. Operating point ~40-80 labels (AUROC 0.65-0.70, err 0.39-0.42).
- **Cost:** CPU; 160 windows; 20 repeats/k; no GPU
- **Results:** eval/results/exp_fewshot/O2.02.json

## O2.03 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:40
- **Verdict:** maybe — 160-label table: 8 stable / 14 flip / 4 chance (same counts as the 60). Only 2/26 change class: frozen_frac flip->stable, global_shift stable->flip. Sharpness trio stable (radial_slope 0.633/0.635, hf_ratio 0.618/0.657, dct_energy 0.631/0.700 with grouped CI [0.55,0.83]); motion_energy still flips (0.308->0.646). Transferable subset (stable & >=0.10): dct_energy, hf_ratio, radial_slope, luma_p05, illum_b, illum_r, frozen_frac.
- **Decision:** Use the sharpness/spatial-spectrum features for regime routing; do not reuse motion/temporal-band signs. Table robust to doubling labels.
- **Cost:** CPU; 150+160 windows x 26 features
- **Results:** eval/results/exp_fewshot/O2.03.json

## O3.01 — Honest review-vs-OOD-error frontier on the 160-window pre-registered OOD label set
- **Why:** Y1/Y2 cut REVIEW below 60% on a 60-window OOD set (n_bad=9); re-measure the frontier on the larger 160-window OOD set (n_bad=22, 47 factories) and pick a defensible operating point.
- **Hypothesis:** Cutting REVIEW to <=60% cannot improve OOD accuracy on the frozen score; rank-average fusion may help at equal review.
- **Data / label source:** 160 human-labelled Egocentric-10K windows (138 good / 22 bad, 47 factories), pre-registered test; 1,546 unlabelled E8.01 rescored windows (workload)
- **Regime:** Egocentric-10K factory 1920x1080/30
- **Method:** band sweep w=0..0.30; coverage gating c=0.40..1.00; rank-average fusion (no new inference); OOD balanced accuracy on decided windows with factory-grouped 2000x bootstrap CI
- **Owner / started / finished:** O3 /  / 2026-10-10 20:40
- **Verdict:** adopt-with-caveats — Frozen score reproduced on 160/160 (accept 64/reject 54/review 42); OOD AUROC usable_prob 0.566 [0.422,0.702], p_eg2 0.589, p_pixel 0.520, rank-avg 0.563. Band w=0.16 -> review 53.4% OOD BA 0.577 [0.415,0.740]; w=0.17 -> 57.4% BA 0.600 [0.440,0.767]. Gate c=0.45 -> review 55.0% BA 0.629 [0.444,0.822] (1546-cutoff variant 0.587 [0.427,0.752]). Rank-avg lifts OOD BA to 0.690 at gate c=0.45 and 0.708 at band w=0.155-0.16 (review 42.8-44.2%) but AUROC is flat (0.563) -> operating-point effect.
- **Decision:** Ship rank-average fusion + band w=0.16 (review 44.2% on 1,546, OOD BA 0.708 [0.521,0.832]); robust equal-review alt = rank-avg + gate c=0.45 (55.0% review, BA 0.690). On the frozen prob-avg score every <=60%-review point stays at chance (all grouped CIs contain 0.5): the review cut is a workload win, not an accuracy win.
- **Cost:** CPU only, ~1 min; no GPU
- **Results:** eval/results/exp_review160/O3.01.json, eval/results/exp_review160/O3.02.json

## O4.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:43
- **Verdict:** adopt — Z3 transferable-6 per-domain z-score: OOD-160 AUROC 0.759 [0.597,0.886] grouped, BA 0.709; stable-8+6 0.781 [0.634,0.895], BA 0.733; frozen 0.566. Paired grouped gain +0.19/+0.21 (99.5% >0). Gain holds at n_bad=22; q2 alone (0.70/0.73) not separable from chance.
- **Decision:** confirm
- **Cost:** CPU only, 100 q2 decodes (~75s)
- **Results:** eval/results/exp_adapt160/O4.01.json

## O4.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:43
- **Verdict:** adopt — All 6 Z3 transferable candidates stable on OOD-160; 9 stable/10 flip/1 chance across 20 candidates; selection set unchanged.
- **Decision:** confirm
- **Cost:** CPU only
- **Results:** eval/results/exp_adapt160/O4.02.json

## RES.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:49
- **Verdict:** adopt — Encoded resolution can drop to 180p (0.8428, 198 KiB) and 120p (0.8022, 137 KiB) with no AUROC loss vs 0.810 native; no knee down to 120p; all rungs within 0.02 (one CI).
- **Decision:** 
- **Cost:** encode~0.6s+dec0.2s+feat0.5s per 10s window (shared pod)
- **Results:** eval/results/exp_res/RES.01.json

## RES.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:49
- **Verdict:** adopt — Target met: CRF28=0.7960 (dAUROC -0.014) at 57% of CRF23 bytes; CRF33=0.7946 (-0.015) at 36%; no CRF broke 0.02.
- **Decision:** 
- **Cost:** flat ms; bytes 1.14MB->230KB
- **Results:** eval/results/exp_res/RES.02.json

## RES.03 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:49
- **Verdict:** adopt — 360p@CRF28=0.8036 at 257 KiB (best bytes/accuracy, 40% of 480p@CRF23 bytes); 240p@CRF28=0.7934 at 169 KiB inside budget.
- **Decision:** 
- **Cost:** 257 KiB/window @ ~3.0 s/window
- **Results:** eval/results/exp_res/RES.03.json

## RES.04 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:49
- **Verdict:** maybe — Pixel-26 OOD AUROC~0.43 (IndEgo head does not transfer) and is resolution-neutral: 480p d-0.002 [-0.026,+0.019], 360p d+0.013 [-0.034,+0.059]. EG2 view NOT recomputed.
- **Decision:** 
- **Cost:** NVDEC; 60 windows
- **Results:** eval/results/exp_res/RES.04.json

## SKIP.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:27
- **Verdict:** maybe — drop1-start err 0.110 (worst, confirms E3) vs drop1-end 0.057 / drop1-evenly 0.033-0.041; drop2-start 0.189. START-drop biases high, END-drop low (bad rate sec1 0.09 -> sec5 0.58). END/MIDDLE/EVENLY keep 7/7 verdicts but every variant is dominated by uniform-every-2 (0.049 @ 51%). No section drop adopted.
- **Decision:** Do not adopt any section-drop rule; keep every-2. Direction (start worst, end/even safest) is evidence for time-spread sampling.
- **Cost:** CPU-only simulation on cached labels+scores, no GPU
- **Results:** eval/results/exp_skip/SKIP.01.json

## SKIP.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:27
- **Verdict:** drop — strat-probe(1/section) err 0.142 @ 33.1% beats equal-budget contiguous probe 0.211 @ 33.1% and lifts agreement 0.57->0.86, but is not better than uniform-every-3 (0.117 @ 34.9%). dense-where-uncertain x1 0.100 @ 45.9% loses to uniform at equal budget (~0.072); x2 0.057 @ 62.5% also loses. Reproduces E3 CUSUM rejection.
- **Decision:** Drop dense-where-uncertain and stratified-as-primary; uniform every-k grid stays.
- **Cost:** CPU-only, cached scores + labels, no GPU
- **Results:** eval/results/exp_skip/SKIP.02.json

## SKIP.03 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:27
- **Verdict:** adopt — Best cell = uniform-every-2: 51.5% processed, error 0.049 [0.019,0.087], 7/7 verdicts @0.30. Only policy meeting agreement>=0.90 AND error<=0.05; still wins when relaxing error to 0.07/0.10. No section-skip policy is cheaper within tolerance.
- **Decision:** Re-affirm E3.01 uniform-every-2 (optionally + E3.03 4s early exit). No skip-anywhere rule adopted.
- **Cost:** CPU-only synthesis, no GPU
- **Results:** eval/results/exp_skip/SKIP.03.json

## Y1.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:49
- **Verdict:** adopt — Raw symmetric band w=0.16 -> review 53.4% (from 75.5%) at IndEgo gold BA 0.846 (err 0.154, 95% CI 0.662-0.931); exact widest band <60% is w=0.177 (59.6%, err 0.150); w=0.12 gives 37.0% at err 0.153. Calibration does NOT cut review at equal error: in-sample temperature is neutral vs raw (31.6% vs 32.0% min review at gold err<=0.155); the LOCO temperature 'gain' (9.4%) is a per-fold small-sample artifact; Platt/isotonic degrade gold BA to 0.71/0.725. OOD-60 decisions stay 0.34-0.53 BA (near chance, n_bad=9) -> unreliable.
- **Decision:** Ship raw band half-width 0.16 (band [0.34,0.66]) on the frozen score; no calibration; keep OOD near-chance caveat in the verdict.
- **Cost:** CPU only, ~1 min; no GPU
- **Results:** eval/results/exp_review/Y1.01.json

## Y2.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:48
- **Verdict:** adopt — Decision-theoretic gating: gating at coverage 45% (cost 1:1) cuts the 1,546-window REVIEW share 75.5%->55.0%; gold-LOCO BA 0.837 [0.645,0.937] (n=150); OOD-60 eval-only BA 0.496 (near chance, AUROC 0.415, n_bad=9); asymmetric costs 2:1/5:1 hurt (weighted err 0.309/0.382 vs 0.162).
- **Decision:** adopt coverage 0.45, cost 1:1
- **Cost:** CPU, <5 s, cached only
- **Results:** eval/results/exp_review/Y2.01.json

## Y2.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:48
- **Verdict:** adopt — Deterministic stage-0 rules fire on 6/6 of each synthetic black/darken/overexpose/frozen and 0/6 clean/blur/noise/shake/lowbitrate/static; on 150 gold they fire 3x (precision_bad 0.67, n=3, matching the funnel's no-certified-rule result); on the 1,546 they fire 39x (2.5%) and drop review 55.0%->54.0%. Combined pipeline reaches 54.0% review. Calibration (gold-LOCO temperature) is a no-op on share and hurts gold BA -> dropped.
- **Decision:** adopt stage-0 reject overlay; temperature off
- **Cost:** CPU, <5 s, cached only
- **Results:** eval/results/exp_review/Y2.02.json

## Y3.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:58
- **Verdict:** adopt — Baseline reproduced: fixed-C two-view prob-average AUROC 0.8534, nested-LOCO 0.8336. Per-view nested AUROC: eg2 0.8458, perpixel121 0.8140, pixel26 0.7916, series16 0.7912, physics16 0.7608, theory18 0.6698. EG2@1fps/@0.5fps caches absent (only @2fps).
- **Decision:** Use as reference; combo sweep in Y3.02
- **Cost:** CPU <1 min
- **Results:** eval/results/exp_review/Y3.01.json

## Y3.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 16:58
- **Verdict:** drop — NEGATIVE: adding cached views does not cut ambiguity at equal/better AUROC; every extra view raises the review-band mass and AUROC is flat/worse. EG2-alone dominates the two-view baseline (AUROC .834->.846, profile-band .653->.420). Rank-average is a free improvement (.653->.607). All AUROC diffs within CI.
- **Decision:** Adopt rank-average (free); consider EG2-only; reject adding per-pixel/physics/series/theory views
- **Cost:** CPU ~3 min
- **Results:** eval/results/exp_review/Y3.02.json

## Z1.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:25
- **Verdict:** maybe — Sharpness-3 head (radial_slope,hf_ratio,dct_energy) cuts OOD AUROC 0.534->0.706 [0.518,0.867] (frozen 0.415); stable-8 only 0.551; in-domain LOCO drops 0.774->0.578. n_bad=9 exploratory.
- **Decision:** Adopt sharpness-3 as the no-label OOD view; keep full-26 for in-domain.
- **Cost:** CPU ~seconds
- **Results:** eval/results/exp_adapt/Z1.01.json

## Z1.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:25
- **Verdict:** drop — No unsupervised alignment beat raw: coral_t2s 0.547, stable8+coral_s2t 0.678, domain_whiten worse (0.33-0.37), perclip_z 0.283; Ego10K-corpus CORAL no gain (fps confound).
- **Decision:** Do not ship alignment; report negative.
- **Cost:** CPU ~2 min
- **Results:** eval/results/exp_adapt/Z1.02.json

## Z1.03 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:25
- **Verdict:** maybe — Factory-grouped few-shot pixel-26: k=12 ->0.717, k=24 ->0.763 (POST-HOC, optimistic; min n_bad train=1). Self-training on unlabelled Ego10K did not help (<=+0.02). Blend zero-shot+few-shot did not beat few-shot.
- **Decision:** ~12 labels matches the label-free sharpness-3 head (0.706); needs 24 to clearly exceed.
- **Cost:** CPU ~3 min
- **Results:** eval/results/exp_adapt/Z1.03.json

## Z2.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:33
- **Verdict:** adopt — Split-conformal decision sets give a monotone U-shaped review/error frontier under clip-grouped LOCO; empirical coverage tracks nominal. Min review 3.3% at coverage 0.80 (marginal FA 6%); coverage 0.85 gives review 12% with marginal FA 4%<=5% and stated marginal false-decision bound 15%.
- **Decision:** Adopt coverage-0.85 operating point (review 12% vs 75.5% today)
- **Cost:** CPU only, <1 min, seed 7
- **Results:** eval/results/exp_conformal/Z2.01.json

## Z2.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:33
- **Verdict:** adopt — Marginal split-conformal coverage holds (0.80->0.800, 0.90->0.893) because ICC of the conformity score is ~0.02 despite raw-score ICC 0.46; but clip-conditional coverage fails (worst clip 0.647 at nominal 0.90). Mondrian-by-clip lifts clip-conditional coverage to 0.873; Mondrian-by-score-strata near-nominal; cluster-bootstrap over-corrects at K=7.
- **Decision:** Report marginal coverage; use Mondrian-by-clip when per-clip fairness matters
- **Cost:** CPU only, <1 min
- **Results:** eval/results/exp_conformal/Z2.02.json

## Z2.03 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:33
- **Verdict:** drop — With K=7 exchangeable clips a nonparametric cluster UCB cannot certify a 5% false-accept bound at 90% confidence (floor UCB 0.66; EB 1.0; LOCO-honest no fold). Window-Bentkus reaches 5% at 88.7% review but assumes window exchangeability, which intra-clip correlation violates. The guarantee is data-limited by the number of clips, not the tail bound.
- **Decision:** Do NOT ship a 5% CRC bound on 7 clips; recommend the conformal 15% bound and collect more labelled clips for a rigorous r-bound
- **Cost:** CPU only, <1 min
- **Results:** eval/results/exp_conformal/Z2.03.json

## Z3.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:33
- **Verdict:** maybe — 20 candidates: 9 stable / 11 flip. Stable: ct_fsize_std, ct_fsize_cv, inv_clip_frac, inv_laplacian_var, inv_hist_entropy, mo_hand_motion_excess, inv_dct_hf_lf, ct_iframe_frac, ct_fsize_entropy. Flips: absolute bitrate/frame-size, all raw motion residuals (translation AND full affine), noise_snr, dhash.
- **Decision:** advance stable candidates to the combination test
- **Cost:** CPU ~5 min
- **Results:** eval/results/exp_features/Z3.01.json

## Z3.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:33
- **Verdict:** no — Affine fix did NOT restore sign (mo_affine_resid 0.322/0.728 flips, same as translation-only 0.301/0.704). Absolute bitrate flips (0.376/0.802). Survivors: hand-region motion EXCESS (0.649/0.771), compression variability ct_fsize_std (0.620/0.747), invariants clip_frac/laplacian_var/hist_entropy.
- **Decision:** drop raw motion and absolute bitrate; keep hand-excess, size-variability, invariants
- **Cost:** CPU ~1 min
- **Results:** eval/results/exp_features/Z3.02.json

## Z3.03 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:33
- **Verdict:** yes — Raw-feature logistic zero-shot OOD 0.497 (stable8+6) - collapses. Per-domain z-score signed equal-weight: 6-transferable IndEgo 0.684 / OOD 0.852 [0.732,0.950]; stable8+6 IndEgo 0.736 / OOD 0.841 [0.712,0.942]. Frozen ensemble OOD 0.415. OOD few-shot (factory CV) 0.776.
- **Decision:** Adopt per-domain normalised transferable score; cross-domain failure is largely scale/offset, not concept
- **Cost:** CPU ~2 min
- **Results:** eval/results/exp_features/Z3.03.json

## Z4.01 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:30
- **Verdict:** maybe — q75 aggregator best clip AUROC 0.917 (mean 0.833); LOCO clip BA <=0.75 (5/7) for every aggregator -> clip verdicts are discriminative but thin. Uncertainty-targeted review: 35% of windows -> error<=1/7, 65% -> 7/7 (vs every-2 51.5% -> 7/7).
- **Decision:** No single aggregator lifts clip BA above 5/7; use q75/mean only as a ranking signal, not a certified verdict.
- **Cost:** CPU-only on cached labels+scores, no GPU
- **Results:** eval/results/exp_clip/Z4.01.json

## Z4.02 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:30
- **Verdict:** drop — Best tolerated SPRT cell (A=4.9,B=3.7, hard) decides 5/5 correct with 2/7 abstained but at 66.5% of windows -- MORE than every-2's 51.5%. Label-oracle SPRT also 65.5% at 7/7. Soft expected-LLR never accepts (clip-level miscalibration). Sequential early stop does NOT drop review at equal error.
- **Decision:** Keep fixed every-2 grid; use SPRT only as a clip-level abstain gate. Negative result kept.
- **Cost:** CPU-only, cached scores, no GPU
- **Results:** eval/results/exp_clip/Z4.02.json

## Z4.03 — 
- **Why:** 
- **Hypothesis:** 
- **Data / label source:** 
- **Regime:** n/a
- **Method:** 
- **Owner / started / finished:**  /  / 2026-10-10 17:30
- **Verdict:** maybe — Best tolerated cell = A=4.9,B=3.7 hard (5/5 correct, 2/7 abstain, 66.5% windows). Next-100 active labels written (47/47 factories). OOD-60 eval: uncertainty U AUROC 0.349 (anti-correlated) -> uncertainty sampling does not find confidently-wrong domain-shift errors; list is coverage/audit, not guaranteed error reduction.
- **Decision:** Label the top-100 for factory coverage/audit; do not expect them to be the OOD errors. Add domain-shift-aware acquisition before relying on active learning.
- **Cost:** CPU-only, cached features, no GPU
- **Results:** eval/results/exp_clip/Z4.03.json
