# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E0.01 | E0 | done | maybe | Label persistence and position bias on IndEgo gold | IndEgo human gold, 150 windows, 7 clips (human labels) / 480p/10fps IndEgo |
| E1.01 | E1 | done | adopt | IndEgo pixel-feature fps sweep {10,5,3.33,2,1} | IndEgo gold 150 windows human labels 100good/50bad / 480p 10fps H264 in-domain |
| E1.02 | E1 | done | maybe | EG2 window-embedding fps subsample {2,1,0.5} + 5/10fps re-embed | IndEgo gold 150 windows human labels / 480p 10fps in-domain |
| E1.03 | E1 | done | maybe | OOD 1080p30-source fps sweep {30,15,10,5,3,2,1} | 60 human-labelled Egocentric-10K OOD windows (labels/ood_lab / out-of-domain Egocentric-10K 1080p 30fps HEVC, 47 factories, 1 window each |
| E1.04 | E1 | done | drop | Frame-selection strategies at an equal 20-frame budget (IndEgo LOCO) | 150 IndEgo gold windows (human labels, 100 good / 50 bad) fr / in-domain IndEgo 480p 10fps H.264, 7 clips, ~3 wearers, one labeller |
| E1.05 | E1 | running | - | Cost / Pareto re-bench: clean minima for the E1 fps and selection arms | cost only, no labels: 1 IndEgo window (labels/manifest_gold. / in-domain IndEgo 480p 10fps H.264 + out-of-domain Egocentric-10K original 1080p 30fps HEVC, decode CPU cv2 160x90 |
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
| E4.01 | E4 | done | adopt | Regime descriptors: IndEgo vs Egocentric-10K domain-shift diagnosis (re-derived) | 150 IndEgo gold windows (human) + 60 OOD labelled (human, 9  / 480x480/10fps H264 (IndEgo) vs 1080p/30 HEVC (OOD src) vs 854x480/30 (OOD tc) vs 854x480/10 (queue2) |
| E4.02 | E4 | done | drop | Regime-specific vs global models within IndEgo (nested LOCO) | 150 IndEgo gold windows, human labels, leave-one-clip-out ov / terciles of sharpness/lighting/motion |
| E4.03 | E4 | done | drop | Domain-gap control: re-encode to the IndEgo spec (480x480/10fps) | 60 OOD queue-1 windows (human labels, 9 bad); paired source  / 480x480/10 H264 vs 1080p/30 HEVC vs 854x480/30 H264 |
| E4.04 | E4 | done | adopt | Profiles product layer: schema + 5 profiles + deterministic router | synthetic descriptors + E4.01 descriptor ranges (n=730, from / all |
| E5.00 | E5 | running | - | Frozen ensemble OOD prior (usable_prob AUROC on 60 human OOD windows) | 60 Egocentric-10K human-labelled windows (51 good / 9 bad, s / 480p 10s transcodes of 1080p/30fps factory clips; ~1 window/factory |
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
| E9.01 | E9 | done | adopt | standard eval harness for PR comparison | IndEgo gold manifest_gold.json n=150 (100 good/50 bad, 7 cli / IndEgo 480p 10fps H264 in-domain; Egocentric-10K 1080p30 HEVC out-of-domain |

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

## E1.03 — OOD 1080p30-source fps sweep {30,15,10,5,3,2,1}
- **Why:** The OOD regime is original 1080p/30fps HEVC, not IndEgo 480p/10fps; the fps answer may differ per regime and the frozen IndEgo model is known not to transfer, so we also need label-free score stability across fps.
- **Hypothesis:** Frozen IndEgo model AUROC stays 0.4-0.53 at every fps (no transfer); score rank-correlation across fps is high (>=0.8) so the fps choice does not change OOD ranking; per-feature direct AUROC is more stable than the frozen model.
- **Data / label source:** 60 human-labelled Egocentric-10K OOD windows (labels/ood_labels.json: 51 good / 9 bad, human, 1 labeller) decoded from queues/label_queue_random.json source_path+source_start_s (original 1080p HEVC 30fps); plus 100 UNLABELLED queue-2 windows (labels/ood_labels2.json absent) for descriptive score stability only.
- **Regime:** out-of-domain Egocentric-10K 1080p 30fps HEVC, 47 factories, 1 window each
- **Method:** decode each window at 160x90 with cv2 (CPU), index rule idx=arange(0,T,step), steps {30:1,15:2,10:3,5:6,3:10,2:15,1:30}, 26 cpu_window_features with TRUE effective fps=30/step; frozen logistic fit on ALL 150 IndEgo windows at 10fps (C=0.3, balanced, StandardScaler); window-bootstrap CI95 B=2000; spearman score rank-correlation across fps; per-feature signed AUROC.
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 15:49
- **Verdict:** maybe — OOD n=60 (51 good/9 bad human; n_bad=9 EXPLORATORY, CI by window optimistic): frozen IndEgo logistic AUROC 0.3595 [0.180,0.554] at 30fps, 0.4314 [0.237,0.627] at 10fps, 0.5468 [0.373,0.715] at 3fps, 0.5512 [0.375,0.717] at 2fps, 0.5468 [0.379,0.708] at 1fps - every CI95 includes 0.5, no transfer at any fps. Label-free score rank stability (spearman) 3vs2 0.9949, 5vs1 0.9813, 10vs5 0.9763, 30vs1 0.8285. Per-feature direct AUROC is fps-invariant for middle-frame features (dct_energy 0.6993, hf_ratio 0.6863, radial_slope 0.6449, range 0.0000) and collapses for band/rPPG features (rppg_pulse 0.706->0.285). 100 queue-2 windows UNLABELLED: descriptive stability only, no accuracy claimed.
- **Decision:** Do NOT choose the OOD fps from an IndEgo-trained model. OOD fps stays open until an Egocentric-10K-trained head exists (n_bad=9 cannot fit one). For descriptive scoring a cheap fps is acceptable because the frozen ranking is stable below 10fps (rho>=0.98 for 5vs1), with no accuracy claimed. The OOD cost lever is DECODE (3004 ms/window), not fps.
- **Cost:** decode 3004 ms/window clean (10.0 ms/frame; 21.7 ms/frame in the loaded shard run) + features 798 -> 32 ms/window from 30 -> 1 fps; 60 windows x 7 fps; CPU only, no GPU used
- **Results:** eval/results/exp_fps/E1.03.json

## E1.04 — Frame-selection strategies at an equal 20-frame budget (IndEgo LOCO)
- **Why:** Uniform sampling may waste the 20-frame budget on static stretches; motion-adaptive or keyframe-only selection could give the same accuracy for less decode/compute, or better accuracy at equal cost.
- **Hypothesis:** motion-adaptive >= uniform20; keyframe-only is worse than uniform (I-frame placement is arbitrary for content) and degenerates to uniform when a 10s window has fewer than 20 I-frames; seeded random sits between uniform and motion.
- **Data / label source:** 150 IndEgo gold windows (human labels, 100 good / 50 bad) from labels/manifest_gold.json; 480p ~10fps H.264 in-domain.
- **Regime:** in-domain IndEgo 480p 10fps H.264, 7 clips, ~3 wearers, one labeller
- **Method:** equal budget 20 frames/window (~2fps): uniform (idx=arange(0,T,5)) vs motion-adaptive (top-19 inter-frame |diff| + frame0 anchor) vs keyframe-only (I-frame pts_time via ffprobe mapped to the 10fps grid, padded with uniform when <20 I-frames) vs seeded random (seed 0); same LOCO logistic C=0.3 + clip-bootstrap CI95 B=1000; per-window frame overlap (Jaccard) and score rank-correlation vs uniform20.
- **Owner / started / finished:** experimenter / 2026-10-10 15:17 / 2026-10-10 15:49
- **Verdict:** drop — n=150 IndEgo human windows, LOCO logistic C=0.3, equal 20-frame budget: uniform-20 0.7866 [0.650,0.866] bal 0.725 vs motion-adaptive 0.7620 [0.675,0.802], keyframe-only 0.7798 [0.662,0.835], seeded random 0.7386 [0.586,0.805]; all CI95 overlap, no arm beats uniform at equal cost. Motion picks different frames (mean Jaccard vs uniform 0.1347, score rho 0.733) and is worse; keyframe has mean 1.23 I-frames/window and 150/150 windows below the 20-frame budget, so it is always padded back to uniform (Jaccard 0.9244, 63/150 identical). Corrected the bundle's partial e1_04.json, whose keyframe arm silently equalled uniform (ffprobe 6.x: pkt_pts_time is empty).
- **Decision:** Keep uniform sampling (arange(0,T,5)) at the equal 20-frame budget; do not implement motion-adaptive, keyframe-only or random selection. No arm reduces cost: the full native window is decoded in every arm (decode 129 ms/window clean), so selection only changes which 20 frames the features see.
- **Cost:** feature 53 ms/window clean (220 ms/window in the loaded run) at 20 frames/window, identical for all four arms; decode 129 ms/window clean; 0 GPU
- **Results:** eval/results/exp_fps/E1.04.json

## E1.05 — Cost / Pareto re-bench: clean minima for the E1 fps and selection arms
- **Why:** The PR #88 review (F5/F6) found the committed Pareto artefact had no single-command regeneration (run_all.sh omitted --benches) and its cost_note quoted an unsupported 2-4x shard-run multiplier; the pod load also changed since the first bench, so re-measure the clean minima and record the ratio per regime/stage instead of a hand-typed range.
- **Hypothesis:** Re-benching keeps the same Pareto front (IndEgo 10/3.33/2/1 fps + uniform20, OOD 1-2 fps) and the shard-run means stay slower than the clean minima, by a regime/stage-dependent factor of roughly 2-9x.
- **Data / label source:** cost only, no labels: 1 IndEgo window (labels/manifest_gold.json, human) + 1 Egocentric-10K labelled window (labels/ood_labels.json source_path, human); accuracy rows read from E1.01/E1.03/E1.04.json (n=150 in-domain LOCO, n=60 out-of-domain, 51 good/9 bad)
- **Regime:** in-domain IndEgo 480p 10fps H.264 + out-of-domain Egocentric-10K original 1080p 30fps HEVC, decode CPU cv2 160x90
- **Method:** eval/exp_fps/pareto.py --reps 3 --benches 3: min over 3 reps x 3 bench runs of decode + cpu_window_features per fps setting, OMP_NUM_THREADS=1, cv2 threads=1; Pareto front over (AUROC, total ms/window) per regime; clean_vs_shard_ratio computed in-script from E1.01/E1.03/E1.04.json
- **Owner / started / finished:** experimenter / 2026-10-10 16:12 / —
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

## E4.01 — Regime descriptors: IndEgo vs Egocentric-10K domain-shift diagnosis (re-derived)
- **Why:** Every IndEgo-trained scorer failed on real Egocentric-10K windows (0.4-0.53); need deterministic descriptors showing which content descriptors separate the datasets beyond container spec
- **Hypothesis:** resolution/fps/codec separate by construction; content descriptors (sharpness, exposure, ego-motion) also separate but with smaller effects
- **Data / label source:** 150 IndEgo gold windows (human) + 60 OOD labelled (human, 9 bad) on both 480p/30 transcode and 1080p/30 source + 100 queue-2 480p/10 transcodes (unlabelled) + 300 seed-fixed Ego10K sample (unlabelled)
- **Regime:** 480x480/10fps H264 (IndEgo) vs 1080p/30 HEVC (OOD src) vs 854x480/30 (OOD tc) vs 854x480/10 (queue2)
- **Method:** deterministic per-window descriptors (ffprobe + 8 sampled-frame stats, no GPU); Cohen's d ranking
- **Owner / started / finished:** experimenter-a3 / 2026-10-10 15:18 / 2026-10-10 15:30
- **Verdict:** adopt — 730 deterministic descriptor records: 150 IndEgo gold + 120 OOD-60 (480p/30 transcode + 1080p/30 source) + 100 OOD-2 transcodes + 300 seed-7 Ego10K + 60 IndEgo-spec re-encode. Container separators are exact by construction (width/height/fps d=inf); strongest CONTENT separators IndEgo-vs-OOD-source are chroma_spread d=-3.19, bright_frac -2.65, luma_mean -2.57, luma_p95 -2.54, temp_energy -2.37; sharpness only d=0.61 (overlap). ci95 n/a (deterministic).
- **Decision:** Use content separators to justify per-regime profiles; do not treat resolution/fps as findings
- **Cost:** 730 windows x 8 frames, CPU only, no GPU, ~10 min
- **Results:** eval/results/exp_regimes/E4.01.json

## E4.02 — Regime-specific vs global models within IndEgo (nested LOCO)
- **Why:** Decide whether blurry/low-light/walking terciles deserve their own models or thresholds
- **Hypothesis:** Regime-specific models beat global only where regimes have both classes and enough clips
- **Data / label source:** 150 IndEgo gold windows, human labels, leave-one-clip-out over 7 clips
- **Regime:** terciles of sharpness/lighting/motion
- **Method:** logistic (C=0.01, balanced) on 26 per-pixel feats and EG2 embedding; global vs regime-only; 2000x clip-bootstrap CI
- **Owner / started / finished:** experimenter-a3 / 2026-10-10 15:18 / 2026-10-10 15:30
- **Verdict:** drop — No regime split adopted. Under identical LOCO folds regime-only logistic never beat global by >0.03 with a CI supporting it: best feat26 cell lighting_lo +0.125 [-0.10,0.35]; EG2 regime models were worse (delta -0.01..-0.18). Global AUROC feat26 0.712 [0.634,0.811], EG2 0.848 [0.719,0.912] (n=150, 7 clips, 2000x clip bootstrap). Motion terciles are thin (motion_lo n_bad=4).
- **Decision:** Keep one global model per view; no per-regime model; report thin regimes
- **Cost:** 150 windows, CPU, ~10 min
- **Results:** eval/results/exp_regimes/E4.02.json

## E4.03 — Domain-gap control: re-encode to the IndEgo spec (480x480/10fps)
- **Why:** Test whether matching the input container spec fixes OOD transfer or whether the gap is content
- **Hypothesis:** Re-encoding shrinks container descriptors but not content descriptors, and transfer stays poor
- **Data / label source:** 60 OOD queue-1 windows (human labels, 9 bad); paired source 1080p/30 vs 480p/30 transcode vs new 480x480/10 re-encode; frozen ensemble ood_scores.json baseline
- **Regime:** 480x480/10 H264 vs 1080p/30 HEVC vs 854x480/30 H264
- **Method:** re-encode with ffmpeg; paired descriptors; IndEgo-vs-OOD Cohen's d before/after; frozen-ensemble score audit
- **Owner / started / finished:** experimenter-a3 / 2026-10-10 15:18 / 2026-10-10 15:30
- **Verdict:** drop — Re-encoding 60 OOD sources to the IndEgo spec (480x480/10fps H.264) shrinks the IndEgo content gap for only 6/18 content descriptors; several grow (luma_p05 d from -1.34 to +5.02). Frozen-ensemble OOD AUROC 0.415 (n=60, n_bad=9, exploratory) on the 480p/30 transcode was not improved (re-encode not re-scored; frozen pipeline unavailable). Matching the container does not fix the content gap.
- **Decision:** Do not re-encode as a transfer fix; keep container/spec as a router input only
- **Cost:** 60 windows ffmpeg re-encode + descriptors, CPU, ~12 min
- **Results:** eval/results/exp_regimes/E4.03.json

## E4.04 — Profiles product layer: schema + 5 profiles + deterministic router
- **Why:** Builder requires a customisable product that adapts pipeline to whatever data type it receives
- **Hypothesis:** Rule-based router on descriptors selects the right profile with a margin and falls back safely on unseen regimes
- **Data / label source:** synthetic descriptors + E4.01 descriptor ranges (n=730, from eval/results/exp_regimes/E4.01.json; matches the 150+60+60+100+300+60 pool incl. the 60 E4.03 IndEgo-spec re-encodes)
- **Regime:** all
- **Method:** profiles/schema.json, 5 profiles, jadal/regimes/router.py::choose_profile, >=9 unit tests (<30s, no video)
- **Owner / started / finished:** experimenter-a3 / 2026-10-10 15:18 / 2026-10-10 15:50
- **Verdict:** adopt — Deterministic router choose_profile + 5 profiles + schema; 13/13 unit tests pass ('Ran 13 tests ... OK', unittest.TestCase + pytest, regenerateable via eval/exp_regimes/make_results.py). UNSEEN_REGIME fallback now range-checks the descriptors the rules decide on (clip_frac<=0.85, temp_energy<=0.40, static_frac in [0,1]) so e.g. clip_frac=0.9 or temp_energy=2.0 -> default. receipt_fragment() returns a helper dict; it is NOT yet wired into the product QA receipt. Thresholds calibrated on the E4.01 pool n=730 (from E4.01.json; includes the 60 E4.03 re-encodes).
- **Decision:** Ship router + profiles; product selects profile per descriptor regime per window; wire receipt_fragment into the receipt builder in a later change
- **Cost:** 13 unit tests <0.5 s, no video, no network
- **Results:** eval/results/exp_regimes/E4.04.json

## E5.00 — Frozen ensemble OOD prior (usable_prob AUROC on 60 human OOD windows)
- **Why:** The E5 report quotes the frozen-ensemble out-of-domain AUROC 0.415 [0.210,0.647], but that number lived only in docs/reports/ensemble_views.md on main / PR #79, not in this branch or in a committed results JSON. Reproduce it from the frozen bundle scores so every E5 number traces to a committed file.
- **Hypothesis:** The frozen IndEgo ensemble scores ~0.415 AUROC on the 60 human-labelled Egocentric-10K windows (exploratory, n_bad=9); reproducing it from the frozen scores gives the same headline.
- **Data / label source:** 60 Egocentric-10K human-labelled windows (51 good / 9 bad, single labeller, ood_labels.json) scored by frozen ensemble_v1 (bundle caches/ensemble/ood_scores.json); human labels
- **Regime:** 480p 10s transcodes of 1080p/30fps factory clips; ~1 window/factory
- **Method:** load frozen usable_prob; AUROC vs human labels; window-bootstrap 95% CI (optimistic)
- **Owner / started / finished:** experimenter / 2026-10-10 16:12 / —
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

## E9.01 — standard eval harness for PR comparison
- **Why:** PR-triggered agents need one deterministic number to compare a stranger's PR against main; reproduces the documented IndEgo baselines
- **Hypothesis:** LOCO logistic on the 26 per-pixel features reproduces ~0.81 and the EG2 window embedding reproduces ~0.85; the frozen IndEgo model does not transfer OOD (0.4-0.53)
- **Data / label source:** IndEgo gold manifest_gold.json n=150 (100 good/50 bad, 7 clips, human, one labeller); OOD ood_labels.json n=60 (51 good/9 bad, human)
- **Regime:** IndEgo 480p 10fps H264 in-domain; Egocentric-10K 1080p30 HEVC out-of-domain
- **Method:** LOCO logistic + clip-bootstrap CI; frozen-model OOD with window-bootstrap CI; CPU sklearn, no network
- **Owner / started / finished:** infra / 2026-10-10 15:22 / 2026-10-10 15:23
- **Verdict:** adopt — Reproduces both documented baselines on n=150 IndEgo human windows (100 good/50 bad, 7 clips): 26 per-pixel features LOCO logistic C=0.3 -> AUROC 0.8100 (clip-bootstrap 95% CI 0.693-0.887); EG2 window embedding LOCO logistic C=0.01 -> 0.8480 (0.734-0.909). OOD (n=60, n_bad=9, window-bootstrap, optimistic): frozen IndEgo pixel26 model does NOT transfer, 0.4314 at fps=10 (E1.03 protocol); the features.json-trained variant saturates (all p=1.0, reported as 0.5).
- **Decision:** Use eval/standard/run_all.py as the single comparison number for PR-triggered review; exec tier runs it in a throwaway clone behind the run-tests label.
- **Cost:** CPU sklearn, no GPU, ~14-36 s, no network
- **Results:** eval/results/standard/run_all.json
