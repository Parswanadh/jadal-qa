# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E1.01 | E1 | done | adopt | IndEgo pixel-feature fps sweep {10,5,3.33,2,1} | IndEgo gold 150 windows human labels 100good/50bad / 480p 10fps H264 in-domain |
| E1.02 | E1 | done | maybe | EG2 window-embedding fps subsample {2,1,0.5} + 5/10fps re-embed | IndEgo gold 150 windows human labels / 480p 10fps in-domain |
| E1.03 | E1 | done | maybe | OOD 1080p30-source fps sweep {30,15,10,5,3,2,1} | 60 human-labelled Egocentric-10K OOD windows (labels/ood_lab / out-of-domain Egocentric-10K 1080p 30fps HEVC, 47 factories, 1 window each |
| E1.04 | E1 | done | drop | Frame-selection strategies at an equal 20-frame budget (IndEgo LOCO) | 150 IndEgo gold windows (human labels, 100 good / 50 bad) fr / in-domain IndEgo 480p 10fps H.264, 7 clips, ~3 wearers, one labeller |

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
