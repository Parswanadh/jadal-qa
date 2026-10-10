# Pod experiment wave summary

Status: **refreshed from the wave-1 result files after the adversarial-review fixes
(2026-10-10 16:33 UTC)** — supersedes the 16:15 fill. Every number in the table below is
copied from the result JSON named in the last column (or, for E0.01, from the registry
entry — see "Pending / not backable"). Verdicts are the registries' verdicts, copied verbatim
(adopt | maybe | drop | inconclusive); none was upgraded. All 34 experiments are now
`done` (10 adopt / 11 maybe / 12 drop / 1 inconclusive).

Sources: `experiments/registry.jsonl` — the append-only union of
`experiments/registry_snapshot.jsonl` and the live per-worktree registries, merged with
`python3 tools/pod/union_registry.py` (34 experiments; the four review-corrected
registries — exp-fps-pod, exp-mosaic, exp-domain-adapt, exp-rescore — listed last so they
win) — the per-experiment pages written
by `tools/exp.py new` (`experiments/<id>/README.md`), and the result JSON named in the
last column. The rendered log is `experiments/EXPERIMENTS.md`.

Result-file paths are **pod paths inside this repo's other worktrees** (they are results
files, not frames: no video is committed). Short prefixes below:
`fps:` = `/workspace/wt/exp-fps-pod/eval/results/exp_fps/`,
`mosaic:` = `/workspace/wt/exp-mosaic/eval/results/exp_mosaic/`,
`regimes:` = `/workspace/wt/exp-regimes-pod/eval/results/exp_regimes/`,
`domain:` = `/workspace/wt/exp-domain-adapt/eval/results/exp_domain/`,
`motion:` = `/workspace/wt/exp-motion-cue/eval/results/exp_motion_cue/`,
`cost:` = `/workspace/wt/exp-cost-model/eval/results/exp_cost/`,
`std:` = `/workspace/wt/prwatch/eval/results/standard/`,
`rescore:` = `/workspace/wt/exp-rescore/eval/results/exp_rescore/`, and
`bundle:` = `/workspace/jadal_pod_bundle/local_work/exp-temporal/eval/results/exp_temporal/`
(E3; the registry points at `/workspace/exp-temporal/...` which is the same bundle run).
Until those files are merged into this branch, the in-repo pointer to every one of these
numbers is its `experiments/registry.jsonl` entry, which carries the same `summary` and a
`results` field naming the file.

## Wave 1

| id | verdict | headline number with CI | cost | regime | file |
|---|---|---|---|---|---|
| E0.01 | maybe | Neighbour-label agreement 0.73 vs 0.56 chance (n=143 pairs, ~15 s apart); bad rate first 120 s 0.19 vs 0.42 later — start is cleaner, skip-start NOT supported. No CI (label-only). n=150 / 7 clips, human (1 labeller). | none (labels only) | 480p/10fps IndEgo | experiments/registry.jsonl (E0.01) — no results JSON on the pod |
| E1.01 | adopt | Pixel-view LOCO AUROC n=150 (human, 1 labeller, IndEgo): 10fps 0.809 [0.693,0.890], 3.33fps 0.807 [0.697,0.873] (within 0.02 at 34 vs 100 frames), 5fps 0.793, 2fps 0.787, 1fps 0.774. | decode 154 ms/window + features 238/119/80/47/26 ms at 10/5/3.33/2/1fps, CPU | 480p 10fps H.264 in-domain | fps: e1_01.json |
| E1.02 | maybe | EG2 window-embedding LOCO AUROC n=150 (human, 1 labeller, IndEgo): 2fps 0.848 [0.734,0.909], 1fps 0.860 [0.760,0.914], 0.5fps 0.870 [0.763,0.924] (CIs overlap; fewer frames NOT worse). Two-view prob-avg 0.852–0.854. | 0 GPU (cached embeddings), CPU eval | 480p 10fps in-domain | fps: e1_02.json |
| E1.03 | maybe | OOD n=60 (51 good / **9 bad**, human, 1 labeller; exploratory): frozen IndEgo logistic AUROC 0.360 [0.180,0.554] @30fps, 0.431 [0.237,0.627] @10fps, 0.547 [0.373,0.715] @3fps, 0.551 [0.375,0.717] @2fps, 0.547 [0.379,0.708] @1fps (0.409 @15fps, 0.499 @5fps) — every CI includes 0.5, no transfer at any fps. rppg_pulse collapses 0.706 @30fps → 0.285 @2fps → 0.500 @1fps (constant feature there). | decode 2900.4 ms/window clean (9.67 ms/frame) / 6495.7 ms/window in the loaded shard run; features 843.0→32.4 ms/window clean (30→1 fps; 3312.4→168.9 loaded), CPU only | OOD Egocentric-10K 1080p 30fps HEVC | fps: E1.03.json (cost from fps: pareto.json) |
| E1.04 | drop | Equal 20-frame budget, LOCO n=150 (human): uniform-20 0.787 [0.650,0.866] bal 0.725, motion-adaptive 0.762 [0.675,0.802], keyframe-only 0.780, seeded random 0.739; all CIs overlap, no arm beats uniform. Motion20's CI is ~40% narrower (width 0.127 vs 0.215) — recorded, not evidence (its point estimate is lower). | feature 53.7 ms/window clean (219.9 loaded) + decode 112.0 ms/window clean (568.8 loaded), 0 GPU, identical across arms | in-domain IndEgo 480p 10fps H.264 | fps: E1.04.json (cost from fps: pareto.json) |
| E1.05 | maybe | Cost / Pareto re-bench, no labels (n=1 IndEgo window + 1 Egocentric-10K 1080p window; 3 reps × 3 bench runs, min-of-9, host load 17.6): clean minima in-domain 480p decode 112.0 ms/window (1.12 ms/frame) + features 288.9/166.7/102.2/53.7/32.4 ms/window at 10/5/3.33/2/1 fps; OOD 1080p HEVC decode 2900.4 ms/window (9.67 ms/frame) + features 843.0/391.8/246.5/128.7/74.9/55.5/32.4 ms/window at 30/15/10/5/3/2/1 fps. Pareto fronts unchanged (in-domain 10/3.33/2/1 fps + uniform20; OOD 1–2 fps). Shard-run means in E1.01/E1.03/E1.04.json are 2.24–8.01× these clean minima (regime/stage-dependent). | ~80 s wall, 1 core, no GPU | in-domain IndEgo 480p/10 + OOD 1080p/30 HEVC | fps: pareto.json |
| E2.01 | drop | Mosaic N9 LOCO n=150 (human): 0.785 [0.643,0.860]; full sweep N=4..100 spans 0.633–0.842, always below matched mean-of-frames (deltas −0.001…−0.214); **CIs overlap at every N**, so the drop is the E2 point rule on a consistent direction, not a significant difference. OOD n=60 (**9 bad**): 0.329 [0.170,0.488] (range 0.155–0.638). | 1 embed pass/window vs N (≤1/4 for N≥4); 224 px tiles (N=9 PNG measured 349–743 KB); A40 VRAM peak 3967 MB | IndEgo 480p10 + Egocentric-10K | mosaic: E2.01.json |
| E2.02 | adopt | Mean-of-frames N9 LOCO n=150 (human): 0.852 [0.752,0.908] (N16 0.860), reproduces the cached EG2 reference 0.848. OOD n=60 (**9 bad**): 0.628 [0.383,0.854]. | N embed passes/window; A40 | same as E2.01 | mosaic: E2.02.json |
| E2.03 | drop | Shuffled tile order N9 LOCO n=150 (human): 0.801 [0.641,0.886] vs ordered 0.785 — delta +0.016 inside the CI. OOD n=60 (**9 bad**): 0.423. | 1 embed pass/window; A40 | same as E2.01 | mosaic: E2.03.json |
| E2.04 | drop | No-timestamp mosaic N9 LOCO n=150 (human): 0.810 [0.677,0.882] vs with-timestamps 0.785 — delta +0.025 inside the CI. OOD n=60 (**9 bad**): 0.444. | 1 embed pass/window; A40 | same as E2.01 | mosaic: E2.04.json |
| E2.05 | maybe | Single middle frame LOCO n=150 (human): 0.784 [0.685,0.849], 0.068 below mean-of-frames N9 0.852. OOD n=60 (**9 bad**): 0.514. | 1 embed pass/window; A40 | same as E2.01 | mosaic: E2.05.json |
| E2.06 | maybe | Motion mosaic \|I_t−I_{t−1}\| N9 LOCO n=150 (human): 0.863 [0.783,0.909], best in-domain variant (mean-of-frames 0.852). OOD n=60 (**9 bad**): 0.401 (fails transfer). | 1 embed pass/window; A40 | same as E2.01 | mosaic: E2.06.json |
| E2.07 | drop | Pooled per-tile stats concat(mean,std) N9 LOCO n=150 (human): 0.826 [0.690,0.882], does not beat mean-of-frames. OOD n=60 (**9 bad**): 0.586. | 9 embed passes/window; A40 | same as E2.01 | mosaic: E2.07.json |
| E3.01 | adopt | Every-2nd window, IndEgo n=150 / 7 clips (human, label oracle): clip bad-fraction error 0.049 [0.019,0.087], 7/7 verdicts @30%, delay 0.3 windows at 51% cost; every-3 0.117, every-4 0.124; skip-first-M 0.088→0.418. | simulated 0.51 footage @every-2 (CPU-only sim, no GPU) | 480p/10fps IndEgo assembly+logistics | bundle: E3.01_policies.json |
| E3.02 | drop | No position effect on OOD: Mann-Whitney bad-vs-good p=0.42 raw / 0.47 normalised (n=60, **9 bad**); medians 70 s vs 90 s. Skip-the-start not justified either regime. | none (labels + manifest arithmetic) | 1080p/30fps Egocentric-10K factory | bundle: E3.02_ood_position.json |
| E3.03 | adopt | EG2-prefix LOCO AUROC n=150 (human): 2 s 0.804, 4 s 0.860, 6 s 0.869, 8 s 0.857, 10.5 s 0.848 — saturates at 4–6 s; 4 s within 0.01 of peak. | 2.5× window-cost cut (4 s/10 s); CPU LOCO seconds | 480p/10fps IndEgo | bundle: E3.03_early_exit.json |
| E3.04 | adopt | 4.404 h corpus (Egocentric-10K manifest, unlabelled): every-2 → 2.27 h (×1.9); every-2 + 4 s exit → 0.91 h (×4.9); every-3 + 4 s → ×7.2 (decode-bound upper bound). | arithmetic, no compute | 1080p/30fps Egocentric-10K factory | bundle: E3.04_compute.json |
| E3.05 | adopt | Adopted IndEgo policy: probe every 2nd window + 4 s early exit ≈ 0.20 footage processed (×5); dropped CUSUM-as-sampler, skip-first-M, random grids; OOD has no certified policy (transfer fails; **9 bad**). | synthesis only | IndEgo + Egocentric-10K | bundle: E3.05_decision.json |
| E4.01 | adopt | 730 deterministic descriptor records; container separators exact by construction; strongest content separators IndEgo-vs-OOD-source chroma_spread d=−3.19, bright_frac −2.65, luma_mean −2.57, temp_energy −2.37; sharpness only d=0.61 (overlap). n/a CI (deterministic). | 730 windows × 8 frames, CPU only, ~10 min | mixed 480×480/10 H264 vs 1080p/30 HEVC | regimes: E4.01.json |
| E4.02 | drop | No regime split adopted: best feat26 cell lighting_lo +0.125 [−0.10,0.35]; EG2 regime models worse. Global feat26 0.712 [0.634,0.811], EG2 0.848 [0.719,0.912] (n=150, 7 clips, human, 2000× clip bootstrap). | 150 windows, CPU, ~10 min | terciles of sharpness/lighting/motion | regimes: E4.02.json |
| E4.03 | drop | Re-encode to the IndEgo spec shrinks the content gap for only 6/18 descriptors (some grow); frozen-ensemble OOD AUROC 0.415 (n=60, **9 bad**, exploratory) not improved (re-encode not re-scored). | 60 windows ffmpeg re-encode + descriptors, CPU, ~12 min | 480×480/10 vs 1080p/30 vs 854×480/30 H264 | regimes: E4.03.json |
| E4.04 | adopt | Deterministic router `choose_profile` + 5 profiles + schema; 13/13 unit tests pass (`Ran 13 tests ... OK`); UNSEEN_REGIME→default with range-checks; receipt_fragment NOT yet wired into the QA receipt. n/a CI (deterministic). | 13 unit tests <0.5 s, no video, no network | all | regimes: E4.04.json |
| E5.00 | maybe | Frozen ensemble_v1 usable_prob vs 60 human OOD windows (51 good / **9 bad**, 1 labeller, exploratory): AUROC 0.4150 [0.201,0.656] (window-bootstrap, optimistic); bal.acc@0.5 0.471. Reproduces the 0.415 prior from the frozen bundle scores. | CPU <1 s; 60 frozen scores; no GPU | 480p 10 s transcodes of 1080p/30 factory clips | domain: E5.00.json |
| E5.01 | maybe | Pooled AI-frame heads on 150 human IndEgo windows (LOCO): work-prob 0.813 [0.695,0.864], hands-prob 0.818 [0.685,0.893], both 0.826 [0.717,0.881], 12-feat logistic 0.789 [0.686,0.862] — below the 0.85 EG2-window-embedding baseline. The window-level 0.789–0.826 **IS** scored against the human IndEgo gold labels; the frame-level 0.911/0.957 is AI-vs-AI (model-vs-model, **NOT accuracy**). | CPU ~2 min; 10k AI frames + 150 windows | head frames 1080p/30 factory stills; eval IndEgo 480p10 | domain: E5.01.json |
| E5.02 | maybe | 60 OOD windows (51 good / **9 bad**, human, 47 factories, grouped CV): pixel-26 0.756 [0.575,0.902] bal.acc@0.5 0.634; EG2-window-mean 0.546 [0.350,0.736] bal 0.520; two-view avg 0.767 [0.596,0.904] bal 0.614. **Trained ON the OOD labels (few-shot), not transfer.** Learning curve pixel-26 (C=1): ~12 → 0.709 (std 0.10; repeats 17/20/20/19 at 12/24/36/45 → 0.709/0.715/0.703/0.691); smallest n_bad in any training fold = 7; grouped folds ~1 window/factory so CV is optimistic. | CPU; 60 windows; 26 feats / 768 EG2 means | 480p transcodes of 1080p/30 factory clips | domain: E5.02.json |
| E5.03 | maybe | 26 per-pixel features, signed AUROC IndEgo (n=150, clip-CI) vs OOD (n=60, **9 bad**, window-CI, 10fps-effective): 8 sign-stable, 14 flip (motion_energy 0.308→0.721, skin_frac 0.667→0.257), 4 chance — chance iff BOTH \|AUROC−0.5\|<0.05, else stable/flip by sign. Sharpness/spatial-spectrum trio (radial_slope 0.633/0.645, hf_ratio 0.618/0.693, dct_energy 0.631/0.702) is sign-stable but still weak OOD (0.64–0.70); illumination does not hold as a group (illum_g flips, illum_r stable only at chance level). Native-30 class stored per row: 3/26 disagree with the 10fps class. | CPU; 150+60 windows × 26 features | IndEgo 480p10 vs Egocentric-10K | domain: E5.03.json |
| E5.04 | inconclusive | ood_labels2.json NOT delivered: n=0 human labels, ci95=n/a; the 160 scored windows are **UNLABELLED** (`label_source` none). Descriptive only: frozen ensemble_v1 usable_prob mean 0.560 (median 0.460), accept 71 / review 35 / reject 54 (batch1 0.496, batch2 0.598). **No accuracy computed.** | 0 new GPU scoring (frozen scores cached) | 480p 10 s transcodes of 1080p/30 clips | domain: E5.04.json |
| E6.01 | drop | n=150 (human, 1 labeller): ego-motion magnitude oriented AUROC 0.805 [0.675,0.893] (unoriented 0.195); residual-motion cue oriented 0.692; Spearman partial vs label (controlling ego-motion) −0.102 [−0.347,+0.072] (CI includes 0). | CPU-only: 100 frames/window, 150 windows in 98.6 s = 657 ms/window; A40 GPU idle | 480p/10fps IndEgo mp4_480 | motion: E6.01.json |
| E6.02 | drop | n=150, 7 clips: LOCO raw pooled 0.692 [0.477,0.839]; within-clip −\|z\| 0.606 [0.525,0.708]; majority-prior sign 0.610 [0.449,0.767]. Oracle per-clip sign (upper bound only) mean 0.757. | no extra extraction; 2000-draw bootstraps × 4 variants × 4 rules; A40 CPU | 480p/10fps IndEgo | motion: E6.02.json |
| E6.03 | drop | OOD n=60 (51 good / **9 bad**): IndEgo-frozen direction resid_mean AUROC 0.281 [0.145,0.434] (worse than chance); Spearman +0.271 (opposite sign to in-domain −0.314). | 60 windows, 95.6 s; A40 CPU-only extraction | 480p/30 transcodes of 1080p/30 factory clips | motion: E6.03.json |
| E6.04 | drop | n=150: within-stratum oriented AUROC across 4 schemes × 4 variants spans 0.503–0.756 (interpretable strata ≥3 bad); LOCO logistic cue-only 0.660, ego-only 0.798, cue+ego 0.793 (cue adds −0.005). | no extra extraction; 4×4×2000-draw bootstraps; A40 CPU | 480p/10fps IndEgo | motion: E6.04.json |
| E7.01 | adopt | Cost table on A40: per-profile 0.51–15.22 video-h/wall-h; every-2 + 4 s exit pixel26 565 ms/window → ×3.88 vs full; best profile walking_logistics 15.22 video-h/wall-h (657 ms/window); feature/embedding-bound. n/a labels (cost model). | timed measurement ~9 min wall, peak 1.5 GB VRAM; A40 | 480p/10 IndEgo + 1080p/30 HEVC Egocentric-10K | cost: E7.01.json |
| E8.01 | maybe | n=1546/1546 windows (58 clips, 4.404 h) re-scored, 0 decode failures; `label_source` **none** — the 1,546 windows are UNLABELLED and every verdict is DESCRIPTIVE, not a quality measure. Verdicts accept 296 (19.15%) / reject 83 (5.37%) / review 1167 (75.49%); accept-share 95% CI [0.136,0.254] (clip bootstrap over 58 clips). 0.683 windows/s, 5761 ms/window mean (p50 5195, p95 9371); profiles factory_1080p30 1540 / default 6 (**UNSEEN_REGIME 6/1546**, 0.39%: 3 extreme ego-motion, 3 overexposed). 1546/1546 receipts+digests re-verified. | 4 workers × 2263 s wall (37.7 min) on a shared A40; frozen recipe ensemble_v1 (0.8534 in-domain full-OOF) | factory Egocentric-10K 1080p/30 HEVC | rescore: E8.01.json |
| E9.01 | adopt | Reproduces both baselines on n=150 human IndEgo windows (LOCO): pixel-26 0.8100 [0.693,0.887], EG2 0.8480 [0.734,0.909]. OOD (n=60, **9 bad**, window-bootstrap optimistic): frozen IndEgo model does not transfer — E1.03 reference 0.4314 [0.237,0.627] @fps=10; this harness's features.json variant saturates to 0.5 (all p=1.0, reported not hidden). | CPU sklearn, no GPU, ~14–36 s, no network | IndEgo in-domain + Egocentric-10K OOD | std: run_all.json |

**Pending / not backable:** nothing is pending — all 34 experiments are `done`. Two
caveats: E0.01 has no results JSON on the pod (only
`jadal_pod_bundle/experiments_local/E0.01/README.md`), so its row cites the registry entry,
not a results file; and the E3 result files live in the pod bundle
(`/workspace/jadal_pod_bundle/local_work/exp-temporal/...`), not in this branch. The E8.01
full contract (1,546 windows) is `/workspace/artifacts/windows_rescored.jsonl` (not in git:
`artifacts/` is ignored); only the 20-line sample
`rescore: windows_rescored_sample.jsonl` is committed. One number disagrees with its
registry summary and was taken from the results file: the E2.01 mosaic in-domain sweep spans
**0.633–0.842** over N=4..100 (`mosaic: E2.01.json`; N100 = 0.633), not the "0.757–0.842"
written in the registry entry.

Reproduced baselines (reference the wave is compared against, not wave results; n = 150
human-labelled IndEgo windows, leave-one-clip-out):

| baseline | number | file |
|---|---|---|
| 26 per-pixel features + logistic (C=0.3) | AUROC 0.810 [0.693,0.887] | `experiments/baselines/2026-10-10_baseline_logistic_loco.txt`; reproduced in `std: run_all.json` |
| EmbeddingGemma-2 window embedding + logistic (C=0.01) | AUROC 0.848 [0.734,0.909] | `experiments/baselines/2026-10-10_ensemble_views_exploratory.txt`; reproduced in `std: run_all.json` |
| probability average of the two views | AUROC 0.853 [0.731,0.922] | `experiments/baselines/2026-10-10_ensemble_views_exploratory.txt` |

## Builder decisions the wave supports

- **Profile.** Ship the deterministic router + 5 profiles (`blurry_lowlight`,
  `clear_stationary`, `default`, `factory_1080p30`, `walking_logistics`) that route on
  measured descriptors (E4.01, n=730) and fall back safely to default (E4.04, 13/13 tests,
  `regimes: E4.04.json`). Do **not** split IndEgo into per-regime models: no split beat the
  global model with a CI supporting it (E4.02). Keep one global model per view.
- **Sampling / early exit.** For IndEgo-like data adopt probe every-2nd window + 4 s early
  exit (~0.20 footage, ×5); every-3rd is the half-cost fallback; drop CUSUM-as-sampler and
  skip-first-M (E3.01/E3.05). No skip rule for Egocentric-10K (E3.02, n_bad=9).
- **Frame selection.** Keep uniform sampling at the equal 20-frame budget; motion-adaptive,
  keyframe-only and random selection are dropped (E1.04).
- **fps.** Pixel view: 3.33 fps is within 0.02 of full (E1.01); EG2 needs no denser than
  2 fps (E1.02). The OOD fps answer stays open (E1.03; the frozen model does not transfer).
  Cite `fps: pareto.json` (E1.05) for every cost number; the shard-run means in E1.01/E1.03/
  E1.04 are 2.24–8.01× the clean minima, so never mix the two.
- **Mosaic.** Do not use the mosaic as a model input (E2.01/E2.03/E2.04/E2.07); keep the
  mean-of-frames EG2 as the model-input baseline (E2.02) and ship the mosaic PNG + sidecar
  only as the human "why" artefact. Motion mosaic is in-domain-only, never claimed to
  transfer (E2.06).
- **Open questions.** OOD transfer (E5.04 confirmatory eval still blocked on
  `ood_labels2.json`; E8.01's descriptive re-score of 1,546 windows is done but those
  windows are unlabelled — no accuracy); an Egocentric-10K-trained head (needs >9 bad
  labels); the OOD fps/cost lever (E1.03/E1.05: decode-bound, 2900 ms/window clean at
  1080p/30); whether the motion/stability cue adds anything
  beyond ego-motion magnitude (E6.04: it does not, on n=150); re-time E7.01 on an idle pod
  and measure a pipelined multi-stream + batch-8 path.

## The honesty ceiling (standing paragraph, do not drop)

The numbers above are measured on **150 ten-second windows from 7 IndEgo clips and ~3
wearers, labelled by one human** ([README.md](../../README.md), [`docs/BASELINE.md`](../BASELINE.md)
line 16: gold labels "reflect one labeller on a few IndEgo workers"), so they are
descriptive of this small set, not of head-camera video in general; window-level accuracy
is **not high** yet. All in-domain CIs are clip-bootstrap (7 clips) and wide; all
out-of-domain numbers are on **n_bad = 9** out of 60 human-labelled Egocentric-10K windows
(51 good / 9 bad, one labeller) with a **window-bootstrap CI, which is optimistic** (about
one window per factory), so every out-of-domain statement is exploratory and a "no effect"
result there is absence of evidence, not evidence of absence ([AGENTS.md](../../AGENTS.md)
rule 1). IndEgo-trained scorers did **not** transfer out of domain (**0.4–0.53 AUROC**;
e.g. E1.03 0.431 [0.237,0.627] at 10 fps; E5.00 0.415 [0.201,0.656]; E9.01 documents the
same 0.4–0.53 range) —
never claim transfer that has not been tested. **E8.01's accept 19.15% / reject 5.37% /
review 75.49% over 1,546 Egocentric-10K windows are DESCRIPTIVE rates on UNLABELLED video**
(`label_source` none, `rescore: E8.01.json`), produced by the same non-transferring
IndEgo-trained recipe, so they are **not** a quality measure. Two numbers are especially easy to misread:
**E5.01's frame-level 0.911/0.957 is model-vs-model agreement between an AI-labelled head
and its own AI targets (gemini-2.5-flash), NOT ground-truth accuracy** — while its
window-level 0.789–0.826 **IS** scored against the human IndEgo gold labels
(`domain: E5.01.json`,
`"caveat": "the frame-level 0.911/0.957 is AI-vs-AI (model-vs-model), NOT ground-truth; the window-level 0.789-0.826 IS scored against the human IndEgo gold labels (single labeller) but the head itself was trained on AI labels"`), and **E5.02's
~0.75–0.77 is a few-shot model trained ON the OOD human labels, not transfer** (n_bad=9,
one window per factory, `domain: E5.02.json`). The motion/stability cue predicts *usable*
windows; **part of that cue may be head stability, it is not "hand work is detected"**, and
its strongest scalar is ego-motion magnitude (E6.01: partial correlation controlling for
ego-motion −0.10 [−0.35,+0.07]; E6.04: cue adds −0.005 over ego-motion alone). Receipts are
the **content hash** (SHA-256 of canonical JSON) of the decision, **not signed**
([README.md](../../README.md), `jadal/receipts.py`); the cameras are RGB, **no
hyperspectral** data ([README.md](../../README.md) line 18); the pipeline has **never been
measured on a Raspberry Pi** ([`docs/reports/edge.md`](../reports/edge.md)). E7.01 cost
numbers are from a **contended** shared pod (A40) and its CIs are window-variation only;
E4.04's `receipt_fragment` is **not yet wired into the product receipt**.
