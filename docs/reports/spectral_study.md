# Spectral study — harness-scored comparison

_Generated 2026-10-10T10:09:57Z by eval/spectral/score.py. Every number below comes from eval/results/spectral/summary.json._

## Question

Does per-pixel frequency/temporal-spectral analysis of RGB video (arm B), a bounded hyperspectral check (arm C), a generic VLM (arm D), or the EmbeddingGemma-2 head (arm E) beat the current OpenCV+MediaPipe L0 signals (arm A) at (1) near-duplicate detection and (2) deciding whether a 10-s segment is USABLE work footage — within a 10,000 h in 2–3 h throughput budget, whole-video single-pass? Sampling 4–8 frames per segment is NOT acceptable as the main method.

## Data & protocol

- HUMAN GOLD (live IndEgo labels, snapshot 2026-10-10T09:53:36.001340+00:00, sha256 0bc9e0b8f348…, n_good=65, n_bad=14). CV: leave-one-clip-out; AUROC with clip-level bootstrap CI (B=1000). A CI spanning 0.5 means 'no evidence'. Awign data differs: nothing here is evidence on Awign footage.

- SYNTHETIC (by construction, label says nothing about activity): good-labelled segments + 9 degradations; sensitivity at 5% FAR threshold set on gold-good scores.

- DUPLICATES (by construction): threshold on held-out half 1, P/R/F1 on half 2, per method and per perturbation. Horizontal-flip excluded (NOT a duplicate).

## Results per task

| arm | gold AUROC (95% CI) | bal.acc | synth sens. (median over degs) | dups F1 | status |
|---|---|---|---|---|---|
| A | 0.457 (0.375–0.521) | 0.509 | 0.17 | 0.865 | RUN |
| A-full | 0.675 (0.6–0.8) | 0.642 | 0.0 | — | RUN |
| E | 0.596 (0.505–0.817) | 0.553 | 0.67 | 0.909 | RUN |
| B | — | — | — | — | NOT RUN |
| D | — | — | — | — | NOT RUN |
| B-T0 | 0.43 (0.359–0.683) | 0.448 | — | — | RUN |
| B-T1 | 0.481 (0.41–0.667) | 0.448 | — | — | RUN |

### Gold detail (per arm, LOCO folds)

**A** n=79 (good=65, bad=14, clips=3): `{"head": {"auroc": 0.457, "auroc_ci95": [0.375, 0.521], "n_boot": 1000, "balanced_accuracy": 0.509, "folds": [{"test_clip": "desk", "n_test": 17, "train_threshold": 0.781}, {"test_clip": "ecu_full", "n_test": 48, "train_threshold": 0.7298}, {"test_clip": "ecu_late", "n_test": 14, "train_threshold": 0.7882}]}}`

**A-full** n=79 (good=65, bad=14, clips=3): `{"head": {"auroc": 0.675, "auroc_ci95": [0.6, 0.8], "n_boot": 1000, "balanced_accuracy": 0.642, "folds": [{"test_clip": "desk", "n_test": 17, "train_threshold": 0.7085}, {"test_clip": "ecu_full", "n_test": 48, "train_threshold": 0.7639}, {"test_clip": "ecu_late", "n_test": 14, "train_threshold": 0.8258}]}}`

**E** n=79 (good=65, bad=14, clips=3): `{"head": {"auroc": 0.596, "auroc_ci95": [0.505, 0.817], "n_boot": 1000, "balanced_accuracy": 0.553, "folds": [{"test_clip": "desk", "n_test": 17, "train_threshold": 0.9788}, {"test_clip": "ecu_full", "n_test": 48, "train_threshold": 0.9744}, {"test_clip": "ecu_late", "n_test": 14, "train_threshold": 0.9757}]}}`

**B-T0** n=79 (good=65, bad=14, clips=3): `{"head": {"auroc": 0.43, "auroc_ci95": [0.359, 0.683], "n_boot": 1000, "balanced_accuracy": 0.448, "folds": [{"test_clip": "desk", "n_test": 17, "train_threshold": 0.6659}, {"test_clip": "ecu_full", "n_test": 48, "train_threshold": 0.5208}, {"test_clip": "ecu_late", "n_test": 14, "train_threshold": 0.5973}]}, "as_is": {"auroc": 0.637, "auroc_ci95": [0.467, 0.693], "n_boot": 1000}, "score_logistic": {"auroc": 0.324, "folds": [{"test_clip": "desk", "n_test": 17, "train_threshold": 0.8631}, {"test_clip": "ecu_full", "n_test": 48, "train_threshold": 0.7272}, {"test_clip": "ecu_late", "n_test": 14, "train_threshold": 0.8931}]}}`

**B-T1** n=79 (good=65, bad=14, clips=3): `{"head": {"auroc": 0.481, "auroc_ci95": [0.41, 0.667], "n_boot": 1000, "balanced_accuracy": 0.448, "folds": [{"test_clip": "desk", "n_test": 17, "train_threshold": 0.6814}, {"test_clip": "ecu_full", "n_test": 48, "train_threshold": 0.5566}, {"test_clip": "ecu_late", "n_test": 14, "train_threshold": 0.5384}]}, "as_is": {"auroc": 0.678, "auroc_ci95": [0.5, 0.684], "n_boot": 1000}, "score_logistic": {"auroc": 0.364, "folds": [{"test_clip": "desk", "n_test": 17, "train_threshold": 0.8584}, {"test_clip": "ecu_full", "n_test": 48, "train_threshold": 0.7265}, {"test_clip": "ecu_late", "n_test": 14, "train_threshold": 0.8926}]}}`

### Synthetic detail (sensitivity at 5% FAR)

**A** `{"status": "RUN", "threshold_5pct_FAR_on_gold_good": 0.5695, "note": "threshold t = 5th pct of gold-good scores; item bad iff score < t. Says NOTHING about activity judgement.", "per_degradation": {"black": {"n": 6, "sensitivity_at_5pct_FAR": 0.0, "control_FAR": null}, "blur": {"n": 6, "sensitivity_at_5pct_FAR": 0.0, "control_FAR": null}, "clean": {"n": 6, "sensitivity_at_5pct_FAR": null, "control_FAR": 0.0}, "darken": {"n": 6, "sensitivity_at_5pct_FAR": 1.0, "control_FAR": null}, "frozen": {"n": 6, "sensitivity_at_5pct_FAR": 0.167, "control_FAR": null}, "lowbitrate": {"n": 6, "sensitivity_at_5pct_FAR": 0.333, "control_FAR": null}, "noise": {"n": 6, "sensitivity_at_5pct_FAR": 1.0, "control_FAR": null}, "overexpose": {"n": 6, "sensitivity_at_5pct_FAR": 0.0, "control_FAR": null}, "shake": {"n": 6, "sensitivity_at_5pct_FAR": 0.167, "control_FAR": null}, "static": {"n": 6, "sensitivity_at_5pct_FAR": 1.0, "control_FAR": null}}}`

**E** `{"status": "RUN", "threshold_5pct_FAR_on_gold_good": 0.976, "note": "threshold t = 5th pct of gold-good scores; item bad iff score < t. Says NOTHING about activity judgement.", "per_degradation": {"black": {"n": 6, "sensitivity_at_5pct_FAR": 1.0, "control_FAR": null}, "blur": {"n": 6, "sensitivity_at_5pct_FAR": 0.667, "control_FAR": null}, "clean": {"n": 6, "sensitivity_at_5pct_FAR": null, "control_FAR": 0.167}, "darken": {"n": 6, "sensitivity_at_5pct_FAR": 1.0, "control_FAR": null}, "frozen": {"n": 6, "sensitivity_at_5pct_FAR": 0.667, "control_FAR": null}, "lowbitrate": {"n": 6, "sensitivity_at_5pct_FAR": 0.333, "control_FAR": null}, "noise": {"n": 6, "sensitivity_at_5pct_FAR": 0.333, "control_FAR": null}, "overexpose": {"n": 6, "sensitivity_at_5pct_FAR": 0.333, "control_FAR": null}, "shake": {"n": 6, "sensitivity_at_5pct_FAR": 0.5, "control_FAR": null}, "static": {"n": 6, "sensitivity_at_5pct_FAR": 1.0, "control_FAR": null}}}`

**B-T0** `{"status": "NOT RUN"}`

**B-T1** `{"status": "NOT RUN"}`

**B** `{"status": "NOT RUN"}`

**C** `{"status": "NOT RUN"}`

**D** `{"status": "NOT RUN"}`

**A-full** `{"status": "RUN", "threshold_5pct_FAR_on_gold_good": 0.5478, "note": "threshold t = 5th pct of gold-good scores; item bad iff score < t. Says NOTHING about activity judgement.", "per_degradation": {"black": {"n": 6, "sensitivity_at_5pct_FAR": 0.0, "control_FAR": null}, "blur": {"n": 6, "sensitivity_at_5pct_FAR": 0.0, "control_FAR": null}, "clean": {"n": 6, "sensitivity_at_5pct_FAR": null, "control_FAR": 0.167}, "darken": {"n": 6, "sensitivity_at_5pct_FAR": 1.0, "control_FAR": null}, "frozen": {"n": 6, "sensitivity_at_5pct_FAR": 0.0, "control_FAR": null}, "lowbitrate": {"n": 6, "sensitivity_at_5pct_FAR": 0.0, "control_FAR": null}, "noise": {"n": 6, "sensitivity_at_5pct_FAR": 0.667, "control_FAR": null}, "overexpose": {"n": 6, "sensitivity_at_5pct_FAR": 0.0, "control_FAR": null}, "shake": {"n": 6, "sensitivity_at_5pct_FAR": 0.667, "control_FAR": null}, "static": {"n": 6, "sensitivity_at_5pct_FAR": 0.0, "control_FAR": null}}}`

### Duplicates detail (held-out P/R/F1)

**A** `{"status": "RUN", "method": "pHash", "threshold_maxF1_half1": 0.8125, "heldout_overall": {"n": 64, "P": 0.941, "R": 0.8, "F1": 0.865}, "per_kind": {"bright": {"n": 3, "recall_detect_rate": 1.0}, "crop5": {"n": 5, "recall_detect_rate": 1.0}, "easy_negative": {"n": 25, "false_positive_rate": 0.0, "specificity": 1.0}, "hard_negative": {"n": 19, "false_positive_rate": 0.053, "specificity": 0.947}, "offset1": {"n": 5, "recall_detect_rate": 0.4}, "reencode": {"n": 4, "recall_detect_rate": 1.0}, "speed11": {"n": 3, "recall_detect_rate": 0.667}}, "protocol": "threshold on held-out half 1, metrics on half 2"}`

**E** `{"status": "RUN", "method": "eg2_cosine", "threshold_maxF1_half1": 0.9894, "heldout_overall": {"n": 64, "P": 0.833, "R": 1.0, "F1": 0.909}, "per_kind": {"bright": {"n": 3, "recall_detect_rate": 1.0}, "crop5": {"n": 5, "recall_detect_rate": 1.0}, "easy_negative": {"n": 25, "false_positive_rate": 0.0, "specificity": 1.0}, "hard_negative": {"n": 19, "false_positive_rate": 0.211, "specificity": 0.789}, "offset1": {"n": 5, "recall_detect_rate": 1.0}, "reencode": {"n": 4, "recall_detect_rate": 1.0}, "speed11": {"n": 3, "recall_detect_rate": 1.0}}, "protocol": "threshold on held-out half 1, metrics on half 2"}`

**B** `{"status": "NOT RUN"}`

**C** `{"status": "NOT RUN"}`

**D** `{"status": "NOT RUN"}`

## Cost & Pi-class feasibility

_Single-core wall time normalised by a CPU benchmark (matmul-512 reps/s; see decode_floor.json). NOT measured on a Pi — numbers scale by benchmark ratio,/clock, and memory bandwidth._

| arm | s/segment (1 core) | RTF | cores for 10kh/2h | /3h | edge 50x | live 1x |
|---|---|---|---|---|---|---|
| A | 1.121 | 8.92 | 561 | 374 | False | True | RUN |
| E | 1.629 (GPU) | 6.14 | 815 | 543 | False | True | RUN |
| B | None | None | None | None | None | None | NOT RUN |
| C | None | None | None | None | None | None | NOT RUN |
| D | None | None | None | None | None | None | NOT RUN |
| A-full | 0.833 | 12.0 | 417 | 278 | False | True | RUN |

_Decode share: full cv2 decode ≈ 0.10666666666666667 s/seg (single-core; RTF excl decode in summary.json per arm)._

### Accuracy-vs-cost Pareto (gold AUROC vs cores for 10kh/2h)
- A-full: AUROC 0.675, 417 cores (RTF 12.0)
- A: AUROC 0.457, 561 cores (RTF 8.92)
- E: AUROC 0.596, 815 cores (RTF 6.14)
_Pareto read: cheapest (fewest cores) first; an arm is dominated if another is both cheaper and more accurate._

EG2: 2209 seg/h/GPU → 1629.7 GPU-h (0.19 GPU-years) for 10,000 h = 3.6M segments. VRAM peak 1548.5 MB.

## Streaming


**A** `{"status": "RUN", "operating_threshold": 0.5695, "per_chunk_latency_s_mean": 1.84, "realtime_capable": true, "ttd_per_degradation": {"black": {"n": 6, "median_ttd_s": null, "never_decided": 6}, "blur": {"n": 6, "median_ttd_s": null, "never_decided": 6}, "darken": {"n": 6, "median_ttd_s": 2.0, "never_decided": 0}, "frozen": {"n": 6, "median_ttd_s": 2.0, "never_decided": 5}, "lowbitrate": {"n": 6, "median_ttd_s": 2.0, "never_decided": 1}, "noise": {"n": 6, "median_ttd_s": 2.0, "never_decided": 0}, "overexpose": {"n": 6, "median_ttd_s": null, "never_decided": 6}, "shake": {"n": 6, "median_ttd_s": 2.0, "never_decided": 0}, "static": {"n": 6, "median_ttd_s": 2.0, "never_decided": 0}}, "note": "prefix-recompute latency is an UPPER bound; true incremental is cheaper"}`

**E** `{"status": "RUN", "operating_threshold": 0.976, "per_chunk_latency_s_mean": 0.144, "realtime_capable": true, "ttd_per_degradation": {"black": {"n": 6, "median_ttd_s": 2.0, "never_decided": 0}, "blur": {"n": 6, "median_ttd_s": 2.0, "never_decided": 0}, "darken": {"n": 6, "median_ttd_s": 2.0, "never_decided": 0}, "frozen": {"n": 6, "median_ttd_s": 2.0, "never_decided": 2}, "lowbitrate": {"n": 6, "median_ttd_s": 2.0, "never_decided": 1}, "noise": {"n": 6, "median_ttd_s": 2.0, "never_decided": 1}, "overexpose": {"n": 6, "median_ttd_s": 2.0, "never_decided": 1}, "shake": {"n": 6, "median_ttd_s": 2.0, "never_decided": 2}, "static": {"n": 6, "median_ttd_s": 2.0, "never_decided": 0}}, "note": "prefix-recompute latency is an UPPER bound; true incremental is cheaper"}`

**B-T0** `{"status": "NOT STREAMABLE", "reason": "no chunk scores exposed"}`

**B-T1** `{"status": "NOT STREAMABLE", "reason": "no chunk scores exposed"}`

**B** `{"status": "NOT RUN"}`

**C** `{"status": "NOT RUN"}`

**D** `{"status": "NOT RUN"}`

**A-full** `{"status": "NOT RUN", "reason": "chunk features only for A/E base variants"}`

## Hyperspectral verdict

`{"status": "NOT RUN"}`

## VLM verdict

`{"status": "NOT RUN"}`

## Recommendation for Jadal

_TBD once arms B and D report; current partial read below._

Partial (arms B-final/D pending): A: AUROC 0.457 CI [0.375, 0.521]; E: AUROC 0.596 CI [0.505, 0.817]; B-T0: AUROC 0.43 CI [0.359, 0.683] (PROVISIONAL filename); B-T1: AUROC 0.481 CI [0.41, 0.667] (PROVISIONAL filename); A-full: AUROC 0.675 CI [0.6, 0.8]. CAVEAT on B-T0, B-T1: as-is scores {'B-T0': 0.637, 'B-T1': 0.678} were NOT fitted under LOCO and may be tuned in-sample (optimistic); the LOCO-clean head on the same features scores lower — treat the head number as the honest one until arm B publishes its final protocol. Arm A at chance (CI spans 0.5) means the 16-frame MediaPipe baseline carries no usable-activity signal on current labels; A-full (whole-video cheap signals) leads but its synth profile (catches darken only) shows it keys on activity/motion, not capture quality.

## Limitations

- Only 3 IndEgo clips labelled so far; 14 bad labels → CIs are wide; any CI spanning 0.5 = no evidence.

- Synthetic degradations measure capture-quality sensitivity only, NOT activity judgement.

- Cost numbers are workstation single-core, not a Pi; scaled by benchmark.

- Human labels are IndEgo workers; Awign footage differs.

