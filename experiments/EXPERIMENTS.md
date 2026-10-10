# Experiment log (Q2)

## Q2.01 — Deterministic physics rule-list transfer on 160 OOD human labels
- **Why:** The deterministic rule list (L1.23) was degenerate in-domain; test
  whether a frozen, thresholded physics rule list transfers out-of-domain on the
  160 human Egocentric-10K labels, and which rules survive the shift.
- **Hypothesis:** Quality tripwires (blur, entropy, sharpness) transfer;
  activity/semantic rules may not.
- **Data / label source:** 160 Egocentric-10K human windows (138/22, 47
  factories; `labels/ood160_labels.json`); physics+theory per-window features
  (frozen 160×90 recipe); frozen IndEgo rule table (`configs/perpixel_v1.json` +
  `rulelist.json`, B/verify-rules).
- **Regime:** IndEgo 480p10fps → OOD 1080p30 HEVC downscaled to 160×90.
- **Method:** apply frozen IndEgo thresholds (Q=4, no OOD fitting); OR rule list;
  AUROC + balanced accuracy with factory-grouped bootstrap (n_boot=2000);
  per-rule precision/recall; review-band frontier.
- **Owner / started / finished:** Q2 / 2026-10-10 18:54 / 2026-10-10 18:54
- **Verdict:** **drop** — Frozen rule list does NOT transfer: R3 AUROC 0.475
  [0.327,0.630] grouped, BA 0.441 [0.325,0.581], rejects 60/160, false-accept
  16/22, false-reject 54/138; R5 AUROC 0.493, BA 0.456. Below chance and worse
  than the frozen logistic baseline 0.566/0.546. Review band monotonically worse
  (0.456 → 0.049 at 84 % review). `lowent`/`frozen`/`ffroz`/`still` abstain;
  `blur`/`ten`/`dark`/`lowsnr`/`overexp`/`pbytes`/`clipfrac` misfire (precision
  0.03–0.18); only `madfroz` >chance (0.333, recall 0.09).
- **Decision:** Do not ship the deterministic physics rule list as an OOD gate,
  not even as a review tripwire. Keep the abstaining rules as no-ops; gate with
  the adapted per-pixel model (O2/O4). Re-derive absolute thresholds per regime.
- **Cost:** CPU; 160 windows × physics+theory (170 s, 2 workers); no GPU.
- **Results:** `eval/results/exp_phys/Q2.01.json`
