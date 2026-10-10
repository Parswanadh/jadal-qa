# Z1.160 — Z1 OOD error reduction re-run on 160 OOD labels

**Why:** PR #121 used the old 60 OOD labels (n_bad=9); re-run on the merged 160 labels with honest factory-grouped CIs.

**Hypothesis:** The sharpness-3 zero-shot head (0.706 at n=60) survives at n_bad=22; few-shot curve is lower.

**Data / label source:** IndEgo manifest_gold n=150 + labels/ood160_labels.json n=160 (138 good/22 bad, 47 factories)

**Data regime:** OOD evaluation only; 10 fps-effective view

**Method:** logistic on IndEgo (LOCO) applied zero-shot; unsupervised alignment; per-clip norm; factory-grouped few-shot

**Owner:** M5  **Started:** 2026-10-10 19:25

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:25): adopt.** Sharpness-3 zero-shot head confirmed: OOD-160 AUROC 0.708 [0.588,0.823] vs frozen 0.566; few-shot curve lower than at n=60 (k=48 -> 0.664); alignment/per-clip/self-training still do not help.

Results file: eval/results/exp_domain_adapt160/Z1.01_160.json

Decision: n/a

Cost: n/a
