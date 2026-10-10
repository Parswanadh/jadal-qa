# O4.01 — Confirm Z3 transferable-feature adaptation on the 160 OOD labels

**Why:** Z3.03 reported OOD AUROC 0.84-0.85 on 60 windows (n_bad=9); confirm on the full 160 human labels (n_bad=22)

**Hypothesis:** The per-domain z-score adapted score stays > frozen 0.566 with a factory-grouped CI excluding 0.5 at n_bad=22

**Data / label source:** labels/ood160_labels.json (138 good/22 bad, 47 factories); Z3 candidates (q1 reuse) + q2 recompute; caches/gpu/features.json + ood160_pixel.npz

**Data regime:** OOD Egocentric-10K, human-labelled, evaluation only

**Method:** IndEgo-signed per-domain z-score equal-weight mean; factory-grouped bootstrap CI

**Owner:** O4  **Started:** 2026-10-10 17:41

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (pre-review note, 2026-10-10 17:43): adopt.** This line was written two minutes
after the run started (17:41) and is a **pre-review summary, not a reviewed result**; the
reviewed numbers live in `docs/reports/exp_adapt160.md` and `O4.01.json`.

Headline is the **pooled 160 (q1 contaminated)**: Z3 transferable-6 per-domain z-score
OOD-160 AUROC 0.759 [0.597,0.886] grouped, BA 0.709; stable-8+6 0.781 [0.634,0.895], BA
0.733; stable-8 minus `global_shift` + 6 0.788 [0.648,0.898], BA 0.729; frozen 0.566.
Paired grouped gain +0.19/+0.21/+0.22 (≥99.4% >0). q1 informed feature selection, so it is
not a hold-out; **q2 alone (0.70/0.73/0.74) is not separable from chance** (q2-only grouped
CI includes 0.5 for (a)). The per-domain z-score is transductive (pooled mean/std) and the
threshold 0 is the pooled mean, so the BA is descriptive, not deployable without a
batch-normalisation rule. Exact recipe frozen in `eval/exp_adapt160/o4_recipe.json`
(sha256 `449c27e0…`).

Results file: eval/results/exp_adapt160/O4.01.json

Decision: confirm (ranking only; operating point unproven)

Cost: CPU only, 100 q2 decodes (~75s)
