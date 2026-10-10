# exp_review — Y2 review-gate (decision-theoretic gating + stage-0)

Read [`docs/reports/exp_review_gate.md`](../../reports/exp_review_gate.md) first.

**Goal:** cut the PoC REVIEW share on the 1,546 Egocentric-10K windows below
60% (frozen policy: 75.5%) without decoding anything new.

**Result:** gating at coverage 45% (cost 1:1) → **55.0%** review; + stage-0 →
**54.0%**; temperature is a no-op on the share and hurts gold BA, so it is off.
In-domain gold-LOCO BA 0.837 [0.645, 0.937] (n=150). OOD-60 near chance
(BA 0.496, AUROC 0.415, n_bad=9) — evaluation only.

| id | result |
|---|---|
| Y2.01 | `eval/results/exp_review/Y2.01.json` — gating sweep (gold/OOD/1,546) |
| Y2.02 | `eval/results/exp_review/Y2.02.json` — stage-0 + combined pipeline |

**Figure:** `images/review_gate.png` / `images/review_gate.svg`

**Reproduce:**

```bash
OMP_NUM_THREADS=2 python3 eval/exp_review/run_y2.py
OMP_NUM_THREADS=2 python3 eval/exp_review/make_fig.py
python3 -m pytest tests/test_exp_review.py -q
```

**Honesty:** the 1,546 windows have no labels, so all their numbers are
descriptive. Nothing is fit on the OOD-60.
