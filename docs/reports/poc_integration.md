# PoC integration — adopted sampling + early-exit policy (claim LB2)

**Status: DESCRIPTIVE ONLY.** This report is *not* an accuracy claim and makes
**no out-of-domain (OOD) claim**. The E3.05 selection itself says
"working policy on IndEgo, not validated" (7 clips, sparse gold windows, no
held-out clips, no OOD validation). Negative and thin evidence is kept below.

Branch `b/poc-integrate` · owner `lead-b` · registry id **LB2** · device: **CPU only**
(GPU unavailable in this container; every number below is computed from cached
features — no model was trained on the GPU here).

## What changed

The PoC scorer path now reads one adopted policy from
`jadal/poc_stream/policy.py::ADOPTED` and applies it deterministically:

| Setting | Value | Evidence |
|---|---|---|
| Window sampling grid | every 2nd window (`range(0, n, 2)` per clip) | E3.01 / E3.05 `adopted_policy="every-2"` |
| Early exit inside a window | 4.0 s observed | E3.03 (EG2 prefixes: 2 s 0.804, **4 s 0.860**, 6 s 0.869, 8 s 0.857, full 0.848) |
| EmbeddingGemma-2 view | 2 fps, first 8 frames at exit | E1.02 (2 fps AUROC 0.848 [0.73, 0.91]) |
| Per-pixel view | 26 features + logistic, `prefix[1]` = 4 s | E1.01 (10 fps 0.809 [0.69, 0.89]) |
| Fusion / thresholds | equal-weight probability average; accept > 0.5, reject < 0.4, else uncertain | frozen `caches/ensemble/ensemble_v1.json` |
| Clip verdict | sampled bad fraction ≥ 0.30 → reject | E3.01 / E3.05 |

`eval/poc_integrate/run_integration.py` re-fits both logistic views
**leave-one-clip-out (LOCO)** at the observed 4 s prefix (each clip held out
entirely; frozen ensemble_v1 hyperparameters `C_eg2=0.003`, `C_px=0.3`), fuses
them, and writes per-window verdicts. 95% CIs are clip-bootstrap (2 000
resamples, seed 7). Regenerate:

```
OMP_NUM_THREADS=2 python eval/poc_integrate/run_integration.py
```

Outputs:
- `eval/results/poc_integrated/window_verdicts.jsonl` — 150 IndEgo gold windows + 60 OOD-labelled windows (OOD descriptive only)
- `eval/results/poc_integrated/summary.json` — top-level `descriptive_only`, `n`, `label_source`, `labels_sha256`, `protocol`, metrics, counts, cost
- `tests/test_poc_integration.py` — 12 tests (policy units + artifact schema)

## LOCO numbers (n = 150 human-labelled IndEgo windows, clip-bootstrap 95% CI)

| View | 4 s prefix (adopted) | Full window (10 s) |
|---|---|---|
| 26-feature pixel logistic | **0.797** [0.658, 0.860] | 0.804 [0.685, 0.888] |
| EmbeddingGemma-2 2 fps | **0.860** [0.715, 0.944] | 0.842 [0.711, 0.913] |
| Fused (equal weight) | **0.874** [0.742, 0.942] | 0.849 [0.721, 0.929] |
| Fused balanced accuracy @ 0.5 | 0.740 | 0.765 |

The 4 s prefix is not worse than the full window on the EG2 and fused views
(CIs overlap heavily), consistent with E3.03 saturating at 4–6 s; the fused
4 s point estimate is higher but that difference is **not established** at
n = 150 / 7 clips.

## Counts

- Window verdicts over all 150 gold windows (4 s LOCO policy): **accept 108, reject 27, uncertain 15**.
- Every-2nd grid samples **77 / 150** windows (51.3%; E3.01 reported mean 51.5% of windows per clip).
- Clip verdicts from the sampled windows vs the label-oracle truth (bad fraction ≥ 0.30):
  **5/7 agree** — misses `desk` (sampled bad fraction 0.20 vs truth 0.45) and
  `pallet2` (0.00 vs 0.67, n_sampled = 3). E3.01's 7/7 is a *label-oracle*
  count; this is the model-based count and is worse, as expected.

## Cost ratio (E3.04 arithmetic, reported not re-measured)

- every-2nd window alone: **x1.94** (2.266 h per 4.404 h corpus).
- every-2nd + 4 s exit: **x4.86** (~0.907 h per 4.404 h, 0.2059 of footage).
- Source: E3.04 `with_early_exit`, decode-bound upper bound.

## OOD rows — descriptive only, no claim

The 60 human-labelled Egocentric-10K windows (51 good / 9 bad) are included in
`window_verdicts.jsonl` with their **frozen full-window** ensemble scores read
from `caches/ensemble/ood_scores.json`. The adopted 4 s prefix policy could
**not** be re-applied to them: the pod bundle has no cached per-pixel features
or EG2 embeddings for those windows (only the derived scores). Their verdict
counts (accept 17, reject 29, uncertain 14) are **descriptive only and are not
an accuracy claim**; a previous analysis on the same labels found the
IndEgo-calibrated scorer at chance OOD (AUROC 0.529, 0.32–0.74), i.e. the
rejects there are expected to be mostly false rejects from domain shift. Each
OOD row carries `"claim": "none - descriptive only, not an accuracy claim"`.

## Limitations (kept deliberately)

- 7 clips only; `pallet2` has 6 windows; clip-bootstrap CIs assume clips are exchangeable.
- Gold windows are a sparse sample of each clip — "coverage" here means all gold windows, not continuous video.
- Balanced accuracy at 0.5 and the clip-verdict agreement are descriptive; no certified error bounds exist at n = 150.
- The 4 s prefix for the pixel view is `prefix[1]` (40 decoded frames at 10 fps); E3.03 measured the 4 s curve on EG2 prefixes.
- No GPU, no live video decode in this run: everything is cached-feature evaluation.

See also `experiments/registry.jsonl` (id `LB2`) and `eval/results/poc_integrated/summary.json`.
