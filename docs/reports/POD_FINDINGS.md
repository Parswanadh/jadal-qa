# POD_FINDINGS — consolidated findings (jadal-qa pod, NVIDIA A40)

Scope: everything measured on this pod from the ported laptop code, the opencode swarm (Lead A),
the DeepSeek Harness (Lead B) and the monitor research agents. All OOD numbers use
**factory-grouped** bootstrap CIs; in-domain uses **leave-one-clip-out (LOCO)**.

## 1. Executive summary

| metric | value |
|---|---|
| In-domain (IndEgo 150, LOCO) | frozen two-view **0.853** · EG2 0.848 · pixel-26 0.810 · **deterministic rules 0.822** |
| OOD pooled (160 labels) | frozen **0.566** [0.445, 0.674] · adapted transferable **0.781** [0.634, 0.895] (+0.21 significant) |
| **Confirmatory hold-out (108, one-shot)** | adapted **0.676** [0.525, 0.827] · frozen **0.690** [0.534, 0.823] · sharpness-3 0.715 → **the pooled 0.781 does NOT replicate** |
| Review ↔ error (OOD) | adapted quantile **60% review @ BA 0.857**, or **44% @ 0.793**; frozen ref 44.2% @ 0.708 |
| Throughput (optimized, uncontended) | **29.6 footage-h/wall-hour** (every-2nd) → 27.66 h corpus in **~56 min** |
| Determinism | **180/180 byte-identical receipts ×10 runs**; rule decision **2 ms (0.06%)** |
| **95% accuracy** | **not reachable** with the current signals at any realistic label budget |

## 2. Data

- Corpus: **27.66 h / 391 clips / 85 factories / 9,757 windows** (Egocentric-10K, 1080p30 HEVC; tars pruned after verify).
- Labels: **IndEgo gold 150** (100 good/50 bad, LOCO) · **OOD 160** (138/22, 47 factories) · **round-3 hold-out 108** (92/16, 44 factories, scored once, never tuned on).
- Third-party footage: do not publish.

## 3. Model families

| family | in-domain | OOD pooled | OOD confirmatory | deterministic | explainable |
|---|---|---|---|---|---|
| Deterministic physics rules | **0.822** (beats logistic 0.802) | 0.475–0.493 (fail; 99.55% reject on the corpus) | — | **100%** | per-rule evidence |
| Adapted transferable + per-domain z-score | 0.736 | **0.781** | 0.676 | given the batch | features + fixed signs |
| Probabilistic frozen ensemble | **0.853** | 0.566 | 0.690 | no (GPU numerics) | logistic coefficients |

## 4. Accuracy

- In-domain best: two-view 0.853 (rank-avg 0.856); nested-C 0.834; deterministic rules 0.822; sharpness-3 head 0.578.
- OOD (160): frozen 0.566; EG2 view 0.589; pixel-26 0.520; adapted stable-8+Z3-6 0.781; Z3-6 0.759; sharpness-3 zero-shot 0.708.
- **Confirmatory (108):** adapted 0.676 vs frozen 0.690 — a **tie** (paired Δ −0.009 [−0.197, 0.177]); the earlier +0.21 gain was partly selection on the 160.
- The adapted score is near its ceiling for these features (no fitted model beats the label-free equal-weight mean; features are not domain-invariant, IndEgo-vs-OOD classifier AUROC 0.999).

## 5. The 95% question

- **A3:** label curve plateaus at error ~0.28–0.33 (saturating e∞=0.326); **90% and 95% are unreachable** at any realistic label budget; labels are not the bottleneck — the **features** are (19/22 OOD bad = NO_ACTIVE_WORK semantic).
- **A4:** ≥95% decided-balanced-accuracy is achievable **only at low coverage**: ~**3% in-domain** (honest CI-LB), **~15% on the hold-out**, no confident OOD-160 claim. Deterministic rule form e.g. accept top-23 / reject bottom-6 / review 79 (108).
- Per-factory specialization **hurts** at 1–4 labels/factory (global BA 0.733 → 0.52–0.57); needs ≫4 labels/factory.

## 6. Review ↔ error frontier (OOD)

| method | REVIEW % | OOD BA [grouped CI] |
|---|---|---|
| frozen rank-avg band w=0.16 | 44.2 | 0.708 [0.521, 0.832] |
| adapted asymmetric accept 25% / reject 15% | 60.0 | **0.857 [0.746, 0.939]** |
| adapted global quantile q=0.28 | 44.0 | 0.793 [0.686, 0.891] |
| adapted q=0.30 / q=0.35 | 40.0 / 30.0 | 0.751 / 0.732 |

Joint improvement is achievable via **prior-free rank/quantile thresholds** (calibration cannot rescue the 0.5-band; 0/22 bad cross 0.5). Caveats: not significant at n_bad=22; test-set-selected; validated only by the confirmatory hold-out.

## 7. Throughput / scaling (measured)

| configuration | rate | 27.66 h corpus |
|---|---|---|
| 1 worker | ~1.5 footage-h/wall-hour | ~18 h |
| 7 workers, stride 2, uncontended (5-video run: 196 windows / 132 s) | **29.6 h/h** | **~56 min** |
| 7 workers, stride 1, uncontended (L3) | 19.8 h/h | ~1.4 h |
| 7 workers, stride 1, contended (R1 live) | 8–11 h/h | 2.5–3.5 h |
| N pods | N × ~30 h/h (stride 2) | 5 pods ≈ 11 min |

- Decode is **76–82%** of per-window cost; the single **NVDEC engine** caps in-pod scaling (ceiling at K=4); VRAM is never the limit (17/46 GB); EG2 batch size is flat.
- Sharding is **exact** (K=1/2/4/7 → identical combined digest); scale-out is by **pods**.

## 8. Representation / cost

- fps: pixel-26 holds within 0.02 down to **3.33 fps**; EG2 is neutral-to-better at **1 fps** (0.860) and 0.5 fps (0.870); the apparent OOD "gain" at low fps is a **sampling-rate artifact**.
- Resolution/compression: **360p @ CRF28 ≈ 40% bytes, ΔAUROC<0.02**; 120–180p no cliff; OOD neutral.
- Mosaic: dropped as a model input (≤ mean-of-frames at every density); small tiles (112–160 px) cheaper; **timestamps +0.021**; kept as evidence only (4.3% of baseline time).
- Sampling: **uniform every-2** best; section-skipping anywhere does not help; 4 s early exit; SPRT does not beat every-2.
- Dropping the EG2 view costs ~−0.04 in-domain / −0.05 OOD, saves 7% (optimized) time + 1.7 GB VRAM.

## 9. Determinism

- Rule path: canonical JSON receipts, Q=4 integer comparisons, no floats/clocks/GPU → **180/180 windows byte-identical across 10 runs** (also across processes/hash seeds). Decision 2 ms.
- Recalibrated heads for the optimized sparse pipeline restore the frozen operating point at no in-domain cost (0.840 vs 0.834 nested).

## 10. Labels & certification

- Strategy: **diversity/maximin > breadth > random ≫ depth**; **uncertainty is the worst** selector.
- Certification (95% confidence): **85 factories → error ≤ 0.20** (38–86 factories); 0.15 (67–152); **0.05 needs 600–1,365 factories** — the corpus buys 0.20, not 0.05.
- Plan: 85×4 = **340 labels** → certify ≤0.20; 680 → ≤0.15; 0.90/0.95 needs **new signals**.

## 11. PR / merge hygiene

- All experiment PRs were brought onto current `main` with **own-file-only diffs** and bases retargeted to `main` (~44 merge-ready); `configs/ensemble_v1.json` is byte-identical to main everywhere (differences were path remaps only, coefficients unchanged).
- No force-pushes, no self-merges; env vars used for pod paths.

## 12. Recommendations & open items

1. **Ship** the deterministic physics model **in-domain** (explainable, 0.822, 2 ms, byte-identical receipts); **do not** deploy it OOD without re-quantiling.
2. **OOD:** the adapted score is a *tie* with frozen on the confirmatory hold-out — prefer the frozen ensemble unless a new signal changes the picture.
3. **To exceed ~0.85 in-domain / ~0.70 OOD: new per-pixel signals** (more features, motion/hand-visibility) — not labels (in flight: F30/F40/F50, A1/A2).
4. **Review:** use prior-free quantile rules for the OOD operating point (60% review @ 0.857); the 12% conformal point is in-domain only.
5. **Throughput:** ~30 h/h/pod (stride 2); scale by pods; zero-copy is the in-pod lever (decode-bound).
6. Open: A1/A2 (capacity/signals), F30/F40/F50 (feature expansion), zero-copy (L4), R1 full-corpus run.
