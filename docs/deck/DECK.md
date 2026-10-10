---
marp: true
title: Jadal — deterministic QA for head-camera factory video
paginate: true
---

<!--
Deck source of truth. Render with Marp (`marp docs/deck/DECK.md -o deck.html`) or
read as plain Markdown (each `---` starts a slide). Every number carries its source:
FACTS = docs/FACTS.md (verbatim snapshot of jadal_pod_bundle/docs/FACTS.md,
sha256 77c7331b5e9ca8e8…), REG = experiments/registry.jsonl (rendered to docs/EXPERIMENTS.md).
Honesty rule: OOD failure is slide 3, not an appendix. PENDING items are never results.
-->

# Jadal

## A deterministic QA funnel for crowd-captured head-camera video

Awign “Physical AI” hackathon · Bengaluru · 10–11 Oct 2026
Status documented 2026-10-10 · sources: `docs/FACTS.md`, `experiments/registry.jsonl`

---

# The problem

Crowd workers record factory work with head cameras. Most footage is unusable:
no active work, hands out of frame, too dark, duplicated, frozen.

**We judge every 10-second window:** ACCEPT / REJECT (with evidence) / REVIEW
+ a content-hashed receipt. Deterministic rules, no generative model in the
decision path. Humans only audit samples.

<!-- Source: FACTS “Direction”. -->

---

# ⚠️ Read this before any other number: OOD failure

On **60 real Egocentric-10K windows**, human labels (**51 good / 9 bad**), the
shipped IndEgo-calibrated cascade is **at chance**:

| metric | value |
|---|---|
| score AUROC | **0.529** (95% CI 0.32–0.74, optimistic) |
| good windows falsely rejected | **21 / 51 = 41%** |
| bad windows rejected | 5 / 9 |
| balanced accuracy | **0.51** |

→ The dashboard’s ~40% reject share on Egocentric-10K is **mostly false rejects**
(domain shift 480p/10 fps IndEgo → 1080p/30 fps HEVC). **Not a quality measure.**

<sub>Source: FACTS “UPDATE 18:40 IST — OUT-OF-DOMAIN RESULT”; REG `E6.03`.</sub>

<sub>**Verification note (LB1, `b/verify`, + independent verifier PR #100):** **two different scorers,
both correct.** **0.529** = **PoC cascade as shipped** (`eval/results/poc/ood_check_60.json`):
0.529 [0.316, 0.728], 21/51 good rejected, 5/9 bad, bal-acc 0.513 — reproduces exactly.
**0.415** [0.207, 0.630] = **frozen two-view ensemble** (`caches/ensemble/ood_scores.json`,
`usable_prob`): 26/51 good rejected, 3/9 bad. Both at chance with heavy false rejects —
OOD failure confirmed for both. Do not merge the two figures.</sub>

<!-- Source: FACTS “UPDATE 18:40 IST — OUT-OF-DOMAIN RESULT”. Corroborated by REG E6.03 (frozen motion cue OOD AUROC 0.281 [0.145,0.434]). -->

---

# What we can defend: in-domain only

150 ten-second **IndEgo** windows (7 clips, 480p ~10 fps), **one human labeller**,
100 usable / 50 bad. Leave-one-clip-out, clip-bootstrap 95% CI:

| model | AUROC | 95% CI |
|---|---|---|
| 26 per-pixel features + logistic | 0.810 | [0.69, 0.89] |
| EmbeddingGemma-2 window view | 0.848 | [0.73, 0.91] |
| probability average | 0.853 | [0.73, 0.92] |

Selective accuracy — **pixel26 view, fixed C=0.3, exploratory (not nested)**: 0.74 deciding all 150 →
0.84 on the 80% most confident (nested pixel26 curve: 0.76 / 0.81 / 0.80). The **probability-average
view** gives 0.79 @0.5 and 0.89 at 60% coverage (7 of 63 good wrongly rejected); EG2 gives 0.77 / 0.87.
**~0.8 AUROC is the honest ceiling so far; accuracy is not yet high.**

<sub>Source: FACTS “UPDATE 18:25 IST”; REG `E1.01`, `E1.02`.</sub>

<!-- Source: FACTS “UPDATE 18:25 IST”. REG cross-check: E1.01 (0.809), E1.02 (0.848). -->

---

# Nested validation: gates NOT met

C chosen by inner leave-one-clip-out (n=150):

| view | AUROC |
|---|---|
| EG2 | 0.846 |
| pixel (26 features) | 0.792 |
| probability average | 0.834 |
| rank average | 0.844 |

**Gates AUROC ≥ 0.85 and CI-low ≥ 0.72: NOT met.** Weakest clip 0.606.
The frozen ensemble (sha256 `4da1068a78d5d150…`) was pre-registered before the
OOD labels were read. Its OOD score is **measured: 0.415 [0.207, 0.630]** on the same 60 windows
(26/51 good rejected, 3/9 bad) — a *different scorer* from the shipped PoC cascade (0.529); both at
chance. Only the LB4 / PR #97 **re-derivation** on the pending second label batch is in progress.

<sub>Source: FACTS “UPDATE 18:40 IST — corrections”.</sub>

<!-- Source: FACTS “UPDATE 18:40 IST — corrections”. -->

---

# What is open or not done (never shown as a result)

- Raspberry Pi measurement — **not done and not planned for this submission** (not measured, not claimed; x86 laptop cores only as proxy)
- Egocentric-10K accuracy — **in progress** (no human labels on the 1,546 windows)
- Hyperspectral — **not done and not planned for this submission** (denied: cameras are RGB; we do RGB temporal-spectral analysis, which is not hyperspectral)
- Freshness nonce-LED — **in progress** (Pico shows verdict LEDs only)
- Voice — **in progress** (not measured, not claimed)
- End-to-end GPU number — **in progress** (kernel-only limits measured)
- Second OOD batch `ood_labels2.json` — **not delivered** → OOD stays n=60 / n_bad=9

<!-- Sources: FACTS “Direction”, “Data and ground truth”, “Still NOT done/claimed”; REG E5.04 (labels_missing=true, no accuracy computed). -->

---

# Pipeline

| stage | scope | action |
|---|---|---|
| 0 | first 1–2 s, tiny res | exact rules: dark / blocked / over-exposed / frozen |
| 1 | whole video, every frame (reduced res) | spatial + temporal spectrum, noise model, motion after ego-motion removal → rule list + score |
| 2 | only uncertain windows | optional heavy tools (MediaPipe hands, EG2) |
| receipt | every window | SHA-256 of canonical JSON (not signed); humans audit samples |

Quality features on RAW pixels; activity features on CLEANED video; each cleaning
step must earn its place in cross-validation.

<sub>Source: FACTS “Pipeline as designed now”.</sub>

<!-- Source: FACTS “Pipeline as designed now”. -->

---

# Evidence, not vibes

- Every reject carries **evidence** (value, comparison, threshold).
- Receipts are **content-hashed**, re-verify = re-run the pinned pipeline on the
  original clip and compare; a tampered receipt fails. **Not signed.**
- Determinism: CPU feature path gives an identical quantised digest over 10 runs;
  GPU numerics not claimed identical across devices; 10 identical end-to-end
  receipt hashes **in progress**.
- Certified error bounds **not done and not planned for this submission**: not achievable at n=150 (correction
  needs ~86 clean fires; certification split has 84; windows within a clip are
  correlated).

<!-- Source: FACTS “Measured results so far”, “Determinism”, “Funnel calibration”. -->

---

# Cost and hardware (measured, laptop not Pi)

- 480p decode ≈ 90–100× real time on one core; keyframe-only ≈ 200×.
- Real 1080p/30 fps HEVC: one CPU core 43–55 fps (RTF 1.4–1.8); NVDEC 620–730 fps
  (RTF 21–24), saturates at one engine.
- GPU kernels alone ≈ **700 video-hours per hour** at 4.6 GiB *(supersedes an
  earlier 1,800–2,000 h/h figure)*. End-to-end real throughput is lower
  (decode-bound); the end-to-end GPU number is **in progress**.
- PoC run: all **1,546 windows** (58 clips, 4.404 h, 47 factories) processed
  with descriptive verdicts; ~7× real time so far.
- Adopted IndEgo policy: every 2nd window + 4 s exit ≈ 0.20 of footage (≈ ×5).

<!-- Sources: FACTS “Measured results so far”, “UPDATE 17:40”, “UPDATE 17:50”; REG E3.01–E3.05, E1.04. -->

---

# Negative results (kept on purpose)

- **“Spatial localisation of motion helps” — WITHDRAWN.** Space-pooled grid
  channels 0.790 > spatial-grid ensemble 0.753 > global-only 0.724.
- **“Hand-region motion is the work signal” — NOT SUPPORTED.** Residual-motion
  magnitude is inverted on real windows; part of the cue may be head stability.
- **Custom rhythm network — NEGATIVE.** 0.753 (0.65–0.81) vs 0.810 for the
  26-feature set; 0.351 after training with synthetic negatives.
- **Generic VLM — STOPPED by the builder.** SmolVLM2-2.2B: AUROC 0.57/0.77 at
  n=79 (14 bad); ~58 days of GPU for 3.6 M segments.
- **Conformal calibration does not hold on held-out clips.** Nominal 2.7%
  false-reject → 46% (effective n ≈ 11).
- **Frame selection dropped** (uniform wins at equal budget, REG E1.04);
  **re-encoding to the IndEgo spec does not fix transfer** (REG E4.03).

- **Duplicate detection on real labels (tested):** recall@1 = recall@3 = 0.5 on the 6 real DUPLICATE
  windows (precision_top 0.0476, 63/150 flagged) — constructed-pair F1s (0.947/0.909/0.865) are a
  different, easier setting and their 64-pair manifest is not on this pod.

<sub>Sources: FACTS “UPDATE 18:40”, “UPDATE 18:25”, “Direction”, “UPDATE 17:50”; REG `E1.04`, `E4.03`.</sub>

<!-- Sources: FACTS “UPDATE 18:40”, “UPDATE 18:25”, “Direction”, “UPDATE 17:50”; REG E1.04, E4.03. -->

---

# Where this leaves us

**Honest position:** a working deterministic funnel with measured in-domain
AUROC ≈ 0.8 (not high, gates not met) and a documented **out-of-domain failure**
at chance on real factory footage.

**Next (in progress, in order):** frozen ensemble OOD score → domain adaptation
(REG E5.02: ~12 OOD windows ≈ 0.7 exploratory) → OOD policy decision → Pi
measurement → end-to-end GPU number.

**Never claimed:** transfer, certified bounds, Raspberry Pi, hyperspectral,
signing, voice, Egocentric-10K accuracy.

<!-- Sources: FACTS; REG E5.02, E5.04. -->

---

# Sources

- `docs/FACTS.md` — verbatim snapshot of the pod fact sheet
  (sha256 `77c7331b5e9ca8e8…`); source of every headline number.
- `experiments/registry.jsonl` — union append-only registry, 36 entries
  (“latest line per id wins”; 28 done / 7 running / 1 pending); rendered to `docs/EXPERIMENTS.md`.
- Independent verification: LEAD B claim `LB1` (branch `b/verify`,
  `docs/reports/pod_verification.md`) — **20/20 published quantitative claims reproduced
  exactly** on an A40 pod, CPU-only.
- `docs/explainer.html` — the same content as a sources-annotated page.
- `docs/SUBMISSION.md` — submission pack index.

**No number appears in this deck that is not in one of those two source files.**
No generative model is in the decision path; this deck is documentation.
