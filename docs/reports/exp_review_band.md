# Y1.01 — Review-share reduction: abstain-band + calibration sweep (target < 60%)

**Registry id:** Y1.01 · **Track:** Y1 (review) · **Owner:** Y1 · **Branch:** `m/review-band` ·
**Base:** `pod/base` @ `f20013b` · **Device:** none (CPU only; all scores cached) · **Host:** 96 cores, 503 GB RAM
**Results:** `eval/results/exp_review/Y1.01.json` (+ `Y1.01_support.json`) · **Figure:** `docs/experiments/exp_review/images/y1_01_review_vs_error.png`
**Command:**
```
cd /workspace/wt/m-review-band && OMP_NUM_THREADS=2 python3 eval/exp_review/run_y1.py
cd /workspace/wt/m-review-band && OMP_NUM_THREADS=2 python3 eval/exp_review/make_y1_artifacts.py
```

## 1. What this is (and is not)

The frozen PoC decides ACCEPT/REJECT only when the score is **outside** an abstain
band; inside the band → REVIEW. On the 1,546 Egocentric-10K windows of E8.01 the
frozen `factory_1080p30` operating point (`score ≥ 0.75` accept, `≤ 0.25` reject,
else review) yields **75.5 % review**. This experiment shrinks the symmetric band
(half-width `w`, rule `accept if score ≥ 0.5+w`, `reject if score ≤ 0.5−w`, else
review) and re-calibrates the score, to see how far review can be cut without
pretending the model got better.

> **The 1,546 windows have NO human labels** (`label_source = "none"`). Their review
> share is a **workload / descriptive** number, not accuracy. Accuracy is only
> measurable on (a) the 150-window IndEgo human gold (LOCO over 7 clips) and
> (b) the 60-window human-labelled Egocentric-10K sample (51 good / 9 bad), which is
> the **pre-registered OOD test — evaluation only, never fit on**. The two are never pooled.

Error is `1 − balanced accuracy` over **decided** windows only (windows the band
sends to REVIEW cost human time, not a model error). "good-rej" is good windows
decided REJECT (100 gold good; 51 OOD good).

## 2. Reproduction of the frozen three-way verdicts

Read from `/workspace/artifacts/windows_rescored.jsonl` (1 header + 1,546 window lines):
using each window's own profile thresholds, the verdict is reproduced on **1,546/1,546**
rows, giving accept **296 (19.15 %)**, reject **83 (5.37 %)**, review **1,167 (75.49 %)** —
exactly E8.01. A uniform symmetric band `w = 0.25` gives 75.42 % review (1-window
difference: the 6 `default`-profile windows use the default operating point in E8.01).

## 3. Band sweep (raw frozen score)

`w` from 0.00 to 0.30 step 0.02; review % on the 1,546; error on the two labelled sets.
Full curve (this is the deliverable curve; the fine 0.005 grid is in `Y1.01_support.json`):

| w | review% | gold BA | gold err | n dec | good-rej | OOD BA | OOD err | OOD n |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.795 | 0.205 | 150 | 11/100 | 0.471 | 0.529 | 60 |
| 0.02 | 2.65 | 0.823 | 0.177 | 142 | 8/100 | 0.471 | 0.529 | 60 |
| 0.04 | 5.82 | 0.820 | 0.180 | 139 | 8/100 | 0.473 | 0.527 | 59 |
| 0.06 | 10.93 | 0.830 | 0.170 | 134 | 8/100 | 0.435 | 0.565 | 54 |
| 0.08 | 19.92 | 0.835 | 0.165 | 125 | 7/100 | 0.409 | 0.591 | 50 |
| 0.10 | 29.56 | 0.840 | 0.160 | 118 | 6/100 | 0.425 | 0.575 | 46 |
| 0.12 | 37.00 | 0.847 | 0.153 | 114 | 6/100 | 0.439 | 0.561 | 43 |
| 0.14 | 45.08 | 0.848 | 0.152 | 109 | 6/100 | 0.419 | 0.581 | 37 |
| **0.16** | **53.36** | **0.846** | **0.154** | **102** | **6/100** | **0.471** | **0.529** | **29** |
| 0.18 | 60.74 | 0.856 | 0.144 | 94 | 5/100 | 0.458 | 0.542 | 22 |
| 0.20 | 66.24 | 0.854 | 0.146 | 82 | 4/100 | 0.344 | 0.656 | 19 |
| 0.22 | 70.25 | 0.835 | 0.165 | 67 | 4/100 | 0.344 | 0.656 | 19 |
| 0.24 | 72.77 | 0.829 | 0.171 | 56 | 3/100 | 0.393 | 0.607 | 17 |
| 0.25 (frozen) | 75.42 | 0.815 | 0.185 | 52 | 3/100 | 0.393 | 0.607 | 17 |
| 0.26 | 77.23 | 0.802 | 0.198 | 49 | 3/100 | 0.393 | 0.607 | 16 |
| 0.28 | 80.47 | 0.808 | 0.192 | 40 | 2/100 | 0.500 | 0.500 | 12 |
| 0.30 | 82.92 | 0.799 | 0.201 | 34 | 2/100 | 0.500 | 0.500 | 10 |

**Review is monotonically increasing in `w`** (wider band = more review), so "first `w`
reaching a review target" means the **widest** band that still meets it (the least
review reduction). On the prescribed 0.02 grid:

| review target | w | review% | gold BA [95 % clip-bootstrap] | gold err | good-rej | OOD BA [95 % win-bootstrap] | OOD n |
|---|---|---|---|---|---|---|---|
| **< 60 %** | **0.16** | **53.36** | 0.846 [0.662, 0.931] | 0.154 | 6/100 | 0.471 [0.229, 0.754] | 29 |
| < 50 % | 0.14 | 45.08 | 0.848 [0.669, 0.934] | 0.152 | 6/100 | 0.419 [0.183, 0.702] | 37 |
| < 40 % | 0.12 | 37.00 | 0.847 [0.672, 0.933] | 0.153 | 6/100 | 0.439 [0.200, 0.679] | 43 |

On the finer 0.001 grid the widest bands are `w = 0.177` (59.57 % review, gold err 0.150),
`0.150` (49.55 %, 0.156), `0.125` (39.84 %, 0.153). The gold error is **flat (~0.144–0.165)
across `w = 0.08–0.20`** and every point is inside the clip-bootstrap CI of the others:
within this range the band buys review reduction almost for free, and the differences
between `w = 0.12`, `0.16`, `0.18` are noise.

## 4. Calibration (fit LOCO on the gold only)

Fitted on the 150 gold windows: **temperature T = 0.865** (mild sharpening; global fit),
Platt `logit → 1.076·logit + 0.762`, isotonic (28 distinct levels). Nothing was fit on
the OOD-60 or on the 1,546.

The LOCO re-sweep (in `Y1.01.json`) *looks* like a win — e.g. LOCO temperature at
`w = 0.08` → **14.55 % review with gold BA 0.848**. **This is an artifact and we do not
claim it.** Calibration is a *monotone* map and the band is symmetric, so it cannot
reorder the frontier; it only re-parameterises the band. The consistency check
(`Y1.01_support.json`) makes this explicit:

| at gold err ≤ 0.155 | min review% |
|---|---|
| raw | 31.95 |
| temperature **in-sample** (same fit used on E8) | 31.57 |
| temperature **LOCO** | 9.44 ← artifact |
| Platt in-sample / Platt LOCO | not reachable (BA ≤ 0.73 at every `w`) |
| isotonic in-sample / LOCO | not reachable (BA ≤ 0.79) |

The in-sample temperature curve lies on top of the raw curve (same review at equal
error). The LOCO curve is lower only because **per-fold** temperature fits change *which*
gold windows land inside the band — a small-sample effect (6 training clips), not a
deployable gain. Platt and isotonic are actively **harmful**: LOCO BA at `w = 0` is
0.710 (Platt) and 0.725 (isotonic) vs **0.795 raw** — NLL-optimal calibration is not
balanced-accuracy-optimal on 150 windows.

**How much does calibration alone cut review at equal error? Essentially none.** The
only thing that cuts review is a narrower band, which re-introduces the same errors;
calibration merely relabels which `w` you are on.

## 5. Recommendation

**Operating point: raw frozen score, symmetric band half-width `w = 0.16`
(band `[0.34, 0.66]`), no calibration.**

- Review share on the 1,546 windows: **53.4 %** (from 75.5 %) → meets the < 60 % goal.
- In-domain error: gold BA **0.846** (err 0.154), 102/150 decided, 6/100 good rejected,
  95 % clip-bootstrap CI [0.662, 0.931].
- OOD-60 error: BA **0.471** (err 0.529), only 29/60 decided, CI [0.229, 0.754].

If a wider band is preferred (least change from frozen), `w = 0.177` gives **59.6 %**
review at err 0.150; if review must be cut harder, `w = 0.12` gives **37.0 %** at
err 0.153 — statistically indistinguishable from `w = 0.16`. We do **not** recommend
`w ≥ 0.18`: the review share is no longer < 60 % and no measured error improvement
survives the CI.

### OOD honesty — stated plainly

OOD decisions **remain unreliable and near chance**. On the 60 labelled windows the
frozen recipe scores **0.41–0.53 AUROC**; across the whole sweep the OOD balanced
accuracy on decided windows stays in **0.34–0.53**, i.e. at or below chance, and the
`n_bad = 9` makes every OOD CI wide (e.g. [0.229, 0.754] at `w = 0.16`). Narrowing the
band only forces *more* near-chance OOD decisions; it does not make them correct. The
review-share reduction is a **workload** result on unlabelled data, not an accuracy
result, and calibration fit on IndEgo **may not transfer OOD**.

## 6. Protocol, provenance, limits

- **n / label source:** review % → 1,546 Egocentric-10K windows, `label_source = "none"`
  (E8.01). Gold error → 150 IndEgo windows, human gold, single labeller, LOCO (7 clips).
  OOD error → 60 Egocentric-10K windows, 51 good / 9 bad, human-labelled, **test only**.
- **Reused, not modified:** the frozen policy is re-implemented from
  `eval/exp_rescore/contract.py::verdict_for` (read-only; the E8 artifacts are read-only).
- **Deterministic:** fixed seed 7; bootstrap 2,000×.
- **Limits:** OOD `n_bad = 9`; single gold labeller; gold "error" counts REVIEW windows
  as neither right nor wrong; a monotone calibrator cannot be shown to help from these
  data and the LOCO apparent gain must not be shipped.
