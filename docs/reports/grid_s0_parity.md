# GRID.05 / GRID.09 / GRID.07 — S0 parity, brightness-calibrated blur, and cost

Stream S2 (layout / cost / quality) on branch `dsh/grid-sweep`. CPU only, this
laptop, no GPU used and no cross-device claim anywhere in this document.

All three experiments are registered in `experiments/registry.jsonl` and their
numbers live in the results JSONs named below. Every figure in this report is
`[MEASURED]` in this session unless it is explicitly tagged otherwise; the log
that produced it is the results JSON itself, whose `command` field regenerates
it.

| id | result file | verdict |
|---|---|---|
| GRID.09 | `eval/results/exp_grid/GRID.09.json` | adopt the statistic + rule, shipped DISABLED by default (see below) |
| GRID.05 | `eval/results/exp_grid/GRID.05.json` | adopt for BLACK/DARK/OVEREXPOSED/LENS_BLOCKED/FROZEN/BLUR; FROZEN attribution overlaps on dark windows |
| GRID.07 | `eval/results/exp_grid/GRID.07.json` | descriptive (cost, no gate) |

## 0. Does this change the shipped decision path? Yes — and it ships OFF

`jadal/poc/cascade.py` and `scripts/poc_run.py` are modified. Stated plainly:

* **Added, no verdict change:** `scripts/poc_run.py::s0_from_luma` now also
  returns `lapvar_cal` (160x90) and `lapvar_cal_45` (80x45). The five shipped keys
  are arithmetically identical to before (pinned by
  `tests/test_grid_s0_blur.py::test_shipped_five_keys_unchanged`).
* **Added, currently inert:** `jadal/poc/cascade.S0_RULES` gained a sixth entry
  `BLUR` as the LAST rule. Its threshold is
  `jadal/poc/s0_blur.BLUR_LAPVAR_CAL_MIN`, which **ships as `0.0` = DISABLED**;
  `lapvar_cal < 0.0` can never fire, so the shipped five rules decide exactly as
  before. The rule is appended after FROZEN so a black clip still reports BLACK.
* **Enabling it is a one-constant change** to
  `BLUR_LAPVAR_CAL_MIN_CALIBRATED = 0.021092022956273312`. Enabling it *does*
  change shipped verdicts: it rejects blurred windows at S0 that previously
  reached S1. Every number in `GRID.09.json` is measured at that calibrated
  threshold, and the JSON records `shipped_default_threshold: 0.0` and
  `rule_enabled_by_default: false` next to them.
* `jadal/mosaic/s0_parity.py` is a NEW module. Nothing in the shipped pipeline
  imports it; its two calibrated thresholds are consumed only by the GRID.05
  harness.

The tests assert both states, so neither the disabled default nor the calibrated
value can drift silently.

Supporting artifacts (all numbers, no pixels):
`grid09_calibration.json`, `grid09_head.json`, `probe_blur_windows.json`,
`grid05_calibration.json`, `grid05_clips.json`, `grid05_windows.json`.

---

## 1. Where the shipped S0 probe actually comes from

Quoted, not invented. The production consumer is
`jadal/poc/cascade.py::s0_decide` over the dict built by
`scripts/poc_run.py::s0_from_luma(lum[:60])`, which the portal calls at
`portal/pipeline.py:189` and `portal_harden/pipeline.py:307`:

```python
# scripts/poc_run.py  (before this branch)
def s0_from_luma(luma60):
    g = luma60.reshape(60, 45, 2, 80, 2).mean(axis=(2, 4))   # 160x90 -> 80x45
    return {"mean": float(g.mean()), "p5": float(np.percentile(g, 5)),
            "p95": float(np.percentile(g, 95)),
            "spatial_var": float(g[:10].mean(axis=0).var()),
            "temp_e": float(abs(np.diff(g, axis=0)).mean())}
```

with the five shipped rules

```python
S0_RULES = [("BLACK","p95","<",0.03), ("TOO_DARK","mean","<",0.05),
            ("LENS_BLOCKED","spatial_var","<",1e-5), ("OVEREXPOSED","p5",">",0.70),
            ("FROZEN","temp_e","<",5e-4)]
```

Nothing above measures sharpness, which is why a blurred upload reaches S1.

---

## 2. GRID.09 — brightness-calibrated blur rule

**What ran**

```
JADAL_BUNDLE=<bundle> nice -n 10 python eval/exp_grid/grid09.py --phase probe --workers 2
python eval/exp_grid/grid09.py --phase calibrate
# threshold frozen in jadal/poc/s0_blur.py
JADAL_BUNDLE=<bundle> python eval/exp_grid/grid09.py --phase head --head-n 8
python eval/exp_grid/grid09.py --phase evaluate
```

**Raw evidence** `eval/results/exp_grid/GRID.09.json` (+ `grid09_calibration.json`,
`grid09_head.json`, `probe_blur_windows.json`).

**Result — the failure being fixed.** The shipped 26-feature head was run over
the full 10 s path on 16 eval-split human-good windows (8 IndEgo `g00x`, 8
OOD-160 `q0xx`), with the defect injected at 480p *before* the 160x90 downsample,
exactly as a blurred upload would arrive. The rule is shipped disabled, so the
"shipped 5 rules" columns are what the branch actually does today:

| arm | n | accepted by the head | median p | max p | S0 exits (shipped 5) | S0 exits (BLUR enabled) |
|---|---|---|---|---|---|---|
| clean | 16 | 11 (68.8%) | 0.9431 | 1.0000 | 0 | 0 |
| blur σ=3 | 16 | **6 (37.5%)** | 0.5116 | **0.9999** | **0** | 11 |
| blur σ=8 | 16 | **4 (25.0%)** | 0.3495 | 0.9889 | **0** | 16 |

So the shipped five S0 rules reject **0 of 16** blurred windows at either σ, the
head's median on blurred windows collapses to ≈0.5 while a tail still scores
0.9999, and 4 of 16 σ=8-blurred windows are accepted outright — the reported
failure, reproduced and quantified.

**Result — the statistic.** `lapvar_cal` = Laplacian variance / (mean luma + eps),
one definition in `jadal/poc/s0_blur.py`, called by both
`jadal/mosaic/features._lapvar_cal` (per 224 px tile) and
`s0_from_luma` (per frame of the decoded **160x90** luma).

Resolution was chosen on the calibration split, not asserted. The 80x45 probe
buffer that the other S0 statistics use is a much weaker blur detector, because a
σ=3 blur at 480p survives the 160x90 downsample but not the extra 2×2 box:

| statistic on eval split | AUROC σ=1.5 | σ=3 | σ=8 | σ=3 + ×1.4 gain |
|---|---|---|---|---|
| `lapvar_cal` (160x90, adopted) | 0.741 | **0.955** | 1.000 | **0.927** |
| `lapvar_cal_45` (80–45 candidate) | — | 0.757 | — | — |
| `lapvar_raw_160` (uncalibrated) | — | 0.922 | — | 0.831 |
| `lapvar_raw` (80x45, uncalibrated) | — | 0.722 | — | 0.559 |
| `lapvar_cal_tmean` (temporal mean) | — | 0.632 | — | — |

**Catch and false-reject, held-out EVAL split** (grouped even/odd by clip and by
factory; calib n=106 / 26 groups, eval n=132 / 24 groups). Threshold
**0.021092022956273312** = the 2nd percentile of the clean CALIB distribution over
the **calibration split = even-index IndEgo clips (`desk`, `ecu_late`, `organise`,
`pallet2`) and even-index OOD-160 factories (23 factories); the odd-index clips
and factories are the EVAL split.** It never looked at a blurred window:

| variant | n | catch | AUROC |
|---|---|---|---|
| blur σ=1.5 | 132 | 18.9% [5.8, 40.4] | 0.741 [0.666, 0.857] |
| blur σ=3 | 132 | 72.0% [38.8, 89.4] | **0.955 [0.876, 0.991]** |
| blur σ=8 | 132 | **100%** [100, 100] | 1.000 [1.000, 1.000] |
| blur σ=3 + ×1.4 | 132 | 67.4% [33.3, 86.0] | 0.927 [0.847, 0.981] |
| blur σ=3 + ×0.5 | 132 | 92.4% [75.6, 100] | 0.990 [0.960, 0.999] |

Intervals are group bootstrap, 2000 draws, seed 0, grouped by clip/factory. The
per-dataset split is large and is reported rather than pooled away: at σ=3 the
IndEgo windows are caught **63/63 (100%)** and the OOD-160 windows **32/69
(46.4%) [21.1, 70.9]**.

**False-reject side** — 238 human-good windows (100 IndEgo + 138 OOD-160), no
synthetic defect:

| population | n | shipped-5 exits | blur-rule exits | extra exits | rate |
|---|---|---|---|---|---|
| all good windows | 238 | 12 | 12 | **0** | 5.04% [1.92, 9.96] |
| EVAL split | 132 | 9 | 9 | **0** | 6.82% [1.73, 16.4] |
| IndEgo-150 good | 100 | 3 | 3 | **0** | 3.00% [0.62, 8.52] |
| OOD-160 good | 138 | 9 | 9 | **0** | 6.52% [3.04, 12.0] |

`extra exits = 0`: every good window the blur rule rejects was already rejected
by the shipped five rules, so the marginal false-reject rate on this population
is 0/238. It is **not** zero in general, and the brightness-only controls say
why — this is the reading the decision memo asked to confirm:

| control (brightness shift only, NO blur) | n | reject rate at the frozen threshold |
|---|---|---|
| ×1.4 gain (brighter, sharp) | 132 | 4.55% pooled — **1.6% IndEgo (1/63), 7.2% OOD-160 (5/69)** |
| ×0.5 gain (darker, sharp) | 132 | **18.9% pooled — 22.2% IndEgo (14/63), 15.9% OOD-160 (11/69)** |

Confirmed: 1.6% and 7.2% are the ×1.4-gain-only control (brighter-but-sharp), and
they are *false rejects the rule would add* if it were enabled. So the
calibration is doing real work in the direction it was designed for — on the
σ=3 + ×1.4 (bright blurred) control its AUROC is **0.927 vs 0.831** for the
uncalibrated Laplacian variance at the same resolution, +9.6 points — but it is
**not gain-invariant**. `lapvar_cal` removes the *quadratic* contrast dependence
of a raw Laplacian variance and stays *proportional to gain*, so the dangerous
direction is a darkened window: ×0.5 alone rejects 18.9% of good windows. That is
the honest limitation of this statistic, and it is the same mechanism behind the
mosaic FROZEN rule firing on the `too_dark` clips in GRID.05.

**Operating-point curve** (thresholds from clean CALIB quantiles only; the frozen
2% row is the shipped one):

| clean CALIB quantile | threshold | eval FRR | σ=1.5 | σ=3 | σ=8 | σ=3 + ×1.4 |
|---|---|---|---|---|---|---|
| 0.5% | 0.01705 | 3.0% | 9.1% | 66.7% | 100% | 60.6% |
| **2% (shipped)** | **0.02109** | **6.8%** | **18.9%** | **72.0%** | **100%** | **67.4%** |
| 5% | 0.02904 | 8.3% | 45.5% | 82.6% | 100% | 76.5% |
| 10% | 0.03553 | 11.4% | 56.8% | 90.2% | 100% | 83.3% |

The Youden-J reference arm (chosen on CALIB, not shipped) sits at 0.0344 with
89.4% catch at σ=3 and an 11.4% eval false-reject rate; even there, one σ=3
window (`q012`) still reaches the head and is accepted at p=0.9935.

**Decision.** Adopt the statistic into the probe (no verdict change) and adopt the
rule, **shipped disabled** per the wrap-up instruction: at the calibrated
threshold it removes 11/16 σ=3 and 16/16 σ=8 blurred windows before S1 while
adding **zero** marginal false rejects on 238 human-good windows. Flipping
`BLUR_LAPVAR_CAL_MIN` to `BLUR_LAPVAR_CAL_MIN_CALIBRATED` is the whole change;
the two ways it can hurt are the darkened-window control (18.9%) and the
OOD-160 catch gap (46.4% at σ=3), both measured above.

**Not measured** — no real defocus/motion-smear capture was available; every
positive is a synthetic Gaussian at 480p. No per-dataset operating point is
shipped. `jadal/poc_stream/replay.py` keeps its own separate S0 rule copy and was
deliberately not touched.

---

## 3. GRID.05 — S0 parity on the mosaic canvas

**What ran**

```
python eval/exp_grid/grid05.py --phase clips
python eval/exp_grid/grid05.py --phase windows --workers 2
python eval/exp_grid/grid05.py --phase calibrate
# thresholds frozen in jadal/mosaic/s0_parity.py
python eval/exp_grid/grid05.py --phase evaluate
```

**Raw evidence** `eval/results/exp_grid/GRID.05.json` (+ `grid05_clips.json`,
`grid05_windows.json`, `grid05_calibration.json`). Scratch clips were generated
under `/tmp/opencode/grid05_clips` and are **not** committed.

**Probe.** `jadal/mosaic/s0_parity.py` rebuilds the same probe dictionary from
canvas pixels — pooled tile luma percentiles for `mean/p5/p95`, pooled pixel
variance for `spatial_var`, `it_mad_mean` for `temp_e`, mean per-tile
`lapvar_cal` — so `jadal/poc/cascade.s0_decide` can consume it unchanged. The
four brightness/geometry rules **inherit the shipped thresholds verbatim**;
FROZEN and BLUR get their own thresholds because 224 px tiles change the units:

* `MOSAIC_TEMP_E_MAX = 0.05470779972318289`
* `MOSAIC_BLUR_LAPVAR_CAL_MIN = 0.013163918969293153`

Both are the 2nd percentile of the clean CALIB distribution (n=106, same grouped
split as GRID.09). Youden-J reference on CALIB: FROZEN 0.99, BLUR 0.981.

**Test set 1 — 56 labelled synthetic clips** (7 variants × 4 CALIB-source clips ×
2 windows; `black` and `lens_blocked` are added here because
`scripts/make_demo_clips.py` has neither, and their construction is recorded in
the results JSON):

| defect | n | production, 5 rules | production, + BLUR | mosaic | mosaic fired rule |
|---|---|---|---|---|---|
| black | 8 | 100% | 100% | 100% | BLACK (8) |
| lens_blocked | 8 | 100% | 100% | 100% | LENS_BLOCKED (8) |
| frozen | 8 | 100% | 100% | 100% | FROZEN (8) |
| blur | 8 | 0% | **100%** | **100%** | BLUR (8) |
| too_dark | 8 | 12% | 88% | 100% | FROZEN (8) |
| overexposed | 8 | 0% | 0% | 0% | none (8) |
| none (control) | 8 | 0% | 0% | 0% | none (8) |

Intervals (Wilson, n=8) are in the results JSON. Agreement: **five-rule vs
five-rule 48/56 = 85.7% [74.3, 92.6]** and **six-rule vs six-rule 55/56 = 98.2%
[90.6, 99.7]**; on the six non-`too_dark` defect classes the two paths agree
56/56 for `black`/`lens_blocked`/`frozen`/`blur`/`none`.

Two honest negatives sit in that table:

1. **`overexposed` is caught by neither path (0/8).** The generator's defect
   (gain 1.7 + offset 30) lands above the shipped `p5 > 0.70` operating point on
   every source window. This is a property of the injected defect strength, not a
   parity failure — but it means OVEREXPOSED parity is *untested*, not verified.
2. **`too_dark` is rejected by the mosaic for the wrong reason (FROZEN 8/8).**
   Darkening by ×0.25 also cuts the inter-tile MAD below a threshold that sits at
   the 2nd percentile of clean windows, and the production arm's rejections are
   likewise mostly BLUR (6/8) rather than TOO_DARK (1/8). Same gain-proportional
   mechanism as the GRID.09 dark control. Right verdict, wrong attribution.

**Test set 2 — 12 unlabelled `interview/demo_kit` clips.** **These clips carry no
labels at all**, so only verdicts and agreement are reported, never a catch rate:
11/12 agree. `demo_09` is rejected as BLUR by both, `demo_10` as FROZEN by both,
`demo_11` as BLACK by both (after the padding fix — the S1 pooling reads TOO_DARK).

`demo_03` is the one disagreement: **production fires nothing, mosaic fires
FROZEN**. Explained, not hand-waved:

* The two FROZEN statistics are different measurements. Production's `temp_e` is
  the mean absolute frame-to-frame difference over the first 60 frames (2 s) at
  80x45 → `0.003136` for `demo_03`, which is 6× *above* the shipped `5e-4`
  threshold, so production correctly does not fire. The mosaic's `temp_e` is the
  mean absolute difference between consecutive 224 px tiles (9 samples over 10 s),
  and its threshold `0.0547` sits at the 2nd percentile of the clean window
  distribution — only ~2–4× below the clean median (`0.123` IndEgo / `0.186`
  OOD-160).
* `demo_03` is a very dark clip: production `mean = 0.0602`, `p95 = 0.1544` —
  close to but above `TOO_DARK` (`mean < 0.05`). Low contrast reduces the
  inter-tile MAD, so it lands just under the mosaic floor.
* So this is the same brightness-sensitivity mechanism as the `too_dark`
  attribution above, exposed by a real unlabelled upload. n = 1 clip; it is
  reported, not averaged away, and no threshold was re-tuned to remove it.

**Test set 3 — 238 clean human-good windows** with the defect injected on the
canvas (independent of the demo clips):

| variant | n | caught | fired rules |
|---|---|---|---|
| frozen | 238 | **100%** | FROZEN 238 |
| blur σ=3 | 238 | 96.2% | BLUR 215, FROZEN 14, none 9 |
| blur σ=3 (EVAL split) | 132 | 93.2% | BLUR 112, FROZEN 11, none 9 |

**False-reject side** (the mosaic's six rules on clean human-good windows):

| population | n | groups | shipped-5 | mosaic | extra | rate |
|---|---|---|---|---|---|---|
| all good | 238 | 50 | 0 | 18 | **18** | 7.56% [4.84, 11.64] |
| EVAL split | 132 | 24 | 0 | 12 | 12 | 9.09% [5.28, 15.22] |
| IndEgo-150 good | 100 | 7 | 0 | 7 | 7 | 7.00% [3.43, 13.75] |
| OOD-160 good | 138 | 43 | 0 | 11 | 11 | 7.97% [4.51, 13.71] |

**The four inherited rules disagree with the production path on 0 of 238 clean
windows** — the "easy parity" expectation holds exactly. The 18 extra rejects are
all FROZEN (6) or BLUR (12) at their calibrated floors, i.e. the mosaic's
marginally cheaper path now rejects ~7.6% of human-good windows that the shipped
S0 passes. That is the price of the two new mosaic rules and it should be read
next to GRID.09's marginal 0/238.

**A real bug found and fixed.** For non-square sources the builder letterboxes
with mid-grey, and `features.letterbox_mask` drops a row only when ≥98% of it is
mid-grey — but the burned timestamp box sits on top of the padding in the
top-right, so those rows survive and ~50% luma leaks into the pooled percentiles.
Measured: the 640×480 all-black `demo_11.mp4` reads `p95 = 0.5020`, `mean =
0.0272` → **TOO_DARK**, while the shipped probe reads `p95 = 0.0000` → **BLACK**.
Diagnostic `bg_leak_frac` on clean OOD-160 windows: **0.0823** with the S1 mask
vs **0.0127** with the strict detector. `jadal/mosaic/s0_parity.py` fixes it
locally by running the row test on the columns outside the timestamp box
(`padding_detect="strict"`, the default); the S1 pooling is retained as an
ablation with its own thresholds. The fix is pinned by
`tests/test_grid_s0_parity.py::TestPaddingLeak`. In files changed:
`jadal/mosaic/s0_parity.py`, `eval/exp_grid/grid05.py`,
`tests/test_grid_s0_parity.py`.

**Not measured** — no human labels exist for the generated clips or for
`demo_kit`; the labels are `SYNTHETIC-GOOD`/`SYNTHETIC-DEFECT` by construction.
No real lens-blocked or defocused capture was available. Container defects
(`wrong_fps`, `truncated`, `garbage`, `reupload_duplicate`) were skipped: neither
path is a container validator.

---

## 4. GRID.07 — cost, CPU only, 480p

**What ran**

```
JADAL_BUNDLE=<bundle> nice -n 10 python eval/exp_grid/grid07.py --phase all --workers 2
```

**Raw evidence** `eval/results/exp_grid/GRID.07.json` (per-pass stage timings for
every repeat are in `raw_single_worker`).

12 human-good IndEgo windows × 3 repeats, single process, every stage timed
in-process; then a **measured** 2-worker aggregate (72 mixed jobs) and a
**measured** 2-worker mosaic-only run (36 jobs). All windows come from one clip
(`ecu_full`) so the shipped arm's 30 fps normalisation covers exactly the same
footage. `[MEASURED]`, medians in ms, this laptop, one session:

| stage | shipped, 30 fps-normalised stream | shipped, raw 10 fps + repeat | mosaic (3×3, 224 px) |
|---|---|---|---|
| decode | 130.3 (ffmpeg 160x90, 300 frames) | 104.3 (ffmpeg 160x90, 100 frames) | 58.4 (cv2 480p, 9 kept frames) |
| frame repeat | 0 | 1.0 | — |
| build | — | — | 18.6 |
| S0 | 61.2 | 61.2 | 63.3 (canvas parity probe, **excluded from total**) |
| features | 151.1 (26 features) | 151.1 | 112.0 (mosaic vector, all 3 groups) |
| **total** | **348.4** (IQR 336–453) | **330.3** (IQR 314–405) | **190.5** (IQR 182–197) |

| measure | value |
|---|---|
| footage-hours per wall-hour, shipped (normalised) | **28.7** |
| footage-hours per wall-hour, shipped (raw + repeat) | **30.3** |
| footage-hours per wall-hour, mosaic, 1 worker | **52.5** |
| footage-hours per wall-hour, mosaic decode+build+features only | **75.7** |
| footage-hours per wall-hour, mosaic, measured 2 workers | **66.8** (36 jobs, 5.4 s wall) |
| footage-hours per wall-hour, mixed shipped+mosaic, 2 workers | **52.6** (72 jobs, 13.7 s wall) |
| shipped ÷ mosaic median ratio | **1.83** |
| 30 fps normalisation pre-pass (one-off per clip) | 49.7 ms per source second, i.e. **497 ms per 10 s window amortised** — not in any row above |

`footage_hours_per_wall_hour = 10000 / ms_per_window` `[DERIVED]` from the
measured medians, assuming back-to-back windows and no idle time; the formula is
recorded next to the numbers in the results JSON.

Two things the split shows that were not assumed:

* **The mosaic path is cheaper per window than the shipped full-window path**
  (190.5 vs 348.4 ms) even though it decodes at 480p instead of 160x90, because
  the shipped path pays 151 ms for the 26 temporal features plus 61 ms for the S0
  probe, while build+features on the canvas costs 131 ms and the resized decode
  only 58 ms.
* **The second worker does not buy 2×.** Measured mosaic-only throughput is 52.5
  → 66.8 footage-h/wall-h (**1.27×**) while the per-job median rises 190.5 →
  223.5 ms: the laptop is the bottleneck, not the job count. Windows per hour go
  from ~5.25 to ~6.68.

The S1 stream reported ~195 ms/window for mosaic build+features excluding decode;
this session measures **131–132 ms** for the same configuration on the same
laptop. Both are single-session medians under different desktop load; the
difference was not investigated and neither number is presented as the other's
correction.

**Not measured** — no GPU number, no cross-device claim, **no Raspberry Pi or any
other edge board** (never measured), no thermal soak, no multi-hour run, no
service overhead (upload, container, receipt write, PNG encoding of the canvas),
nothing about 1080p.

---

## 5. What could not be done

* **No real blurred or defocused footage.** Every blur positive is a synthetic
  Gaussian applied at 480p. No claim is made about motion smear or real defocus.
* **No per-dataset blur operating point.** One pooled threshold is shipped; the
  per-dataset 2% clean quantiles (0.0281 IndEgo / 0.0184 OOD-160 for GRID.09)
  differ by 1.5×, which is exactly why the per-dataset catch rates differ so much.
* **OVEREXPOSED parity is untested, not verified** — the injected defect never
  reaches either path's operating point on these source windows.
* **Mosaic FROZEN attribution is wrong on dark windows** (right verdict, wrong
  rule) for the reason measured above; no fix was attempted.
* **`jadal/poc_stream/replay.py`** has its own S0 rule copy used by the streaming
  demo; it was not changed and no blur rule was calibrated for it.
* **The blur rule is left DISABLED** (`BLUR_LAPVAR_CAL_MIN = 0.0`). Enabling it
  is a deliberate one-constant decision for the stream owner, not something this
  branch does silently.
* **Deployment note.** `jadal/poc/cascade.S0_RULES` now has six entries and the
  probe gained two keys. `s0_decide` skips keys it cannot see, so legacy stored
  probes still decide as before, but a consumer that indexes `s0[key]` directly —
  as `portal/pipeline.py` and `portal_harden/pipeline.py` do inside their rule
  loop — would raise on a stored pre-GRID.09 probe dict once the rule is enabled.
  Both portal paths recompute the probe from frames (`poc_run.s0_from_luma`), so
  live scoring is unaffected; replaying stored `s0` dicts from before this branch
  is not.
* **One pre-existing test-collection error is not mine**: `tests/test_exp_motion_cue.py`
  imports `pytest`, which is not installed in `/home/parshu/envs/jadal`, so
  `unittest discover -s tests` reports 809 tests with `errors=1` (that module) and
  11 skips. Everything else passes, including the 28 new GRID tests.
