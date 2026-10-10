# exp_skip — Q2: skipping sections *anywhere* (middle / end / evenly)

**Claim X2** (experimenter, owner *monitor*). Branch `m/skip`, lineage `pod/base`.
Extends **E3** (E0.01/E3.01/E3.02), whose "skip the START" rule was **not** supported.

## Question

Instead of only skipping the **start** of a video, can we skip sections **anywhere**
(middle / end / evenly spread) and still get the **same clip verdict at lower cost**?

- **Data:** 7 IndEgo clips, 150 human-labelled 10-s windows (`labels/manifest_gold.json`).
  Window labels are the clip-level ground truth (bad fraction / majority verdict).
- **Scores (cached, no GPU):** `caches/ensemble/oof_probs.npz` (`p_ens`, LOCO AUROC 0.834).
- **Protocol:** divide each clip's windows (time-ordered) into 5 equal sections; simulate
  keeping/dropping sections. Metrics per policy: (a) mean processed fraction of footage,
  (b) error of the clip bad-fraction estimate vs full coverage, (c) per-clip verdict
  agreement at the **0.30** bad-fraction threshold. CI = clip bootstrap (2000 resamples).
- **Truth bad fractions:** desk 0.45, ecu_full 0.10, ecu_late 0.29, logistics4 0.18,
  organise 0.53, pallet 0.65, pallet2 0.67 → **4 reject / 3 accept** at 0.30.

> ⚠️ Windows are ~10 s but non-uniformly spaced, so "fraction of windows processed" is a
> proxy for footage fraction. Dropping **non-contiguous** windows saves feature compute but
> not decode/seeks, so a lower processed-% is not a free decode saving.

## Headline answer

**No — section skipping anywhere does not give the same verdict at lower cost than the
already-adopted uniform every-2nd-window grid.** Directionally, section position matters a
lot (bad rate climbs from 0.09 in sec1 to 0.69 in sec5), which is exactly why dropping the
start is the worst rule and dropping the end is the least-harmful contiguous one — but every
section-drop variant is **dominated** by the contiguous every-2 grid:

- Best cell = **`uniform-every-2`**: 51.5 % processed, clip error **0.049** [0.019, 0.087],
  verdict agreement **7/7** — the *only* policy meeting agreement ≥ 0.90 **and** error ≤ 0.05.
- The best section drop (`drop1-end`): 83.1 % processed, error 0.057, agreement 7/7 —
  more cost **and** more error than every-2.
- The lowest-raw-error section drop (`drop1-evenly-s2`): 80.6 % processed, error 0.033, but
  agreement **6/7** (flips `ecu_late`).
- `drop2-start` is the worst drop (error **0.189**, 55 % cost): removing the cleanest section
  inflates the estimate. This **confirms E3's "skip the start" rejection**, now with the
  opposite-signed position profile.

## Findings table

| policy | processed % | bad-frac error [CI] | verdict agree @0.30 | what worked / did NOT | verdict | why | links |
|---|---|---|---|---|---|---|---|
| **uniform-every-2** (E3 adopted) | 51.5 % | **0.049** [0.019, 0.087] | **1.00 (7/7)** | Best cell; meets 90 %/0.05 bar | **adopt** | Cheapest policy within tolerance; section drops never beat it | [SKIP.03.json](../../eval/results/exp_skip/SKIP.03.json) |
| drop1-evenly (s2) | 80.6 % | 0.033 [0.014, 0.054] | 0.86 (6/7) | Lowest raw error of ALL drops | **maybe** | Flips `ecu_late`; costs more than every-2 | [SKIP.01.json](../../eval/results/exp_skip/SKIP.01.json) |
| drop1-end | 83.1 % | 0.057 [0.041, 0.073] | 1.00 (7/7) | Safest *contiguous* drop | **maybe** | Verdict kept but error > every-2 at higher cost | [SKIP.01.json](../../eval/results/exp_skip/SKIP.01.json) |
| drop1-middle | 80.6 % | 0.063 [0.037, 0.087] | 0.86 (6/7) | Better than start, worse than end/even | **drop** | Dominated; no cost benefit | [SKIP.01.json](../../eval/results/exp_skip/SKIP.01.json) |
| **drop1-start** (control) | 76.5 % | **0.110** [0.050, 0.192] | 0.86 (6/7) | Worst drop-1; start is cleanest section | **drop** | Confirms E3: skipping the start biases high | [SKIP.01.json](../../eval/results/exp_skip/SKIP.01.json) |
| drop2-evenly (s2) | 61.5 % | 0.052 [0.018, 0.097] | 1.00 (7/7) | Only drop-2 with 7/7 at ~0.05 | **maybe** | Still ≥ every-2 cost for ≥ every-2 error | [SKIP.01.json](../../eval/results/exp_skip/SKIP.01.json) |
| drop2-middle | 59.5 % | 0.075 [0.037, 0.119] | 1.00 (7/7) | Verdict kept | **drop** | Error 0.075 > every-2's 0.049 | [SKIP.01.json](../../eval/results/exp_skip/SKIP.01.json) |
| drop2-end | 64.0 % | 0.110 [0.057, 0.159] | 1.00 (7/7) | Verdict kept | **drop** | Drops the bad-heavy end → underestimate 0.11 | [SKIP.01.json](../../eval/results/exp_skip/SKIP.01.json) |
| **drop2-start** | 55.4 % | **0.189** [0.103, 0.272] | 0.86 (6/7) | Worst overall (control) | **drop** | Removes the cleanest 40 % → strong positive bias | [SKIP.01.json](../../eval/results/exp_skip/SKIP.01.json) |
| **strat-probe** (1/section, 5 windows) | 33.1 % | 0.142 [0.080, 0.193] | 0.86 (6/7) | Beats equal-budget contiguous probe | **maybe** | err 0.142 vs contiguous 0.211; but uniform-every-3 does better (0.117 @ 34.9 %) | [SKIP.02.json](../../eval/results/exp_skip/SKIP.02.json) |
| contig-probe (first 5) | 33.1 % | 0.211 [0.104, 0.320] | 0.57 (4/7) | Control for strat | **drop** | Same budget, far worse than stratified | [SKIP.02.json](../../eval/results/exp_skip/SKIP.02.json) |
| dense-where-uncertain (x1) | 45.9 % | 0.100 | 0.86 | CUSUM/SPRT densify idea | **drop** | Beaten by uniform at equal budget (≈0.072); reproduces E3's CUSUM rejection | [SKIP.02.json](../../eval/results/exp_skip/SKIP.02.json) |
| dense-where-uncertain (x2) | 62.5 % | 0.057 | 0.71 | — | **drop** | Worse than every-2 at equal cost | [SKIP.02.json](../../eval/results/exp_skip/SKIP.02.json) |
| uniform-every-5 | 23.5 % | 0.121 [0.067, 0.188] | 1.00 (7/7) | Cheapest 7/7 policy | **maybe** | Verdict safe but error 0.121 > 0.05 bar | [SKIP.02.json](../../eval/results/exp_skip/SKIP.02.json) |

## Section position profile (why start-drop fails, end-drop biases low)

Window-weighted mean bad fraction by section:
**sec1 0.09 · sec2 0.22 · sec3 0.43 · sec4 0.41 · sec5 0.58** (unweighted: 0.09/0.35/0.56/0.53/0.69).

Bad activity concentrates in the **second half**. Dropping early clean windows → over-estimate;
dropping late bad windows → under-estimate. No single fixed drop is unbiased; only a
**time-spread** design (uniform every-k grid, or the evenly-spaced drop) samples the trend.

## Best cell (SKIP.03)

Cheapest policy with **verdict agreement ≥ 0.90** and **bad-fraction error ≤ 0.05**:

> **`uniform-every-2`** — 51.5 % processed, error **0.049** [0.019, 0.087], **7/7** verdicts.

No section-drop policy qualifies at a lower cost; relaxing the error bar to ≤ 0.07 / ≤ 0.10
still returns `uniform-every-2`. **E3.01's adopted policy stands.** Keep E3.03's 4-s early exit
(multiplicative, orthogonal) for the real cost cut.

## Figures

| figure | file |
|---|---|
| drop rules timeline (kept vs dropped sections) | [skip_timeline_drop_rules.png](../experiments/exp_skip/images/skip_timeline_drop_rules.png) |
| processed-% vs clip error | [skip_error_vs_cost.png](../experiments/exp_skip/images/skip_error_vs_cost.png) |
| per-clip verdict agreement | [skip_perclip_agreement.png](../experiments/exp_skip/images/skip_perclip_agreement.png) |
| section position profile | [skip_section_profile.png](../experiments/exp_skip/images/skip_section_profile.png) |

## Results / artefacts

- `eval/results/exp_skip/SKIP.01.json` — section-drop rules (+ per-clip)
- `eval/results/exp_skip/SKIP.02.json` — stratified / contiguous probe + dense-where-uncertain
- `eval/results/exp_skip/SKIP.03.json` — best-cell selection + per-clip detail + frontier
- scripts: `eval/exp_skip/run_skip.py`, `eval/exp_skip/make_figs.py`

## Caveats (honest)

- **n = 7 clips / 150 windows.** Clip bootstrap CIs are coarse; verdict agreement is a 7-point
  metric, so one flip moves it by 14 pts.
- Gold windows are a **sparse sample** of each clip, not exhaustive coverage → "full coverage"
  is full-gold, not full-video.
- Only **IndEgo** in-domain. OOD Egocentric-10K showed no position effect (E3.02, p=0.42,
  n_bad=9): do not extrapolate a drop rule there.
- Model error at full coverage (E3: 0.167) still dominates sampling error; sampling is not the
  current bottleneck.

## Verdicts

- **SKIP.01 → maybe** (direction confirmed: START worst, END/EVENLY safest; none adopted over every-2).
- **SKIP.02 → drop** (stratified > contiguous at equal budget, but dense-where-uncertain and
  stratified both lose to the uniform grid at equal budget).
- **SKIP.03 → adopt** (re-affirm `uniform-every-2` as best cell; no section-skip policy selected).
