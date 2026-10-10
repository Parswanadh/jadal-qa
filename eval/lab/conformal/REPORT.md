# Direction 4 — Distribution-free risk control at n=150: REPORT

Subagent branch `feat/breakthrough-lab`, worktree `/home/parshu/projects/jadal-qa-wt/breakthrough-lab`.
Commit nothing, pushed nothing. Only `eval/lab/conformal/` written.
Scores used: **own LOCO cross-fitted baseline** (Arm-B score files cover only the old 79-window
subset, so they were not used). Read-only on all shared data. CPU-only, OMP_NUM_THREADS=2.

Evidence tags per project rule: `[MEASURED]` = printed by a command this session
(raw JSON in `results/`); `[DERIVED]` = arithmetic on measured inputs (formula shown);
`[EXTRAPOLATED]` = scaled beyond measurement (assumption stated); `[NOT MEASURED]` = placeholder.

What ran: `OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python eval/lab/conformal/proto_conformal.py`
Raw evidence: `eval/lab/conformal/results/00_*`–`06_*.json`, run manifest `00_manifest.json`.

## 0. Data actually in hand [MEASURED]

- n=150 windows: 100 usable (label 1), 50 bad (label 0). 7 clips:
  ecu_full 48 (43 good), desk 29 (16), organise 19 (9), logistics4 17 (14),
  pallet 17 (6), ecu_late 14 (10), pallet2 6 (2).
- Wearers extracted from file paths (`User_<id>` regex): **only 2 distinct wearers**
  (`user_1`, `user_15`) [MEASURED] — fewer than the "~3 wearers" in the brief; the third
  wearer is [NOT MEASURED] in the manifest paths.
- Baseline score s(x)=P(usable): L2-logistic (C=1) on 4 cheap Arm-A signals
  (log blur_var, brightness_mean, motion_mean, hands_visible_frac),
  **leave-one-clip-out cross-fitted** (fit on 6 clips, score the 7th; no in-sample optimism).
  LOCO AUROC = **0.7532** [MEASURED] (`01_scores.json`). This is a diagnostic baseline,
  NOT a product model.

## 1. Arithmetic check: "Bonferroni fixed-sequence needs ~86 clean fires, CERTIFY has 84"

Zero-failure rule (Bonferroni over m hypotheses, target error p, confidence 1−δ):

n_req = ceil( ln(δ/m) / ln(1−p) )   [DERIVED, verified in `02_arithmetic.json`]

| p | δ | m | n_req [DERIVED] |
|---|---|---|-----------------|
| 0.05 | 0.05 | 1 | **59** |
| 0.05 | 0.05 | 2 | 72 |
| 0.05 | 0.05 | 3 | 80 |
| 0.05 | 0.05 | **4** | **86** |
| 0.05 | 0.05 | 6 | 94 |
| 0.035 | 0.05 | 1 | 85 |
| 0.03 | 0.05 | 1 | 99 |

Exact Clopper–Pearson 1-sided 95% upper bound with k=0: n=84 → **3.5035%**; n=86 → **3.4234%** [DERIVED].

Restatement (blunt): the "~86" number is **conditionally correct** — it is exactly the
zero-failure requirement for (p=5%, δ=5%, Bonferroni m=4): ln(0.0125)/ln(0.95)=85.4→86 [DERIVED].
The "CERTIFY split has 84" figure is taken from the brief — it is [NOT MEASURED] by this
direction (no certify-split artifact was inspected here). **Conditional verdict: 84 < 86, so a
Bonferroni-m=4 fixed-sequence certification at 5%/95% fails by 2 windows.** But the claim is
fragile both ways: with m=1 only 59 clean fires are needed, so 84 would *pass*; certifying
≤3.5% needs 85 (fails by 1), certifying ≤3% needs 99. And all of this is moot for our operating
point because we observe **33 rejects out of 100 good windows, not zero** — zero-failure math
does not apply to the system as built (§3).

## 2. Best honest guarantee (split-conformal / CRC with clip-grouped exchangeability)

Protocol (pre-committed before looking at scores): calibration clips = 4 largest
(ecu_full, desk, organise, pallet; n_cal_good=73), test clips = 3 smallest
(logistics4, ecu_late, pallet2; n_test_good=26). Rule "keep iff s ≥ τ".
CRC selection (Angelopoulos et al. 2022, B=1): τ̂ = sup{τ : (n/(n+1))·R̂(τ) + 1/(n+1) ≤ α} [MEASURED].

| α | τ̂ [MEASURED] | cal emp. risk (k/n) | cal CP-upper 95% | HB p-value (H0: risk≥α) | **held-out test risk** | test CP-upper | test retention |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.4332 | 0.027 (2/73) | 0.083 | 0.153 | **0.462 (12/26)** | 0.638 | 0.595 |
| 0.10 | 0.5186 | 0.081 (6/73) | 0.154 | 0.657 | **0.500 (13/26)** | 0.673 | 0.568 |

(`03_crc.json`.) LOCO stability: CRC τ̂ recomputed leaving each clip out has mean 0.194 ± 0.114
at α=0.05 (one held-out clip, logistics4, pushes τ̂ to 0.47; rest sit ≈0.13–0.15) [MEASURED]
(`04_crc_loco_stability.json`).

**Headline: the conformal guarantee does not transfer.** Calibration promises ≤5% false-reject;
held-out clips show 46%. The CRC *marginal* guarantee E[risk]≤α is technically computed correctly,
but its exchangeability premise is refuted by the test measurement itself. The best honest
finite-sample statement from the split is therefore the **test-side** one:
at τ̂=0.4332, held-out false-reject rate 12/26 = 46.2% with 1-sided 95% CP upper bound **63.8%**
[DERIVED]. That is a valid bound on *those three clips*, not on deployment.

## 3. Operating point + Clopper–Pearson bounds (pooled vs clustered)

"Youden-J" threshold on pooled LOCO scores (in-sample-selected, stated as such):
τ\*=0.7233 [MEASURED] (`05_operating_point.json`).

- False-reject (good called bad): 33/100 = **33.0%, 95% CI [23.9%, 43.1%]** (two-sided CP, pooled) [DERIVED].
- False-accept (bad called good): 7/50 = **14.0%, 95% CI [5.8%, 26.7%]** [DERIVED].
- Per-clip FR rates [MEASURED]: logistics4 **14/14 = 100%** (CI [76.8%, 100%]),
  pallet2 2/2 = 100%, pallet 4/6 = 67%, organise 4/9 = 44%, desk 6/16 = 38%,
  ecu_full 3/43 = 7%, ecu_late 0/10 = 0% (CI [0%, 30.8%]).
- Clustering: ICC (windows within clip, FR indicator) = **0.583** [DERIVED, one-way random effects];
  design effect **8.75**, effective n ≈ **11.4 of 100** [DERIVED].
  Kish-adjusted FR CI: **[10.9%, 69.2%]** [DERIVED]. Cluster bootstrap (resample 7 clips, B=5000):
  FR 95% interval **[12.8%, 74.4%]**, CRC τ̂(0.05) 95% interval [0.077, 0.552] [MEASURED].
- Confound to state plainly: τ\* was selected on the same pooled scores it is evaluated on
  (optimistic); the LOCO fitting only removes *model-fit* optimism, not *threshold-selection*
  optimism. Pooled CIs assume independent windows — refuted by ICC=0.58.

## 4. Assumption audit (blunt)

Claimed exchangeability: "windows exchangeable across clips; calibration clips stand in for
deployment footage." Verdict: **fails on every axis we can test** [MEASURED]:

1. *Across clips:* test risk (46%) vs calibration risk (2.7%) differ by 17×; per-clip FR spans
   0%→100%. Clips are not draws from one distribution — logistics4 is systematically
   score-invisible (all 14 good windows rejected).
2. *Across wearers:* only 2 wearers in 150 windows [MEASURED]; a third wearer, any new camera
   rig, any new site = unrepresented. Nothing in n=150 licenses a wearer-level claim.
3. *Across time:* all windows come from the same 7 clips/dataset; deployment footage is a
   distribution shift (new scenes, lighting, activity mix), which conformal coverage does not
   survive without weighting (Barber et al. 2023).
4. *Label-conditional structure:* risk is conditioned on label=good (Mondrian/class-conditional
   setup), which needs exchangeability *within* the good class across clips — the per-clip table
   refutes exactly this.
5. What *is* defensible: window exchangeability **within a clip** is not rejected by anything we
   measured (no within-clip trend test was run — [NOT MEASURED]); clip-level resampling CIs (§3)
   are honest about the clustering but with G=7 clips they are ±30pp wide, which is the finding.

## 5. Labels needed + active-labelling plan

Power calc at observed FR p=0.33 (normal approx, then ×DEFF=8.75 for clustering) [DERIVED,
`06_power.json`]:

| target ±half-width (95%) | windows (i.i.d. fantasy) | windows (×DEFF, [EXTRAPOLATED]) | ≈ extra clips like today's |
|---|---|---|---|
| ±5pp | 340 good | ~2,970 good (+2,870 vs today) | ~200 |
| ±3pp | 944 good | ~8,260 good | ~570 |
| ±2pp | 2,124 good | ~18,600 good | ~1,300 |

The clustered column assumes future clips resemble today's (ICC≈0.58) — [EXTRAPOLATED] and
pessimistic by construction, because today's ICC is inflated by a *broken score* on logistics4,
not just irreducible noise. **Do not label 200 more clips of the same kind.** Instead:

1. **Fix the score first** (1–2 days, no labels): logistics4's 14/14 rejection with AUROC≈0.75
   elsewhere screams missing feature (darker scene? different motion regime?). Diagnose per-clip
   score distributions before buying labels; a better score lowers ICC and collapses the table above.
2. **Label new clips, not more windows per clip** (DEFF≈8.75 means ~9 windows in one clip carry
   the information of ~1 independent window). Priority: (a) new wearers/sites (only 2 wearers today);
   (b) scenes like logistics4/pallet (warehouse aisles, pallet handling) where FR is 67–100%;
   (c) more BAD windows (only 50 today; FA CI is [5.8%, 26.7%]) especially rare reasons —
   UNDECODABLE has n=1, TOO_DARK n=6, DUPLICATE n=6.
3. **Re-run this exact prototype** after each labelling batch: the LOCO τ̂-stability check (§2)
   is the acceptance test — when leaving any one clip out stops moving τ̂ by >2×, calibration
   is becoming clip-robust. Target: ≥12–15 clips before printing any clip-generalized number.
4. Certification path: with a fixed score and a *fresh* certify set, the zero-failure table (§1)
   sets the bar (59 clean good-windows for 5%@95% single-hypothesis; 86 with Bonferroni m=4).
   Today's system (33% rejects) cannot enter that path — it would fail the first window.

## 6. What to print on the dashboard (one paragraph)

> Quality-gate v0 (baseline, n=150 windows / 7 clips / 2 wearers): at threshold τ\*=0.72 the
> observed false-reject rate is 33% (95% CI 24–43% pooled; 11–69% after clip-clustering,
> effective n≈11) and false-accept 14% (CI 6–27%). Calibration on 4 clips promised ≤5%
> false-reject but measured 46% on 3 held-out clips — thresholds do not transfer across clips,
> so no deployment-wide error guarantee is claimed. NOT claimable: any "certified ≤X% error,"
> any per-wearer or deployment-footage number, anything using the 86-vs-84 arithmetic (that
> comparison concerns a zero-failure regime; this system has 33 observed rejects).

## 7. Literature (fetched vs recalled)

Fetched (identifier retrieved this session via web search):
- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control* — arXiv:2208.02814
  (v1 Aug 2022; v4 Jun 2025); ICLR 2024 spotlight. CRC selector τ̂ used in §2. [FETCHED]
- Angelopoulos, Bates, Candès, Jordan, Lei — *Learn then Test: Calibrating Predictive
  Algorithms to Achieve Risk Control* — arXiv:2110.01052 (2021); journal version Ann. Appl.
  Stat. 19(2), 2025, DOI 10.1214/24-AOAS1998. Fixed-sequence testing + Hoeffding–Bentkus
  p-values used in §1–2. [FETCHED]
- Clopper & Pearson — *The use of confidence or fiducial limits illustrated in the case of
  the binomial* — Biometrika 26(4):404–413, 1934, DOI 10.1093/biomet/26.4.404. All CP bounds. [FETCHED]
- Barber, Candès, Ramdas, Tibshirani — *Conformal prediction beyond exchangeability* —
  Ann. Statist. 51(2):816–845, 2023, DOI 10.1214/23-AOAS1998 (arXiv:2202.13415). Non-exchangeability
  coverage-gap framing for §4. [FETCHED]
Recalled (seen only inside fetched sources' reference lists, not independently retrieved —
descriptions only, cite at own risk):
- Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World* (Springer, 2005) —
  the conformal-prediction foundation text. [RECALLED]
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification* — arXiv:2107.07511 (2021). [RECALLED]
- Clustered/grouped conformal (e.g. class-clustered conformal for many-class problems,
  arXiv:2306.09335 lineage) — concept referenced for the clip-as-group design; no specific
  theorem invoked. [RECALLED]

## 8. Not measured (mandatory)

- The "CERTIFY split has 84" input (§1) — belongs to another direction; taken as given.
- Within-clip temporal exchangeability; wearer-conditional risks (only 2 wearers, no per-wearer split run).
- Arm-B spectral scores on full n=150 (files cover 79 windows only); whether a stronger score
  would restore clip transfer.
- Anything about deployment footage: no deployment sample was available; all bounds are
  in-sample-of-7-clips.
- Confounds: τ\* threshold-selection optimism (§3); Kish CIs are approximate under high ICC;
  power table assumes ICC persists post-fix (pessimistic — see §5).

## 9. Files

- `proto_conformal.py` — full prototype (this report is its output interpreted).
- `results/00_manifest.json`, `00_data_summary.json`, `01_scores.json`, `02_arithmetic.json`,
  `03_crc.json`, `04_crc_loco_stability.json`, `05_operating_point.json`, `06_power.json`.
- No other `eval/lab/*` dirs touched. No ports used; no processes killed; nothing committed.
