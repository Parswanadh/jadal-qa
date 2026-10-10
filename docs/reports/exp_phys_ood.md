# Q2 — Deterministic physics rule-list transfer on 160 OOD human labels

Branch `m/phys-ood` (off `pod/base`). Experiment **Q2.01** (registry:
`experiments/registry.jsonl`; result: `eval/results/exp_phys/Q2.01.json`).
This measures how the **frozen, deterministic physics rule-list model** (fixed
IndEgo thresholds, quantised, no per-window fitting) transfers to the 160 human
OOD windows, and which individual rules survive the domain shift.

All numbers come from the committed result JSON; it carries `n`, `label_source`,
`protocol`, `rule_source`, `threshold_rule`, `ci95`. Reproduce:

```
OMP_NUM_THREADS=2 python3 eval/exp_phys/extract_ood.py --jobs 2   # ~3 min CPU
OMP_NUM_THREADS=2 python3 eval/exp_phys/run_phys_ood.py
python3 eval/exp_phys/make_figure.py
```

## TL;DR

**The deterministic rule list does not transfer. Verdict: drop as an OOD decider.**

| model | OOD AUROC [factory-grouped 95% CI] | OOD bal.acc [grouped CI] | rejects | false-accept (bad) | false-reject (good) |
|---|---|---|---|---|---|
| **R3** frozen `perpixel_v1` quality stage (`lowent`,`blur`,`ten`) | **0.475** [0.327, 0.630] | **0.441** [0.325, 0.581] | 60/160 (37.5 %) | **16/22 (72.7 %)** | **54/138 (39.1 %)** |
| **R5** active rules (`blur`,`lowent`,`ten`,`madfroz`,`pbytes`) | **0.493** [0.349, 0.636] | **0.456** [0.337, 0.597] | 63/160 (39.4 %) | **15/22 (68.2 %)** | **56/138 (40.6 %)** |
| frozen `ensemble_v1` `usable_prob` (logistic baseline) | **0.566** [0.424, 0.700] | **0.546** [0.444, 0.661] | — | — | — |

Both rule sets sit **at or below chance** on AUROC and **below chance** on
balanced accuracy: the list **rejects more GOOD windows (39–41 %) than BAD
windows (27–32 %)**. It is *anti-correlated* with the OOD label, so it is worse
than doing nothing and worse than the frozen logistic baseline it was meant to
beat. No threshold was changed and nothing was fitted on OOD.

## 1. Protocol

- **Rule table (frozen, reused read-only).** The verified `B/verify-rules`
  artefact — `configs/perpixel_v1.json` (pre-registered quality stage:
  `lowent`, `blur`, `ten`) and `eval/results/perpixel/rulelist.json` (all 12
  candidate rules + frozen thresholds, active = `{blur, lowent, madfroz, pbytes,
  ten}`). Thresholds are the **IndEgo training** values (nested LOCO, Q = 4
  fixed-point), exactly the "fixed thresholds from the IndEgo training clips
  only, quantised, no per-window fitting" method. No OOD tuning.
- **Features.** `jadal/perpixel/{series,physics,theory}.py` (vendored read-only
  from `b/verify-lb2-main`), 160×90 ffmpeg decode, `extract_physics` +
  `extract_theory` per window. 160/160 extracted, 0 errors, 170 s on 2 CPU
  workers.
- **Decision.** BAD iff **any** rule fires (OR). Graded score = max normalised
  margin over the rule set (positive = fires); `usable_score = −score_bad`.
- **CI.** **Factory-grouped** cluster bootstrap (n_boot = 2000, seed 0) is the
  honest CI (~1 window/factory); window bootstrap is optimistic.

## 2. Review / error cost

There is **no review/error trade-off to buy**: because the score is
anti-correlated OOD, widening the review band removes the *confident* decisions
first and balanced accuracy falls monotonically.

| review band `w` | REVIEW share | accept | reject | bal.acc (review = miss) |
|---|---|---|---|---|
| 0.00 | 0.0 % | 97 | 63 | **0.456** |
| 0.02 | 3.8 % | 93 | 61 | 0.442 |
| 0.05 | 13.8 % | 84 | 54 | 0.394 |
| 0.10 | 30.0 % | 69 | 43 | 0.282 |
| 0.15 | 53.1 % | 42 | 33 | 0.206 |
| 0.20 | 83.8 % | 1 | 25 | 0.049 |

At **0 % review** the model already hard-rejects 39 % of good windows to catch
27 % of bad ones. Adding review only destroys accuracy. **Verdict: not viable;
the review/error cost is unbounded (review makes it worse).**

## 3. Per-rule transfer table

Fire counts and OOD precision (`P(bad | rule fires)`) on the 160 windows.
`train_J` / `train_frr` are the frozen IndEgo values.

| rule | class | feature (op) | frozen thr | fires OOD | fires good | fires bad | **OOD precision** | OOD recall | good false-alarm | train J / FRR | OOD safety |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `madfroz` | activity | `mad_frozen_frac` > | 0.60 | 6 | 4 | 2 | **0.333** | 0.091 | 0.029 | 0.143 / 0.00 | weak |
| `clipfrac` | activity | `clip_frac` > | 0.20 | 17 | 14 | 3 | 0.176 | 0.136 | 0.101 | 0.000 / 0.00 | UNSAFE |
| `dark` | quality | `p05` < | 0.009 | 29 | 25 | 4 | 0.138 | 0.182 | 0.181 | −0.030 / 0.03 | UNSAFE |
| `overexp` | quality | `p95` > | 0.6053 | 159 | 137 | 22 | 0.138 | 1.000 | 0.993 | −0.434 / 0.72 | UNSAFE |
| `blur` | quality | `blur_norm` < | 95.0933 | 37 | 33 | 4 | 0.108 | 0.182 | 0.239 | 0.429 / 0.00 | UNSAFE |
| `ten` | quality | `ten_norm` < | 339.275 | 34 | 31 | 3 | 0.088 | 0.136 | 0.225 | 0.399 / 0.03 | UNSAFE |
| `lowsnr` | quality | `snr_db` < | 8.140 | 33 | 32 | 1 | 0.030 | 0.045 | 0.232 | −0.030 / 0.03 | UNSAFE |
| `pbytes` | activity | `p_size_med` < | 2183.6 | 2 | 2 | 0 | 0.000 | 0.000 | 0.014 | 0.246 / 0.04 | UNSAFE |
| `lowent` | quality | `hist_ent` < | 6.4861 | **0** | 0 | 0 | — | 0.000 | 0.000 | **0.704 / 0.01** | abstains |
| `frozen` | activity | `frozen_idx` < | 1.50 | 0 | 0 | 0 | — | 0.000 | 0.000 | 0.000 / 0.00 | abstains |
| `ffroz` | activity | `frm_frozen_frac` > | 0.05 | 0 | 0 | 0 | — | 0.000 | 0.000 | 0.000 / 0.00 | abstains |
| `still` | activity | `still_run` > | 0.70 | 0 | 0 | 0 | — | 0.000 | 0.000 | 0.000 / 0.00 | abstains |

**Reading the table.**

- **No rule transfers as a decider.** The single best OOD precision is
  `madfroz` at **0.333** (fires 6 windows, catches 2/22 bad) — barely better than
  the 0.138 base rate and far too sparse to use.
- **The two strongest IndEgo quality rules fail differently.**
  `lowent` (IndEgo J = 0.704, the best rule in-domain) **never fires OOD**
  (histogram entropy is systematically higher on 1080p30 HEVC: median
  `hist_ent` ≈ 7.5 vs the 6.49 threshold) → it **abstains**, contributing nothing.
  `blur` (IndEgo J = 0.429) and `ten` (J = 0.399) fire often but almost
  exclusively on **good** windows (OOD precision 0.09–0.11) → pure false alarms.
- **The shift is resolution/codec, not semantics.** `blur_norm` is a
  *dimensionless* Laplacian-variance/σ² ratio, yet its OOD distribution sits well
  below the IndEgo threshold (good windows at `blur_norm` ≈ 78–90 vs the frozen
  95.09) — the 160×90 downscale of 1080p30 HEVC has different sharpness/noise
  statistics than the 480p10fps IndEgo clips. Absolute thresholds do not carry
  across the regime change.
- **`overexp` is degenerate OOD** (fires 159/160, precision = base rate): the
  IndEgo `p95` threshold is simply out of range. It was already a fallback rule
  in-domain; OOD it is a constant reject.
- **Activity rules are not the problem — quality rules are.** The frozen
  motion/frozen detectors (`frozen`, `ffroz`, `still`) **abstain** (0 fires, no
  harm); the quality tripwires are the ones that misfire. This inverts the
  task's prior ("quality rules should transfer"): here **no** class transfers,
  and the quality thresholds are the active source of error.

**Safe OOD / must abstain.**

- **Safe (no OOD harm):** `lowent`, `frozen`, `ffroz`, `still` — all abstain
  (0 fires). They can be kept as *no-ops* but carry no OOD signal.
- **Must abstain (anti-correlated false alarms):** `blur`, `ten`, `dark`,
  `lowsnr`, `pbytes`, `clipfrac`, `overexp`.
- **Only rule with any OOD precision:** `madfroz` (0.333, recall 0.09) — weak,
  not usable alone.

## 4. Batch split (descriptive)

Batch is not a random split, so this is descriptive only.

| batch | n | good/bad | AUROC (R5) | bal.acc | rejects |
|---|---|---|---|---|---|
| q1 (old 60) | 60 | 51/9 | 0.501 | 0.526 | 24 |
| q2 (new 100) | 100 | 87/13 | 0.479 | 0.408 | 39 |

Both batches are at/below chance; the failure is not a single-batch artefact.

## 5. Verdict

**Is a complete deterministic physics model viable OOD? No — at any review
cost.**

- The frozen deterministic rule list scores **AUROC 0.475 (R3) / 0.493 (R5)**,
  both below chance, and **balanced accuracy 0.441 / 0.456**, both below chance.
  It is **worse than the frozen logistic baseline** (0.566 AUROC / 0.546 BA).
- The error cost is the wrong way round: it **rejects 39–41 % of good windows**
  to catch only 27–32 % of bad ones (false-accept 15–16/22, false-reject
  54–56/138).
- **Review does not help**: widening the band monotonically lowers balanced
  accuracy (0.456 → 0.049 as review grows to 84 %).
- Per-rule, **nothing survives the shift as a decider**. The in-domain best rule
  (`lowent`) abstains; the sharpness rules (`blur`, `ten`) misfire on good
  windows; only `madfroz` has >chance precision, at negligible recall.

**Recommendation.** Do **not** ship the deterministic physics rule list as an
OOD gate, not even as a review tripwire. Keep the rules as **abstain-only**
no-ops (`lowent`, `frozen`, `ffroz`, `still`) and gate with the adapted
per-pixel model instead (see O2/O4: logistic on transferable per-pixel features,
AUROC 0.65–0.70). Any absolute IndEgo threshold (`blur_norm`, `ten_norm`,
`hist_ent`, `p95`) must be **re-derived per regime** before reuse — the 160×90
downscale of 1080p30 HEVC moves them all.

## Caveats

- OOD = **160 human windows, 22 bad, 47 factories, single labeller**; ~1
  window/factory → wide CIs, exploratory. Factory-grouped bootstrap is the
  honest CI.
- The rule table is the verified `B/verify-rules` artefact (PR #124); reused
  read-only. Q1's own worktree (`m/phys-rules`) had not produced a rule table at
  run time, so the same-method frozen table was reused verbatim.
- Feature extraction reuses the vendored `jadal/perpixel` code; 160/160 windows
  decoded, 0 errors. No labels were edited.
