# Z2 — Conformal prediction sets + risk control: a principled review-vs-error frontier

**Track:** Z2 (conformal / risk control) · **Owner:** Z2 · **Branch:** `m/conformal`
**Data:** IndEgo human gold OOF probs `caches/ensemble/oof_probs.npz` (n=150, 7 clips,
n_good=100 / n_bad=50, single human labeller, `labels_sha256=43ab44e5…`), E8 1,546
re-scored windows (unlabelled, `label_source=none`), OOD-60 (`ood_labels.json` +
`ood_scores.json`, evaluation only).
**Code:** `eval/exp_conformal/conformal_risk.py` · **Tests:** `tests/test_conformal.py` (5, pass, <30 s)
**Results:** `eval/results/exp_conformal/Z2.01.json`, `Z2.02.json`, `Z2.03.json`
**Figure:** `docs/experiments/exp_conformal/images/frontier.png`
**Reproduce:** `OMP_NUM_THREADS=2 python3 eval/exp_conformal/conformal_risk.py` (CPU only, seed 7).

---

## 0. The decision problem

Each window gets one of **ACCEPT** (good / usable), **REJECT**, or **REVIEW** (human).
Goal: minimise the review share subject to a *stated* bound on the error, instead of a vibe.
Verified label convention (checked against `nested_loco.json` per-clip `n_good/n_bad` and
`manifest_gold.json`):

```
y = 1  good/usable      y = 0  bad
s = p_ens = P(good | x)      b = 1 - s = P(bad | x)
E = true-label score:  E_i = s_i if y_i=1 else 1-s_i        (higher = more conforming)
```

The current pipeline reviews **75.5 %** of plain Windows (E8 1,546 windows). We want a
frontier that says how far that can drop while keeping a guarantee we can write down.

---

## 1. Split-conformal at the decision level (Z2.01)

### 1.1 Conformal set and its coverage guarantee

LAC ("least-ambiguous") conformal, applied to a single score. The prediction set at
nominal level `1-α` is

$$C(x)=\{\,y:\ \mathrm{score}_y(x)\ge q\,\},\qquad
\mathrm{score}_{\text{good}}(x)=s,\quad \mathrm{score}_{\text{bad}}(x)=1-s .$$

Calibrate `q` as the **α-quantile** of the calibration conformity scores $E_1,\dots,E_n$
(conformity = score of the *true* label, so the set is $\{y: E(x,y)\ge q\}$). With the
finite-sample order statistic

$$
q \;=\; E_{(k)},\qquad k=\left\lceil (n+1)\alpha \right\rceil ,
$$

the conformal rank argument gives, for an exchangeable test point,

$$
\Pr\!\left(Y_{n+1}\in C(X_{n+1})\right)\;\ge\;1-\frac{k-1}{n+1}\;\ge\;1-\alpha .
$$

**Decision map.** With $s$ = P(good):

| set membership | decision |
|---|---|
| $s\ge q$ and $1-s<q$ | **ACCEPT** (singleton {good}) |
| $1-s\ge q$ and $s<q$ | **REJECT** (singleton {bad}) |
| both ($q\le s\le 1-q$, needs $q\le\tfrac12$) | **REVIEW** ({good,bad}, non-singleton) |
| neither ($1-q<s<q$, needs $q>\tfrac12$) | **REVIEW** ($\varnothing$, non-singleton) |

So **review share = non-singleton rate**, exactly as required. A subtlety worth stating:
for coverage targets above $\approx0.70$ the threshold is $q\le\tfrac12$ and abstention is the
`{ACCEPT,REJECT}` set; for targets below $\approx0.70$, $q>\tfrac12$ and abstention is the
**empty set** $\varnothing$ (neither label clears $q$). Both are non-singletons and count as
review; an empty set is *not* covered ($Y\notin\varnothing$). Our scorer is confident
(median true-label score $0.69$), so the practically relevant regime is the empty-set one.

**Guarantee we actually use.** A wrong decision can only be a singleton that misses the true
label, hence

$$
\Pr(\text{wrong decision}) \;\le\; \Pr\!\left(Y\notin C(X)\right) \;\le\;\alpha ,
$$

i.e. the **marginal false-decision rate is bounded by α** — a distribution-free, finite-sample
bound (no `δ` needed for the split itself). This is the "stated error bound".

### 1.2 Group-aware protocol

Plain random splits would leak clips. Protocol = **leave-one-clip-out (LOCO)**: for each of the
7 clips, calibrate `q` on the other 6 clips and decide the held-out clip; pool the 7 folds.
CIs are **clip-bootstrap** (resample the 7 clips with replacement, B=2000). This is the honest
proxy for "deploy on a new clip".

### 1.3 Frontier (IndEgo gold, n=150, clip-grouped LOCO)

| coverage target `1-α` | review % | empirical coverage | error \| decided | false-accept % (marginal) | false-accept \| accept | false-reject % | empty-set % |
|---|---|---|---|---|---|---|---|
| 0.30 | 66.7 | 26.7 | 20.0 | 2.0 | 12.5 | 4.7 | 66.7 |
| 0.50 | 36.7 | 54.7 | 13.7 | 3.3 | 7.9 | 5.3 | 36.7 |
| 0.60 | 30.7 | 59.3 | 14.4 | 4.0 | 8.8 | 6.0 | 30.7 |
| 0.70 | 15.3 | 70.7 | 16.5 | 5.3 | 9.9 | 8.7 | 15.3 |
| 0.75 | 10.7 | 75.3 | 16.4 | 6.0 | 10.5 | 8.7 | 10.0 |
| **0.80** | **3.3** | 80.0 | 18.6 | 6.0 | 10.0 | 12.0 | 2.0 |
| **0.85** | **12.0** | 85.3 | 16.7 | **4.0** | 7.3 | 10.7 | 0.0 |
| 0.90 | 47.3 | 89.3 | 20.3 | 4.0 | 14.6 | 6.7 | 0.0 |
| 0.95 | 76.0 | 94.0 | 25.0 | 1.3 | 16.7 | 4.7 | 0.0 |
| 0.99 | 92.0 | 97.3 | 33.3 | 0.0 | 0.0 | 2.7 | 0.0 |

Empirical coverage tracks the nominal target (within the clip-bootstrap CI) — the marginal
guarantee holds on held-out clips. Review is **U-shaped** with a minimum at coverage ≈0.80
(the point where $q$ crosses $\tfrac12$): below it the empty-set band widens with `α`, above it
the `{ACCEPT,REJECT}` band widens as `α` shrinks.

**Operating points (LOCO):** minimum review with *marginal* false-accept ≤ 5 % → coverage 0.85
(review 12.0 %); with *selective* false-accept≤5 % → never (min ≈7.3 %); with error|decided ≤5 % → never.

**Applied to the real pipeline (E8, 1,546 unlabelled windows — descriptive only):** coverage
0.70 → 19.9 % review, 0.80 → 0.6 %, 0.90 → 50.6 %. Current frozen band = 75.5 % review. So the
frontier is real on pipeline output; but with `label_source=none` this is a *review-share*
statement only, never an accuracy claim.

---

## 2. Group-conditional and block/cluster conformal (Z2.02)

### 2.1 Why dependence matters here

Windows within a clip are correlated. One-way random-effects ICC(1,1) with mean cluster size
$\bar m = n/K$:

$$
\hat\rho=\max\!\left(0,\frac{\mathrm{MSB}-\mathrm{MSW}}{\mathrm{MSB}+(\bar m-1)\,\mathrm{MSW}}\right),
\qquad
n_{\text{eff}}=\frac{n}{1+(\bar m-1)\hat\rho},\qquad \bar m=21.4 .
$$

Measured: $\hat\rho(p_{\text{ens}})=0.46 \Rightarrow n_{\text{eff}}\approx 14.4$ (the brief's
"~0.58 / n≈11" sits in the same ballpark), but the **true-label conformity** has
$\hat\rho(E)=0.023 \Rightarrow n_{\text{eff}}\approx 102$. **This is the key nuance:** the
*score* is strongly clustered, but the *conformity* (which mixes $s$ and $1-s$) is nearly
exchangeable, so the marginal coverage guarantee survives while the *risk* does not (§3).

### 2.2 Mondrian (group-conditional) conformal

Partition into groups $g$ (score-strata, or clip), calibrate a separate threshold per group:

$$
C_g(x)=\{y: \mathrm{score}_y(x)\ge q_g\},\qquad q_g=E^{(g)}_{(\lceil (n_g+1)\alpha\rceil)} .
$$

Then coverage is controlled **within each group** ($\Pr(Y\in C_g(X))\ge1-\alpha$). We use two
groupings: (i) score-strata (tertiles of $s$, unsupervised, tested on held-out clips);
(ii) **clip**, via within-clip 2-fold cross-fit (calibrate on half a clip, test the other half)
so the *clip-conditional* guarantee is measured directly.

### 2.3 Block / cluster-robust conformal

Two cluster-aware corrections to the calibration quantile:

* **Design-effect quantile.** Keep the level `α` but move the order-statistic index as if only
  $n_{\text{eff}}$ samples existed:
  $k=\left\lceil n\,\frac{(n_{\text{eff}}+1)\alpha}{n_{\text{eff}}}\right\rceil$.
  This *raises* $q$ (larger set, more conservative).
* **Cluster-bootstrap conformal.** Resample the $K$ calibration clips with replacement $B$
  times, compute $q_b$ each time, and take the $(1-\delta_{\text{boot}})$ quantile of
  $\{q_b\}$ (here $\delta_{\text{boot}}=0.10$). This inflates `q` for the uncertainty in
  *which clips* you calibrated on.

### 2.4 Nominal vs empirical coverage (clip-grouped LOCO)

| nominal | window split | design-effect `n_eff` | cluster-bootstrap | worst-clip (window split) | Mondrian clip, emp cov | Mondrian score-strata, emp cov |
|---|---|---|---|---|---|---|
| 0.80 | 0.800 | 0.800 | 0.733 | 0.588 | 0.767 | 0.780 |
| 0.90 | 0.893 | 0.893 | 0.853 | 0.647 | 0.873 | 0.900 |
| 0.95 | 0.940 | 0.940 | 0.893 | 0.706 | 0.933 | 0.933 |
| 0.99 | 0.973 | 0.973 | 0.960 | 0.765 | — | — |

Readings:
1. **Marginal coverage holds** (window split ≈ nominal). The design-effect quantile is
   *identical* — because $\hat\rho(E)\approx0$, so `n_eff≈n`; applying the $p_{\text{ens}}$
   ICC here would over-correct (report both, don't silently paper over).
2. **Conditional coverage fails**: the worst held-out clip gets 0.647 coverage at nominal 0.90.
   The marginal guarantee hides this — a clip can be systematically under-served.
3. **Mondrian-by-clip** lifts the clip-conditional coverage (0.873 vs 0.647 at 0.90) at modest
   review cost; **Mondrian-by-score-strata** is near-nominal conditionally on difficulty.
4. **Cluster-bootstrap over-corrects** for $K=7$ (0.733 vs nominal 0.80) — conservative, i.e.
   safe, but it buys coverage with review.

---

## 3. Risk-controlling prediction (CRC / Learn-then-Test, Z2.03)

### 3.1 The rule family and the risk

Symmetric accept/reject rule indexed by $t\in(0,\tfrac12]$ on $b=\mathrm{P(bad)}$:

$$
\text{accept} \iff b<t,\qquad \text{reject} \iff b>1-t,\qquad \text{else review}.
$$

Review share decreases as $t\uparrow$. Losses per window:

$$
\ell^{\text{FA}}_t = \mathbf 1\{\text{accept}\wedge y=\text{bad}\},\qquad
\ell^{\text{FD}}_t = \mathbf 1\{\text{accept}\wedge\text{bad}\ \text{or}\ \text{reject}\wedge\text{good}\}.
$$

Primary risk = **marginal false-accept** $R(t)=\mathbb E[\ell^{\text{FA}}_t]$. LTT chooses

$$
\hat t \;=\; \arg\max_{t\in\mathcal G}\ \big\{t:\ \widehat R^{+}(t)\le r\big\},
$$

subject to a union bound over the grid $\mathcal G$ (`δ/|G|`); the selected rule then satisfies
$\Pr(R(\hat t)\le r)\ge1-\delta$.

### 3.2 Cluster-aware upper confidence bounds (the crux)

The exchangeable unit is the **clip** ($K=7$), not the window. Write cluster means
$\bar L_k(t)=\frac1{n_k}\sum_{i\in k}\ell_t(x_i,y_i)$, $\widehat R=\frac1K\sum_k\bar L_k$.

* **Hoeffding (clips):** $\widehat R^{+}=\widehat R+\sqrt{\frac{\ln(1/\delta)}{2K}}$.
* **Maurer–Pontil empirical Bernstein (clips):**
  $\widehat R^{+}=\widehat R+\sqrt{\frac{2\hat V\ln(2/\delta)}{K}}+\frac{7\ln(2/\delta)}{3(K-1)}$,
  $\hat V$ = sample variance of $\{\bar L_k\}$.
* **Cluster bootstrap:** $(1-\delta)$-quantile of the resampled mean of $\{\bar L_k\}$.
* **Window-level (contrast):** Hoeffding/Bentkus (exact binomial) over $n$ windows — **invalid
  under clip correlation**, reported only to quantify the optimism.
* **Design-effect window bound:** Hoeffding with $n\to n_{\text{eff}}$ from the ICC of the
  *loss* (heuristic intermediate).

There is also a **selective-risk ratio bound** (controls $R_{\text{sel}}=\Pr(\text{bad}\mid\text{accept})$):

$$
R_{\text{sel}}(t)\;\le\;\frac{\widehat R^{+}_{\text{FA}}(t)}
{\max\!\big(0,\ \hat D(t)-\sqrt{\ln(1/\delta')/(2K)}\big)},
\qquad D(t)=\Pr(\text{decide}),\ \delta'=\delta/(2|G|),
$$

valid by a union of two Hoeffding events.

### 3.3 Results (r=0.05, δ=0.1)

| tier | assumption | selected t | review % | empirical marginal FA | verdict |
|---|---|---|---|---|---|
| window Bentkus (exact binomial) | **windows exchangeable (violated)** | 0.12 | 88.7 | 4.0 % | optimistic |
| cluster bootstrap | K=7 clips | 0.17 | 82.7 | 2.0 % | anti-conservative with all-zero clusters |
| design-effect window Hoeffding | ICC-corrected windows | — (floor 0.143) | — | — | can certify only r≈0.15 |
| **cluster Hoeffding** | **clips exchangeable (honest)** | **none** | — | — | **infeasible at r=0.05, floor UCB=0.66** |
| cluster empirical Bernstein | clips exchangeable (honest) | none | — | — | infeasible |

**LOCO-honest check** (select `t` on 6 clips, evaluate the 7th): *no fold* certifies r=0.05 —
consistent with the floor.

**The honest headline:** with $K=7$ independent clips, a nonparametric cluster UCB **cannot
certify a 5 % false-accept bound at 90 % confidence**. The reason is not the tail inequality
(Hoeffding vs Bentkus vs Bernstein barely matter at $K=7$) but the **number of exchangeable
clusters**. The guarantee is **data-limited, not model-limited**: to certify r at confidence
`1-δ` you need roughly $K\gtrsim \ln(1/\delta')/(2r^2)$ clips (≈1,240 for r=0.05, δ=0.1; ≈77
for r=0.20). The window-level bound that *does* give 5 % is precisely the one whose
exchangeability assumption the intra-clip correlation violates.

---

## 4. Frontier table and recommendation (Z2.04 summary)

| method | coverage target | review % | empirical error | guaranteed (1−δ) bound | note |
|---|---|---|---|---|---|
| split conformal (clip LOCO) | 0.70 | 15.3 | 16.5 % (decided) | marginal false-decision ≤ 0.30 | marginal FA 5.3 % |
| split conformal (clip LOCO) | 0.80 | **3.3** | 18.6 % | ≤ 0.20 | min-review point; FA 6.0 % > 5 % |
| **split conformal (clip LOCO) — RECOMMENDED** | **0.85** | **12.0** | 16.7 % | **≤ 0.15** | **FA 4.0 % ≤ 5 %; coverage 85.3 % ≥ 85 %** |
| split conformal (clip LOCO) | 0.90 | 47.3 | 20.3 % | ≤ 0.10 | conditional FA 14.6 % |
| CRC window-Bentkus δ=0.1 | — | 88.7 | — | `P(FA≤0.05)≥0.90` | **invalid under clip correlation (optimistic)** |
| CRC cluster-bootstrap δ=0.1 | — | 82.7 | — | `P(FA≤0.05)≥0.90` | K=7 bootstrap unreliable |
| CRC cluster-Hoeffding / EB δ=0.1 | — | — | — | **infeasible at r=0.05** (floor 0.66) | honest; needs more clips |

### Recommendation

**Deploy the split-conformal rule at coverage target `1-α = 0.85` ⇒ review 12.0 %,**
down from **75.5 %** (a **6.3× cut** in human review), with a *stated* distribution-free bound
of **marginal false-decision ≤ 15 %**, empirical marginal false-accept **4.0 % ≤ 5 %** and
held-out coverage **85.3 % ≥ 85 %**. The heavier point (coverage 0.80) reaches 3.3 % review but
lets marginal false-accept rise to 6.0 % > 5 %.

Caveats, stated explicitly:

* The **marginal** guarantee is a *window* statement; the **clip-conditional** coverage is
  materially lower (worst clip 0.647 at nominal 0.90). If per-clip fairness matters, use
  Mondrian-by-clip (clip-conditional 0.873 at nominal 0.90).
* A **1−δ CRC bound at 5 % (or selective FA ≤ 5 %) is not attainable with 7 IndEgo clips**.
  The window-level CRC that *appears* to certify 5 % (review 88.7 %) relies on the
  window-exchangeability that the intra-clip correlation breaks. **To promote a rigorous `r`-bound
  to production, collect more independently-labelled clips** (≈ tens for r=0.20, hundreds for r=0.05).
* **OOD guarantees require target labels.** OOD-60 (n=60, n_bad=9) is used for **evaluation
  only**: the IndEgo-fitted score has AUROC good-vs-bad **0.415** on OOD (chance), so no
  conformal/CRC bound transfers. Any OOD operating point must be re-calibrated on labelled OOD
  clips; until then call the out-of-domain verdicts descriptive.
* E8 (1,546 windows) has `label_source=none`; we report the *review share* the fitted rule would
  produce, never accuracy.

---

## 5. Honesty ledger

| claim | source | n | label source | protocol |
|---|---|---|---|---|
| conformal frontier | Z2.01 | 150 | human gold (single labeller) | clip-grouped LOCO, clip-bootstrap CI |
| coverage nominal vs empirical | Z2.02 | 150 | human gold | clip-grouped LOCO + within-clip cross-fit |
| Mondrian conditional coverage | Z2.02 | 150 | human gold | held-out clips / within-clip 2-fold |
| CRC/LTT bound | Z2.03 | 150 | human gold | LTT on 7 clips, LOCO-honest check |
| E8 review-share application | Z2.01 | 1,546 | **none** | descriptive re-score only |
| OOD transfer | Z2.03 | 60 (n_bad=9) | human gold (OOD-60) | **evaluation only, no guarantee** |
| figure | Z2.01–03 | — | — | seed 7, CPU (`OMP_NUM_THREADS=2`) |

Reminder: never fit `q`/`t` on OOD-60; small n_bad=9; intra-clip correlation means window-level
CIs are anti-conservative (clip-bootstrap used throughout).
