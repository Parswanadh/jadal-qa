# OOD AUROC 0.529 vs 0.415 — resolved: two different scorers, not one broken number

**Claim id:** LB1-ood · **Branch:** `b/verify-ood` · **Artifact:**
`eval/results/pod_verify/ood_0529_archaeology.json` · **Command:**
`python3 eval/pod_verify/ood_0529_archaeology.py` (29 attempts, numpy-free, ~2 s)

**No published number is changed by this report.** It is a verification finding.

## Verdict

> **0.529 REPRODUCED from FILE `origin/main:eval/results/poc/ood_check_60.json` with
> PROTOCOL "PoC cascade score vs human labels (60 windows, 51 good / 9 bad, window
> bootstrap)" — but that file is a *different scorer* from the frozen ensemble
> `ensemble_v1` shipped in the pod bundle. The shipped
> `caches/ensemble/ood_scores.json` gives AUROC 0.415 [0.211, 0.632] with 26/51 good
> windows rejected. The qualitative finding (at chance, heavy false rejects) is
> CONFIRMED for both scorers; the two numbers are not in conflict, they are two
> different models.**

The premise "FACTS.md claims the *frozen IndEgo cascade* scores 0.529" is a
conflation. FACTS.md:66 (bundle and `b/submission-docs`) says *"the **PoC cascade** as
shipped (IndEgo-calibrated) is at chance: score AUROC 0.529 …"*, and `POC_STATUS.md`
says *"score AUROC (shipped IndEgo-calibrated cascade) 0.529"*. Both attribute 0.529 to
the PoC cascade. The frozen two-view ensemble's own number is 0.415 and was published
separately (`ood_heads.json`, `docs/reports/ensemble_views.md:140`; cited in FACTS on
the laptop at line 78, which the pod bundle does not carry — the bundle FACTS.md is 73
lines and has no 0.415 line).

## Why the claim cannot come from the shipped artifact

Every published sub-number was fingerprinted against both artifacts. All five land on
the PoC cascade file; none lands on the frozen ensemble:

| published sub-number | PoC cascade artifact | frozen `ensemble_v1` artifact |
|---|---|---|
| AUROC 0.529 | **0.529** ✓ | 0.415 ✗ |
| CI 0.32–0.74 | **[0.316, 0.728]** ✓ | [0.211, 0.632] ✗ |
| good rejected 21/51 | **21** ✓ (24 accept / 6 review) | 26 ✗ (14 accept / 11 review) |
| bad rejected 5/9 | **5** ✓ (4 accept / 0 review) | 3 ✗ (3 accept / 3 review) |
| balanced accuracy 0.51 | **0.513** ✓ (accept-only, the PoC script's own convention) | 0.304 (accept-only) / 0.412 (LEAD B's review-tolerant convention) |

LEAD B's balanced accuracy 0.412 uses a different convention than the published 0.51:
`good_ok = (14 accept + 11 review)/51 = 0.490`, `bad_ok = 3/9 = 0.333 → 0.412`; the PoC
script counts only ACCEPT as pass for good windows, which gives 0.304 on the frozen
artifact. Both conventions are recorded in the artifact JSON.

## Candidate artifacts, all recomputed against the 60 human labels

Same estimator as `eval/poc/ood_check.py` (Mann-Whitney AUROC, average ranks for ties;
stratified window bootstrap n=2000, seed 0). Full 29-row table in the JSON.

| # | source file | what it is | n | AUROC | 95% CI | good rej / bad rej | hits 0.529 |
|---|---|---|---|---|---|---|---|
| P1 | `origin/main:eval/results/poc/ood_check_60.json` (`windows[].poc_score`) | PoC cascade score, labels embedded in the artifact | 60 | **0.529** | [0.316, 0.728] | 21/51 · 5/9 | **yes** |
| P2 | same, joined to `labels/ood_labels.json` "latest" | provenance cross-check | 60 | **0.529** | [0.316, 0.728] | 21/51 · 5/9 | **yes** |
| F1 | `jadal_pod_bundle/caches/ensemble/ood_scores.json` → `usable_prob` | **the frozen ensemble the bundle ships** (sha256 `300c1eda…`, config sha `4da1068a…`) | 60 | 0.415 | [0.211, 0.632] | 26/51 · 3/9 | no |
| F2 | same → `p_eg2` | EG2 head | 60 | 0.586 | [0.357, 0.806] | — | no |
| F3 | same → `p_pixel` | per-pixel head (inverted) | 60 | 0.342 | [0.176, 0.516] | — | no |
| F4 | same → `margin` | fusion margin | 60 | 0.415 | [0.211, 0.632] | — | no |
| C1 | `origin/main:eval/results/ensemble/ood_scores.json` | byte-identical to F1 (sha256 `300c1eda…`) | 60 | 0.415 | [0.211, 0.632] | 26/51 · 3/9 | no |
| C2 | `origin/wip/b/triage:.lb5-main-check/eval/results/ensemble/ood_scores.json` | byte-identical to F1 | 60 | 0.415 | [0.211, 0.632] | 26/51 · 3/9 | no |
| C3 | `origin/main:eval/results/ensemble/ood_scores2.json` | second batch, 75 `r…` windows | 0 labelled | n/a | n/a | n/a | **not computable** — no human labels on the pod (`labels/ood_labels2.json` absent), and no `r…` id appears in `ood_labels.json` |
| C4 | `…:eval/results/exp_domain/inputs/ood_scores2.json` (also `/tmp/opencode/e5cache/ood_scores2.json`) | second batch, variant blob (`b471af…`) | 0 labelled | n/a | n/a | n/a | not computable (same reason) |
| C5 | `b/ood-ready:eval/results/ood_frozen/ood_run.json` | staged pre-registered run | — | n/a | n/a | n/a | `status: PENDING`, `guard: labels_absent` — carries no scores |

Sign-flipped columns (0.585 / 0.414 / 0.658 / 0.585) and six head combinations were also
tried as a negative control: the closest frozen-ensemble values to 0.529 are
`max(p_eg2,p_pixel)` = 0.535 [0.303, 0.763] and `2*p_eg2 + usable_prob` = 0.527
[0.298, 0.749] — numerically adjacent but with different CIs, and they match none of the
other published sub-numbers (21/51, 5/9, 0.51). **No column, sign flip or combination of
the shipped artifact reproduces the claim.**

## Label-snapshot hypothesis: rejected

`labels/ood_labels.json` holds `history` (61 rows) and `latest` (60 rows).

* `history` has 61 rows for the same 60 `segment_id`s — one duplicated row, **no verdict
  conflicts**: joining with last-write-wins gives verdicts identical to `latest` for all
  60 segments. AUROC unchanged (0.529 PoC / 0.415 frozen).
* Dropping the single `SHAKY` window (n=59, 51 good / 8 bad): PoC 0.556, frozen
  `usable_prob` 0.440 (reason-weighted 0.427 at SHAKY weight 0.5). No 0.529.
* Restricting "bad" to `NO_ACTIVE_WORK` only: identical to dropping SHAKY.
* Subsets therefore move both scorers, but neither subset produces the published
  fingerprint. The published claim used `latest`, exactly as `ood_check.py` does.

## Drift linkage: the OOD gap shares the cause, it does not share the *symptom*

The bundle's `caches/ensemble/ood_scores.json` is **byte-identical** to the blob on
`origin/main` (sha256 `300c1eda6e8c0b709b36da433b8b91c464e83b89b58af5b3613cc71b02d15e5b`)
and to the `.lb5-main-check` copy on `origin/wip/b/triage`. It did **not** drift, and the
frozen config hash it records (`4da1068a…`) matches the pre-registered frozen config.
So the 0.529-vs-0.415 gap is **not** an instance of the 14/316 `SHA256SUMS.txt` failure
audited in `bundle_checksum_audit.json`: it is a source-attribution difference between
two distinct scorers, one of which (the PoC cascade) is not shipped in the bundle at all.

The practical consequence is the same as the drift story, though: **the pod bundle
contains no artifact that produces the published 0.529** — only `origin/main` carries
`eval/results/poc/ood_check_60.json`.

## Bottom line for the submission

* Frozen ensemble `ensemble_v1`, n=60, 51 good / 9 bad, exploratory (n_bad < 10):
  AUROC **0.415 [0.211, 0.632]**, 26/51 good windows rejected, 3/9 bad rejected.
  Confirmed independently; matches `ood_heads.json` (0.415 [0.21, 0.647]) and LEAD B.
* PoC cascade, same windows: AUROC **0.529 [0.316, 0.728]**, 21/51 good rejected, 5/9 bad
  rejected, balanced accuracy 0.51. Reproduced exactly from
  `origin/main:eval/results/poc/ood_check_60.json`.
* Both are at chance with heavy false rejects. The published number is not optimistic for
  the PoC cascade claim — it is accurate for that scorer; it simply is not the frozen
  ensemble's number. Any sentence that attaches 0.529 to "the frozen ensemble" is wrong
  and should keep the two scorers separate.
* CI note: bootstrap CIs at n=60 are implementation-sensitive (Python `random` vs numpy,
  percentile interpolation): 0.415 gives [0.207, 0.630] (LEAD B), [0.21, 0.647]
  (`ood_heads.json`), [0.211, 0.632] (this harness). All consistent with chance.
