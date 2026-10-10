# Submission number audit — B1-verify (read-only)

**Claim:** `LB1-audit` · **Auditor:** independent B1 verification pass (LEAD B pod, CPU-only)
**Deliverable:** `eval/results/pod_verify/submission_number_audit.json` (154 numeric claims, machine-readable)
**Scope:** every number in the regenerated submission artifacts, matched against `docs/FACTS.md`, the
union `experiments/registry.jsonl`, and the verified LB1 results. No submission artifact was edited.

## Audited snapshot (pinned)

| Input | Revision | sha256 |
|---|---|---|
| `b-submission-docs:docs/explainer.html` | `f6e6d21` (16:13) | `774e5534dfa705e8…` |
| `b-submission-docs:docs/deck/DECK.md` | `f6e6d21` | `62b7b910fbd8715a…` |
| `b-submission-docs:docs/POC_STATUS.md` | `f6e6d21` | `5f487970d5a46d3a…` |
| `b-submission-docs:docs/SUBMISSION.md` | `f6e6d21` | `c04f2d83c3940f15…` |
| union `experiments/registry.jsonl` (`b/verify`) | `0e4f7de` (16:18) | `b828dd093122718f…` (36 ids) |
| `docs/FACTS.md` (pod bundle) | — | `77c7331b5e9ca8e8…` |
| LB1 `eval/results/pod_verify/ood_and_nested.json` | `0e4f7de` | `be5405e9ba1a6895…` |

Method: full manual read of all four artifacts plus a regex numeric sweep of every numeric token;
each hit was classified and matched to its source. The union registry is a live merge target
(33 lines at 16:00, 36 at 16:18) — counts below are as of the pinned revision.

## Counts per status

| Artifact | ok | contradicts | pending-as-result | unsourced | total |
|---|---:|---:|---:|---:|---:|
| `docs/explainer.html` | 49 | 3 | 2 | 0 | 54 |
| `docs/deck/DECK.md` | 31 | 3 | 2 | 0 | 36 |
| `docs/POC_STATUS.md` | 30 | 3 | 5 | 0 | 38 |
| `docs/SUBMISSION.md` | 22 | 3 | 1 | 0 | 26 |
| **Total** | **132** | **12** | **10** | **0** | **154** |

## Findings (worst first)

### 1 · CRITICAL — the LB1 verification notes in all four artifacts are superseded (4 × `contradicts`)

The notes added at 16:13 (commit `f6e6d21`) say the published `0.529` **“does not reproduce”** and present
`0.415` as the recomputation. **LB1 corrected itself at 16:18:29 (commit `0e4f7de`)** — the first pass
conflated two different scorers:

| scorer | artifact | AUROC | CI | good rejected | bad rejected | bal. acc |
|---|---|---|---|---|---|---|
| **PoC cascade** (the 0.529 claim) | `origin/main:eval/results/poc/ood_check_60.json` | **0.529** | [0.316, 0.728] | **21 / 51** | **5 / 9** | 0.513 |
| frozen two-view `ensemble_v1` | `caches/ensemble/ood_scores.json` | **0.415** | [0.207, 0.630] | 26 / 51 | 3 / 9 | 0.412 |

So **0.529 reproduces exactly** (all five sub-numbers match the published values) and belongs to the PoC
cascade, which is what the artifacts already say in their headline. `0.415` is the frozen ensemble's own
number, not a recomputation of `0.529`. LB1's decision: *“Quote the two OOD numbers separately and never
merge them.”* The individual numbers inside the notes (`0.415 [0.207,0.630]`, `26/51`, `3/9`, `0.412`,
`p_eg2 0.586`, `p_pixel 0.342`) are correct **for the frozen ensemble** and sourced from the LB1 registry
summary — the framing is what is wrong.

Impact: as written, the submission pack tells reviewers that its own headline OOD number is
unreproducible, which is the opposite of the verified state. Both scorers are at chance with heavy false
rejects, so the qualitative finding the pack rests on is confirmed twice — that part is fine.

### 2 · Registry statistics are stale and divergent (6 × `contradicts`)

The pack reports **34 entries / 24 done / 10 running**, verdicts **11 adopt / 8 drop / 6 maybe /
1 inconclusive / 8 none**. The union registry of record at audit time (`b/verify @ 0e4f7de`, 36 ids) has
**28 done / 7 running / 1 pending**, verdicts **10 adopt / 9 drop / 8 maybe / 2 inconclusive / 7 none**.
The pack's figures match its branch-local 34-line copy (`b-submission-docs:experiments/registry.jsonl`),
which has diverged from the union: `E1.03`, `E1.04`, `E4.01`–`E4.04` are closed there but running in the
union, and `LB4`/`LB5` are absent there. Re-render the stats from the union registry at freeze time.

### 3 · Four experiments presented as results while the registry of record says running (10 × `pending-as-result`)

| Item | Where in the pack | Union registry | FACTS §20:00 |
|---|---|---|---|
| `E1.04` frame-selection numbers (`0.787 [0.650,0.866]`, `0.762`, `0.780`, `0.739`) | explainer §4, deck negatives, POC §5 | running, no summary | unfinished |
| `E4.03` frozen-ensemble OOD `0.415` | explainer §4, POC §3, SUBMISSION §2 | running, no summary | unfinished |
| `E1.03` frozen logistic `0.36–0.55` across fps | POC §3, SUBMISSION §2 | running, no summary | unfinished |
| `E4.02` “no regime split” | POC §5 | running, no summary | unfinished |

The underlying result JSONs do exist on worker branches (`exp-fps-pod`, `exp-regimes-pod`) and in the
branch-local registry, so the fix is to close/merge those entries into the union registry (or mark the
claims in-progress in the pack). FACTS.md explicitly lists `E1.03`, `E1.04`, `E4.01`–`E4.04` as
*“Unfinished (workers died when opencode auto-updated v1 -> v2)”*.

### 4 · “OOD score is pending” is stale (2 × `contradicts`)

The explainer §2 and deck slide 4 still say the frozen ensemble's out-of-domain score is *pending*, while
the same documents quote `0.415` for it. The frozen `ensemble_v1` OOD score **has** been computed
(`0.415 [0.207,0.630]`, exploratory, `E4.03`/LB1). What remains pending is the confirmatory
re-derivation from the recovered pre-registered artifact (`configs/frozen/ensemble_v1.prereg.json`,
PR #97 / `LB4`, staged `PENDING` until labels land).

### 5 · What is clean (132/154 ok, 0 unsourced)

- **No unsourced numbers** — every other number matches FACTS.md, the union registry, or the LB1 results.
- **No OOD transfer/accuracy claim anywhere**; the pack explicitly says the shipped scorer does not
  transfer and lists “transfer” under *never claimed*.
- **All PENDING items are correctly shown as in-progress**, never as results: Raspberry Pi,
  Egocentric-10K accuracy, hyperspectral, nonce-LED freshness, voice, end-to-end GPU number, certified
  error bounds, end-to-end determinism proof, spectral arm B re-run, duplicate detection on real labels,
  `ood_labels2.json` (`E5.04`, `labels_missing=true`, n=0).
- **The ~40 % Egocentric-10K reject share is framed as mostly false rejects / “not a quality measure”**
  in all four artifacts — no pre-OOD-failure framing survives.
- In-domain numbers (0.810 / 0.848 / 0.853, nested 0.846 / 0.792 / 0.834 / 0.844, gates not met,
  weakest clip 0.606, selective 0.74 → 0.84), the E1.01/E1.02 fps sweeps, E3.01–E3.05 temporal policy,
  decode/GPU/hardware numbers and the 1,546-window PoC run all match their sources (LB1 reproduced the
  in-domain set 20/20 exactly).

### 6 · Minor notes (recorded, not counted as failures)

- **“20/20 published quantitative claims reproduced exactly”** matches the LB1 registry literally, but
  LB1 also reports **TWO FINDINGS FAILED VERIFICATION** (14/316 `SHA256SUMS.txt` mismatches incl. the
  frozen scorer config; plus the OOD attribution correction) and flags the GPU feature-extraction half
  and the delivered OOD-labels half as unverified. Add that clause where 20/20 appears.
- The frozen artifact sha256 `4da1068a…` is recoverable byte-exactly from git blob
  `0ede3b83d93ce23dc5b68cd20fb7de0af703ad27` (LB1, independent of LB4). Live hazard: `load_config()`
  raises on the default config path while the PoC loader skips the sha256 check.
- FACTS.md is internally inconsistent on power (80 W firmware limit vs “power cap reads 124 W”); the
  artifacts quote 124 W faithfully.
- The selective-accuracy curve is not attributed to a view in the pack; FACTS attributes it to the
  26-feature + logistic view (LB1: 0.74 full coverage). LB1's two-view ensemble curve is different
  (0.79 full / 0.831 at 80 %).

## Recommended fixes before freeze

1. Rewrite the four verification notes: **0.529 = PoC cascade (reproduces exactly); 0.415 = frozen
   ensemble_v1 (exploratory); two scorers, never merged.** Keep the confirmed qualitative finding.
2. Re-render the registry statistics from the union registry, and close or in-progress the
   `E1.03`/`E1.04`/`E4.0x` claims in the pack.
3. Add the LB1 caveat clause to the 20/20 line (two failed findings; GPU/OOD-labels half unverified).
4. Optional: name the selective-accuracy view; reconcile the 80 W/124 W FACTS conflict.

*Read-only audit; no submission artifact was modified. Full per-claim records:
`eval/results/pod_verify/submission_number_audit.json`.*
