# Independent verification — LEAD B (claim LB1)

**Branch:** `b/verify` (worktree `/workspace/wt/b-verify`, forked from `pod/base` @ `f20013b`)
**Verifier:** lead-b (independent re-run; E1.02 also re-implemented from scratch)
**Date:** 2026-10-10 ~16:05 UTC / 21:35 IST · **Device:** RunPod A40 pod, CPU-only (see "Cannot verify" below)
**Data (READ-ONLY):** `labels/manifest_gold.json` (150 IndEgo windows, human labels, 100 good / 50 bad, 7 clips),
`caches/gpu/features.json`, `caches/frames10k/segments.npz`, `caches/ensemble/{oof_probs.npz,ensemble_v1.json}`
**Protocol:** LOCO over the 150 IndEgo windows / 7 clips; clip-bootstrap 95% CI; no tuning on held-out clips.

Every number below is machine-written into a results JSON under `eval/results/pod_verify/` with
`n`, `label_source`, `protocol` and the regenerating command. Nothing here is copied from a report.

## 1. Verdict summary

| # | Claim (source) | Claimed | Reproduced | Status |
|---|---|---|---|---|
| 1 | 26-feature logistic, raw LOCO (FACTS.md) | 0.810 | **0.810** | ✅ exact |
| 2 | EG2 window-embedding view LOCO (FACTS.md) | 0.848 | **0.8480** [0.734, 0.909] | ✅ exact |
| 3 | probability average of the two views | 0.853 | **0.853** [0.731, 0.922] | ✅ exact |
| 4 | rank average of the two views | 0.856 | **0.856** [0.722, 0.925] | ✅ exact |
| 5 | E1.01 fps sweep 10 fps | 0.809 | **0.8086** [0.693, 0.889] | ✅ exact |
| 6 | E1.01 3.33 fps (34 frames) | 0.807 | **0.8068** [0.697, 0.873] | ✅ exact |
| 7 | E1.01 5 / 2 / 1 fps | 0.793 / 0.787 / 0.774 | **0.7932 / 0.7866 / 0.7736** | ✅ exact |
| 8 | E1.02 EG2 fewer frames 2 / 1 / 0.5 fps | 0.848 / 0.860 / 0.870 | **0.8480 / 0.8596 / 0.8702** | ✅ exact (2 implementations) |
| 9 | E1.02 two-view prob-average | 0.852–0.854 | **0.8524 (2 fps) / 0.8538 (1 fps)** | ✅ exact |
| 10 | E3.01 every-2nd window clip-error | 0.049 [0.019, 0.087] | **0.0486** [0.0189, 0.0872] | ✅ exact |
| 11 | E3.01 every-2nd: clip verdicts / delay @30% | 7/7, 0.3 windows | **agree 1.00 (7/7), delay 0.286** | ✅ exact |
| 12 | E3.01 every-3 / every-4 clip-error | 0.117 / 0.124 | **0.1171 / 0.1242** | ✅ exact |
| 13 | E3.01 skip-first-M is worse | 0.088 → 0.418 | **0.4177** at skip-240 s (agreement falls to 0.71) | ✅ supported |
| 14 | E3.02 no OOD position effect (Mann-Whitney) | p = 0.42 / 0.47 | **p = 0.419 (raw) / 0.468 (normalised)** | ✅ exact |
| 15 | E3.03 early exit, EG2 prefixes | 2 s 0.804 → **4 s 0.860** → 6 s 0.869 → 8 s 0.857 → 10.5 s 0.848 | **0.8038 / 0.8596 / 0.8694 / 0.8574 / 0.8480** | ✅ exact |
| 16 | E3.04 compute: every-2 = ×1.9 | ×1.9 (2.27 h of 4.404 h) | **×1.94 (2.266 h)** | ✅ exact |
| 17 | E3.05 strict-pass policy | every-2nd adopted | `write_e305.py`: `strict-pass: ['every-2']` | ✅ exact |
| 18 | Selective accuracy 80% / 60% coverage (FACTS.md) | 0.831 / 0.889 | **0.831 / 0.889** | ✅ exact |
| 19 | Per-clip weak spots: pallet 0.59, ecu_late 0.68 | 0.59 / 0.68 | **0.59 / 0.68** (prob-avg view) | ✅ exact |
| 20 | pallet / pallet2 at chance for every model | 0.50 / 0.50 | **0.50 / 0.50** (26-feature view) | ✅ exact |

**Tolerance used:** a claim counts as reproduced when the re-run matches to the precision at which the
claim was published (3 d.p. for AUROC, 3 d.p. for clip-error) on the same protocol. All 20 matched.
CI endpoints for E1.02 differ in the last digit between the two implementations because the published
bootstrap draws differ; the point estimates are identical.

## 2. Files (all under `eval/results/pod_verify/`)

| file | what it verifies | command |
|---|---|---|
| `baseline_pixel26_loco.json` | rows 1, 20 | `OMP_NUM_THREADS=2 python3 tools/port/baseline_logistic_loco.py` |
| `baseline_ensemble_views.json` | rows 2, 3, 4, 18, 19 | `OMP_NUM_THREADS=2 python3 tools/port/ensemble_views_exploratory.py` |
| `e1_01_fps_sweep.json` | rows 5–7 | `python3 eval/exp_fps/run_fps.py --mode sweep --shard {0,1}/2` then `merge_e1_01.py` |
| `e1_02_eg2.json` | rows 8, 9 | `python3 eval/pod_verify/verify_e102_eg2.py` (independent re-implementation) + `python3 eval/exp_fps/run_eg2.py` |
| `exp_temporal/E3.01_policies.json` | rows 10–13 | `python3 eval/exp_temporal/run_e3.py --out eval/results/exp_temporal` |
| `exp_temporal/E3.02_ood_position.json` | row 14 | same |
| `exp_temporal/E3.03_early_exit.json` | row 15 | same |
| `exp_temporal/E3.04_compute.json` | row 16 | same |
| `exp_temporal/E3.05_decision.json` | row 17 | `python3 eval/exp_temporal/write_e305.py` |

`eval/pod_verify/verify_e102_eg2.py` is written from scratch against the cached `segments.npz` (not a copy of
the branch runner): same LOCO recipe, own LOCO/bootstrap code, own frame-subsampling loop. It produced
0.8480 / 0.8596 / 0.8702, i.e. the E1.02 claim survives an independent implementation, not just a re-run.

## 3. Independent notes / caveats found while verifying

1. **E1.02 publishes "fewer frames are NOT worse"** — verified, but the ordering is counter-intuitive and
   the CIs overlap heavily (2 fps [0.719, 0.912] vs 0.5 fps [0.756, 0.927]). It is a *non-inferiority*
   statement at n=150, not evidence that 0.5 fps is better. Do not present it as an improvement.
2. **E3.01/E3.04 are label-oracle simulations** on the sparse gold sample (the 150 windows are not
   exhaustive coverage of the 7 clips). The ×1.94/×4.9 speed-ups are arithmetic on sampling fractions,
   not measured wall-clock. The full-coverage model error (OOF err 0.167) dominates the sampling error
   (0.049) — the policy choice is not the accuracy bottleneck.
3. **E3.02's p-values are from n=60 with n_bad=9.** I reproduce them exactly, and I agree with the
   registry's own caveat: this is "thin" and must be re-run when `labels/ood_labels2.json` lands.
4. **E1.01 is a per-window re-decode** (10 fps native, deterministic index subsample). The 3.33 fps arm
   uses 34 of 100 frames for −0.002 AUROC — reproduced, and it is the only fps arm whose drop is within
   the pre-registered 0.02 tolerance at a 3× frame saving.
5. **E3.03 saturates at 4–6 s**, so the adopted "4 s early exit" is at the knee, not past it: 4 s = 0.860
   vs 10.5 s = 0.848. Reproduced, and it is the single largest cost win in the adopted stack (2.5×).

## 4. Could NOT be verified here (explicit)

1. **Anything requiring the GPU.** In this container `nvidia-smi` fails
   (`Failed to initialize NVML: Unknown Error`) and `torch.cuda.is_available()` is `False` (CUDA error 304),
   so **no CUDA job can run from this session**. Consequences:
   - E1.02b (5/10 fps **re-embedding** with EmbeddingGemma-2) is not re-runnable — the cached
     `segments.npz` is the only EG2 source. The fps *subsample* claim (E1.02) is verified; a *denser*
     re-embed is not.
   - Any claim whose number came from a GPU feature extraction that is not cached
     (e.g. re-deriving `caches/gpu/features.json` from video) could only be checked against the cache,
     not regenerated. I verified the *model/eval* half, not the *feature extraction* half.
2. **E1.03 / E1.04 (OOD fps sweep, frame selection)** belong to Lead A (`feat/exp-fps-pod`, claim A2) and
   were still running; verifying them concurrently would have duplicated a running claim, so they are
   out of scope for LB1. Their IndEgo-side companion E1.01 **is** verified above.
3. **E7 (cost model) and E8 (1,546-window re-score)** are Lead A's claims B1/B2; not touched.
4. **The frozen pre-registered OOD run** (sha256 `4da1068a78d5d150…`) cannot be scored: `labels/ood_labels2.json`
   is not delivered. OOD claims therefore stay at **n=60, n_bad=9** (score AUROC 0.529 [0.32, 0.74],
   21/51 good windows rejected) — see LB4 for the staged one-command runner.
5. **Wall-clock throughput claims** (×4.9, "7× real time", kernel hours/hour) were not re-measured: the pod
   is saturated by another swarm and measuring throughput here would produce a number that says more
   about contention than about the pipeline. Not reproduced, not contradicted.

## 5. Artifact integrity — the frozen pre-registration does not verify on this pod

While checking the OOD claim I audited every entry of `/workspace/jadal_pod_bundle/SHA256SUMS.txt`
against the bytes on disk (`eval/results/pod_verify/bundle_checksum_audit.json`):

| result | count |
|---|---|
| raw-file hash matches the recorded hash | **302** |
| mismatch | **14** |
| missing file | 4 |

The mismatches are not random. They include the artifacts the whole evidence chain rests on:

| file | recorded | computed now |
|---|---|---|
| `caches/ensemble/ensemble_v1.json` (the **frozen** scorer) | `4da1068a78d5d150…` | `cf7fd7d34df2c3ca…` |
| `labels/manifest_gold.json` (the gold labels) | `44c7217b839accf0…` | `b1524455f61a4962…` |
| `docs/registry_snapshot.jsonl` | `d8be0322535d1e00…` | `daab7df0ba11bc66…` |
| `queues/{label_queue_random,label_queue_random2,manifest_scale}.json` | — | — |
| `local_work/exp-regimes/…/desc_{indego,ood}.jsonl` | — | — |
| `scripts/{baseline_logistic_loco,ensemble_views_exploratory,dl_small}.py` | — | — |
| `opencode_kit/tools/pod/bootstrap.sh`, `SHA256SUMS.txt` (self-hash, expected) | — | — |

**Cause is established, and it is not tampering: path remapping after the manifest was written.**
`ensemble_v1.json` as shipped contains the string `/workspace/jadal_pod_bundle/data/eg2` — a pod path —
while the bundle was checksummed on the source machine. The files carry mtimes of `14:38:36`, i.e. after
`SHA256SUMS.txt` itself (`14:38:24`), consistent with a remap pass rewriting JSON re-serialisations.
I also tried the canonical-JSON variants that a content-hash scheme would use (sorted-compact,
sorted-default, compact-unsorted) against the recorded hash: **none match**.

**Consequence.** The published pre-registration hash `4da1068a78d5d150…` cannot be confirmed from any
artifact present on this pod. This is exactly the check the LB4 guard is supposed to make, and it now
genuinely fails: the frozen scorer on disk is **not** the file that was frozen. LB4 resolved this the
honest way — it recovered the byte-exact frozen file from git
(blob `0ede3b83d93ce23dc5b68cd20fb7de0af703ad27`), verified `sha256 == 4da1068a…`, and committed it as
`configs/frozen/ensemble_v1.prereg.json` (PR #97). No hash was substituted or invented.
Any OOD claim made before that recovery would have been scored with a different artifact than the one
pre-registered — so B4's guard was not ceremonial.

## 6. The published out-of-domain number does NOT reproduce

`FACTS.md` states: 60 real Egocentric-10K windows (human labels, 51 good / 9 bad), shipped
IndEgo-calibrated cascade at chance — score AUROC **0.529** (0.32-0.74 by window, optimistic), rejected
**21/51** good windows (41%), **5/9** bad, balanced accuracy **0.51**.

Recomputing from the artifacts shipped in this bundle
(`caches/ensemble/ood_scores.json` joined to `labels/ood_labels.json` "latest" — the label set is the same
51 good / 9 bad, 8 `NO_ACTIVE_WORK` + 1 `SHAKY`, and no window's verdict ever changed in `history`):

| score column | AUROC |
|---|---|
| `usable_prob` (the shipped decision score) | **0.415** [0.207, 0.630] |
| `margin` | 0.415 |
| `p_eg2` | 0.586 |
| `p_pixel` | 0.342 |

Per-window verdicts: **reject 29 / review 14 / accept 17**; **26 of 51 good windows rejected (51.0%)**;
**3 of 9 bad rejected**; balanced accuracy at the shipped rule **0.412**.

**So the published 0.529 does not reproduce from any column of the shipped score file, and neither do the
reject counts (26/51 and 3/9 vs the published 21/51 and 5/9).** The same artifact drift documented in §5
is the most likely cause — the score file and the labels were both rewritten after the numbers were
computed — but I cannot reconstruct the file that produced 0.529 from anything present here, and I will
not reverse-engineer a number to match a claim.

**What this does and does not change.** The *qualitative* finding that the whole submission is organised
around — the IndEgo-calibrated cascade is at chance on real Egocentric-10K and rejects roughly half the
good windows — is **confirmed independently, and is in fact worse than published** (0.415 vs 0.529;
51% vs 41% of good windows rejected). The exact published figure should not be quoted as reproduced; it
should either be re-derived from a recovered artifact or restated from this recomputation with its own
provenance.

## 7. Bottom line

Every quantitative claim in the **LB1 core scope** reproduced **exactly** (20/20 rows), including the two
baselines, the full E1.01/E1.02 frame-rate sweeps, and the whole E3.01–E3.05 temporal-policy chain. The
adopted stack (every-2nd window, 4 s exit, EG2 2 fps view, 26-feature logistic) rests on numbers I could
re-derive independently, and E1.02 survives a from-scratch re-implementation.

**Two things did NOT survive verification, and they matter more than the 20 that did:**

1. **The frozen pre-registration hash cannot be confirmed on this pod** (§5). 14 of 316 checksums fail,
   including the frozen scorer, because the bundle was path-remapped after it was checksummed. LB4 has
   since recovered the byte-exact frozen file from git and anchored it at
   `configs/frozen/ensemble_v1.prereg.json` (PR #97).
2. **The headline out-of-domain number does not reproduce** (§6): the shipped artifacts give AUROC
   **0.415** [0.207, 0.630] with **26/51** good windows rejected, not 0.529 with 21/51. The direction of the
   finding is confirmed and is worse than published; the published figure must be re-derived, not quoted.

The GPU half of the pipeline and the delivered-labels half of the out-of-domain work remain **unverified
here** for environmental reasons that are stated above, not hidden.
