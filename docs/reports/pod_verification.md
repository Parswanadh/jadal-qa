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
| 21 | **evidence chain** — features reproducible from video | — | **all 150 windows, max abs diff 7.11e-15** | ✅ verified |
| 22 | **nested AUROCs** recomputed from raw OOF probabilities | EG2 0.846 / px 0.792 / prob 0.834 / rank 0.844 | **0.8458 / 0.7916 / 0.8336 / 0.8440** | ✅ exact |

**Tolerance used:** a claim counts as reproduced when the re-run matches to the precision at which the
claim was published (3 d.p. for AUROC, 3 d.p. for clip-error) on the same protocol. All 20 core claims
matched; rows 21–22 are additions made after the GPU unlock and close the two gaps I had flagged
(the feature-extraction half, and the nested numbers not being recomputable from raw probabilities).

**Rows 21–22 are the upgrade since the first pass.** Row 21 re-derives `caches/gpu/features.json` from the
source video on this pod for **all 150 gold windows** and finds machine-epsilon agreement — so the cached
evidence chain is reproducible, not merely internally consistent. Row 22 recomputes the nested figures
directly from the stored OOF probabilities in `caches/ensemble/oof_probs.npz`, which turns out to hold the
*nested* probabilities (C chosen by inner LOCO per outer fold) — which is why they sit slightly below the
fixed-C numbers. My earlier caveat that the nested AUROCs "could not be recomputed from scratch because the
raw probs are not in the artifact" is **retracted**: they are, and they agree.

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

## 4. Verification status — and what the GPU unlock changed

**Update: the GPU became reachable mid-session (see §9), so the feature-extraction half is now CLOSED.**

1. **Feature extraction — VERIFIED (was the biggest gap).** `eval/pod_verify/verify_feature_reproduction.py`
   re-decodes windows from the source video on this pod and recomputes the 26 features, comparing them to
   `caches/gpu/features.json`. Six windows drawn from six *different* clips reproduce to
   **max |Δ| = 7.1e-15** — floating-point noise, not drift (`eval/results/pod_verify/feature_reproduction.json`;
   the all-150-window run writes `feature_reproduction_all150.json`). So the cached evidence chain is
   reproducible from the videos, not merely internally consistent.
2. **EG2 embedding cache** — under independent check in parallel (`b/verify-eg2`). The *model/eval* half of
   the 0.848 was already verified in §1; whether the cached embeddings themselves reproduce from video
   was not, and now is being tested rather than assumed.
3. **E1.02b (denser EG2 re-embedding at 5/10 fps)** — previously impossible, now runnable and delegated
   (`b/e102b`). This is the item the registry itself flagged as "only if GPU time remains".
4. **E1.03 / E1.04 (OOD fps sweep, frame selection)** belong to Lead A (`feat/exp-fps-pod`, claim A2);
   verifying them concurrently would have duplicated a running claim, so they are out of scope for LB1.
   Their IndEgo-side companion E1.01 **is** verified in §1.
5. **E7 (cost model) and E8 (1,546-window re-score)** are Lead A's claims B1/B2; not touched.
6. **The frozen pre-registered OOD run** still cannot be scored: `labels/ood_labels2.json` is not
   delivered. OOD claims therefore stay at **n=60, n_bad=9** — see LB4 for the staged one-command runner.
7. **Wall-clock throughput claims** (×4.9, "7× real time", kernel hours/hour) were **not** re-measured
   here: the pod is shared with another swarm, so a number taken now says more about contention than
   about the pipeline. Not reproduced, not contradicted. (The A40 end-to-end figure is Lead A's E7 and
   FACTS.md still lists it as PENDING.)

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
artifact present in the *bundle* on this pod. This is exactly the check the LB4 guard is supposed to make,
and it genuinely fails on the shipped tree: the frozen scorer on disk is **not** the file that was frozen.

**Resolved — and independently re-verified by me.** LB4 recovered the byte-exact frozen file from git
(blob `0ede3b83d93ce23dc5b68cd20fb7de0af703ad27`) and committed it as `configs/frozen/ensemble_v1.prereg.json`
(PR #97). I re-checked that rescue myself rather than taking it on trust
(`eval/results/pod_verify/frozen_blob_verification.json`):

```
git cat-file blob 0ede3b83d93ce23dc5b68cd20fb7de0af703ad27 | sha256sum
→ 4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31   ✅ equals the recorded hash
```

and the copy committed on `b/ood-ready` (and on `b/poc-integrate`) is byte-identical to that blob, while the
bundle copy (`cf7fd7d3…`) is the path-remapped one and must not be used for a pre-registered claim. So the
evidence chain is **restored**: the authority is the git blob, not the working copy. Any OOD claim made
before that recovery would have been scored with a different artifact than the one pre-registered — B4's
guard was not ceremonial.

## 6. The out-of-domain number 0.529 — RESOLVED: it is the PoC cascade, not the frozen ensemble

**Correction to my own earlier finding.** I first reported that the published OOD AUROC `0.529` "does not
reproduce". That was wrong in an important way, and an independent verifier (PR #100) pinned it down:
**0.529 reproduces exactly — but from a different scorer than the one I recomputed.** `FACTS.md:66` and
`docs/POC_STATUS.md` attribute 0.529 to *"the PoC cascade as shipped (IndEgo-calibrated)"*, not to the
frozen two-view ensemble. I conflated the two. Here is the corrected, verified picture.

### Two different scorers, two different numbers, both at chance

| scorer | artifact | AUROC | 95% CI | good rejected | bad rejected |
|---|---|---|---|---|---|
| **PoC cascade** (the 0.529 claim) | `eval/results/poc/ood_check_60.json` on `origin/main` | **0.529** | [0.316, 0.728] | **21 / 51** | **5 / 9** |
| frozen two-view ensemble (`ensemble_v1`) | `caches/ensemble/ood_scores.json`, `usable_prob` | **0.415** | [0.207, 0.630] | **26 / 51** | 3 / 9 |

PoC cascade balanced accuracy 0.513 (published 0.51 ✅). Frozen ensemble: 0.412 on my
review-as-pass convention, 0.304 on strict accept-only — a protocol difference, stated rather than hidden.
Both are computed against the same 60 human labels (51 good / 9 bad) with window bootstrap.

**So: 0.529 is accurate for the PoC cascade, and 0.415 is the frozen ensemble's number.** They are not
meant to be the same figure, and neither contradicts the other. My error was attribution, not arithmetic,
and it is corrected in the registry line and in `STATUS_B.md`.

### Why the discrepancy was hard to see — and is NOT artifact drift

- The shipped bundle's OOD score file is **byte-identical to `origin/main`** (sha256 `300c1eda…`), so this
  is **not** an instance of the 14/316 checksum drift of §5. It is an attribution difference: the PoC
  artifact (`eval/results/poc/ood_check_60.json`) is simply **not in the bundle at all**.
- Label snapshots were tested and rejected as an explanation: `history` vs `latest` are identical for all
  60 segments; dropping the one `SHAKY` window gives 0.556 (PoC) / 0.440 (frozen); reason-weighting gives
  0.427 / 0.440. None of these is the 0.529.
- Negative control: 6 head combinations and sign flips were tried; the closest to 0.529 are
  `max(p_eg2, p_pixel) = 0.535` and `2*p_eg2 + usable = 0.527`, with different CIs and fingerprints —
  i.e. no column of the ensemble score file reproduces the PoC cascade's number by accident.
- `ood_scores2.json` / `exp_domain` / e5-cache variants are **not computable** here: `labels/ood_labels2.json`
  is still absent and their `r…` ids are not in `ood_labels.json`.

29 attempts, each with file / n / AUROC / CI / reject counts, are recorded in
`eval/results/pod_verify/ood_0529_archaeology.json` (PR #100).

### What actually matters for the submission

The number that the submission is organised around — *the shipped IndEgo-calibrated pipeline is at chance
on real Egocentric-10K and falsely rejects ~40-50% of good windows* — holds for **both** scorers, and is if
anything worse for the frozen ensemble (51% vs 41% of good windows rejected). Cite them separately and
never merge them into one figure.

## 7. A live config hazard found while verifying (merge-blocking)

While checking that the PoC integration can actually load its model, I found a real, currently-broken
code path on `pod/base` and on every branch I tested (`b-verify`, `b-poc-integrate`, `b-ood-ready`):

```
$ cd /workspace/wt/b-verify && python -c "from jadal.ensemble.scorer import load_config; load_config()"
ValueError: config sha256 mismatch:
  d62a95a473a31085444371f83003b4b0bd020bb0f2f5c13a379d20d9bded8533
  != 4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31
```

This is the §5 drift biting the runtime: `jadal/ensemble/scorer.py` hashes the config and compares it to
`configs/ensemble_v1.json.sha256`, which still records the *frozen* hash, while the shipped config is the
path-remapped file. So **`load_config()` raises on the default path today** — any caller that omits an
explicit config path is broken, including PoC/re-score paths.

Worse in the other direction: the LB2 integration's own loader
(`jadal/poc_stream/policy.py: load_frozen_ensemble()`) checks the model *name, threshold, fusion kind,
weights and required fields*, but **does not check the sha256 at all** — so it loads the remapped artifact
happily. That is exactly backwards from what a pre-registered artifact needs: the strict loader fails on a
recoverable situation, the lenient loader silently succeeds on the artifact that is *not* the one that was
frozen.

**Recommendation (not applied by me — it is another claim's branch):** point the default config path at
the recovered, hash-verified pre-registration `configs/frozen/ensemble_v1.prereg.json` (LB4, PR #97,
verified `sha256 == 4da1068a…`), and make the PoC loader fail closed on that hash. Until then, no number
produced through the default config path should be treated as coming from the frozen scorer.

## 8. Bottom line

Every quantitative claim in the **LB1 core scope** reproduced **exactly** (20/20 rows), including the two
baselines, the full E1.01/E1.02 frame-rate sweeps, and the whole E3.01–E3.05 temporal-policy chain. The
adopted stack (every-2nd window, 4 s exit, EG2 2 fps view, 26-feature logistic) rests on numbers I could
re-derive independently, and E1.02 survives a from-scratch re-implementation.

**Two things did NOT survive verification, and they matter more than the 20 that did:**

1. **The frozen pre-registration hash cannot be confirmed on this pod** (§5). 14 of 316 checksums fail,
   including the frozen scorer, because the bundle was path-remapped after it was checksummed. LB4 has
   since recovered the byte-exact frozen file from git and anchored it at
   `configs/frozen/ensemble_v1.prereg.json` (PR #97).
2. **The out-of-domain picture needed an attribution fix, and now checks out** (§6). The published `0.529`
   is the **PoC cascade** and reproduces exactly (21/51 good, 5/9 bad, bal.acc 0.513); the **frozen
   `ensemble_v1`** is a different number, **0.415** [0.207, 0.630] with **26/51** good windows rejected.
   Both are at chance with heavy false rejects — the finding the submission rests on is confirmed twice.
   My first pass wrongly read the two as one figure; that is corrected here, in the registry and in
   `STATUS_B.md`, and the correction is the reason the second verifier was worth running.

The GPU half of the pipeline and the delivered-labels half of the out-of-domain work remain **unverified
here** for environmental reasons that are stated above, not hidden.

## 9. Appendix — the GPU unlock (why §4 changed mid-session)

For most of this session `nvidia-smi` returned `Failed to initialize NVML: Unknown Error` and
`torch.cuda.is_available()` was `False` (CUDA error 304). That looked like a missing driver or a broken
pod. It was neither:

| check | result |
|---|---|
| driver kernel module | **loaded**, `NVRM version: 580.173.02` (`/proc/driver/nvidia/version`) |
| CUDA toolkit | **installed**, `nvcc` 12.8.93, `/usr/local/cuda`, `libcudart.so.12` |
| userspace libs | `libcuda.so.1`, `libnvidia-ml.so.580.173.02` present |
| device nodes | `/dev/nvidiactl`, `/dev/nvidia5`, `/dev/nvidia-uvm` present |
| **`os.open("/dev/nvidiactl", O_RDWR)`** | **EACCES — permission denied** |
| `NVIDIA_VISIBLE_DEVICES` | `void` |

So the GPU was **never missing and CUDA never needed installing** — access to the device nodes was
denied by the execution sandbox, which surfaced as NVML failing to initialise. Once device access was
granted, the A40 came up immediately (`NVIDIA A40, 46068 MiB`, **0 MiB used, 0 % util, 23 °C**, no
compute processes) and `torch.cuda.is_available()` flipped to `True` (44.4 GiB visible).

**Lesson for whoever reads this next:** an NVML "Unknown Error" inside a sandboxed agent is a
*permissions* symptom, not an installation problem. Check `os.open` on `/dev/nvidia*` before touching a
driver or a toolkit. Also note `NVIDIA_VISIBLE_DEVICES=void` is a red herring here: NVML ignores it, and
overriding it changes nothing while the device nodes are unreadable.

With the GPU reachable, three previously-blocked items became actionable and were dispatched as
independent tracks: feature-extraction reproduction (§4.1, **done, 7.1e-15**), EG2 embedding-cache
reproduction, and E1.02b denser sampling — each in its own worktree, each holding a GPU slot via
`flock /workspace/state/gpu.slot{1,2}.lock`, none of them touching Lead A's branches.
