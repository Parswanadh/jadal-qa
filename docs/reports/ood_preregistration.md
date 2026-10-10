# OOD pre-registration (LB4): frozen ensemble_v1, staged one-command run

Status: **STAGED / PENDING — no OOD claim may be updated until this run completes.**
Owner: lead-b (claim LB4, track B). Written 2026-10-10 ~16:10 UTC.

## What is frozen

`ensemble_v1`: a two-view window scorer, frozen and pre-registered before the
out-of-domain labels were read.

- EG2 view: EmbeddingGemma-2 frame embeddings (every 0.5 s, L2-normalised mean) →
  StandardScaler + logistic head (`C=0.003`).
- pixel view: 26 per-pixel/physics features from 160x90 CPU decode → StandardScaler +
  logistic head (`C=0.1`).
- fusion: equal-weight probability average; `usable_prob = round4(p)`.
- pre-registered decision threshold: **0.5** (`p>0.5` accept, `p<0.4` reject,
  `[0.4, 0.6]` review). In-domain reference: nested LOCO on 150 human IndEgo windows.

The artifact is the config JSON itself: `jadal/ensemble/scorer.py` reads the heads
(`views.eg2|pixel` mean/scale/coef/intercept), `threshold` and `abstain_band` from that
file. There is no other copy of the weights.

## Exact hash and where it is recorded

- **Recorded (authority) sha256:**
  `4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31`
- Recorded in:
  - `jadal_pod_bundle/SHA256SUMS.txt` line 1 → `./caches/ensemble/ensemble_v1.json`
  - `jadal_pod_bundle/caches/ensemble/ood_scores.json` → field `config_sha256`
  - `configs/ensemble_v1.json.sha256` (freeze sidecar)
  - cited truncated in `jadal_pod_bundle/docs/FACTS.md`
- **Recovered byte-exactly from git** (the on-disk copies no longer match):
  blob `0ede3b83d93ce23dc5b68cd20fb7de0af703ad27` = `configs/ensemble_v1.json` at
  freeze commit `b21e3af` ("ensemble-views: freeze ensemble_v1 + numpy scorer + batch
  scorer + OOF probs + head tests"). Its raw sha256 equals the recorded hash.
- Committed here as **`configs/frozen/ensemble_v1.prereg.json`** (never edit it), with
  sidecar `configs/frozen/ensemble_v1.prereg.json.sha256`. Verified:
  `sha256sum configs/frozen/ensemble_v1.prereg.json` = the recorded hash, and
  `jadal.ensemble.scorer.load_config()` accepts it.

### On-disk copies that FAIL verification (recorded, never substituted)

The copies outside `configs/frozen/` were **path-remapped after the freeze**
(`embed_command`, `pixel_command`, `views.eg2.embed_model_dir`), so their bytes differ:

| copy | observed sha256 | matches recorded |
|---|---|---|
| `configs/frozen/ensemble_v1.prereg.json` | `4da1068a78d5d150…` | **yes** |
| `configs/ensemble_v1.json` (repo) | `d62a95a473a31085…` | no |
| `jadal_pod_bundle/caches/ensemble/ensemble_v1.json` | `cf7fd7d34df2c3ca…` | no |

Cross-check: the b-verify pod audit (`/workspace/wt/b-verify/eval/results/pod_verify/bundle_checksum_audit.json`,
2026-10-10T16:02:27Z) reports 302 ok / 14 mismatch / 4 missing overall, and for
`./caches/ensemble/ensemble_v1.json`: recorded `4da1068a…`, computed `cf7fd7d3…`.
The recorded hash stays the authority; the runner records observed vs recorded
separately and **aborts** on mismatch instead of rewriting the hash.

Side effect worth knowing: `configs/ensemble_v1.json.sha256` still contains the frozen
hash while the shipped config is remapped, so `jadal.ensemble.scorer.load_config()`
raises `ValueError: config sha256 mismatch` on the default config path today. The staged
runner therefore materialises its own verified runtime config (below) and does not rely
on the default path.

## Primary metric and threshold — registered IN ADVANCE

Registered before the 100-window labels are read; no tuning, refitting or
re-thresholding is permitted in the scored run.

- **Primary metric: AUROC** of `usable_prob` vs the human label, with a **95% CI by
  source-clip bootstrap** (windows resampled by source clip, 2000 resamples, seed 0).
- Secondary (clearly weaker): window-bootstrap 95% CI, labelled *optimistic* because
  windows from one clip are correlated.
- **Balanced accuracy at the pre-registered threshold 0.5** (accept = `p > 0.5`), plus
  accept/reject counts, good-rejected and bad-rejected counts, and frozen verdict
  counts (accept/reject/review).
- Minimum class requirement: a single-class label file is refused (exit 2); exactly
  **100 labelled windows** are required.
- Label source: human, single labeller (`labels/ood_labels2.json`).
- No claim is made about transfer beyond this run; n_bad may stay small, so CIs are wide.

## One-command invocation

```bash
bash tools/pod/run_ood_frozen.sh
```

The wrapper (`tools/pod/run_ood_frozen.sh`) sets `OMP_NUM_THREADS=2` and execs
`eval/exp_ood/run_ood_frozen.py`, which:

1. Resolves the labels file: `labels/ood_labels2.json` in the repo, else
   `$JADAL_BUNDLE/labels/ood_labels2.json` (`JADAL_BUNDLE` defaults to
   `/workspace/jadal_pod_bundle`); `--labels` overrides.
   If absent → **PENDING** (exit 3), nothing scored, results JSON still written.
2. Requires exactly 100 labelled windows (`--expect-n`); otherwise **ABORT** (exit 2).
3. Requires the frozen artifact raw sha256 to equal the recorded pre-registered hash
   before any scoring; otherwise **ABORT** (exit 2) with recorded + observed hashes.
4. Builds a runtime config = frozen bytes + exactly one allow-listed machine-local
   override (`views.eg2.embed_model_dir` from `configs/ensemble_v1.json`, the pod
   EmbeddingGemma-2 snapshot). Everything else — heads, fusion, threshold, abstain
   band — comes from the frozen artifact, asserted to be the only difference.
5. Scores the 100 windows with `jadal.ensemble.scorer.score(video, start_s, end_s)`
   (per-window cache `eval/results/ood_frozen/ood_scores2.json`, resumable), then
   computes the pre-registered metrics and writes
   `eval/results/ood_frozen/ood_run.json` (`status: COMPLETE`, exit 0).
6. Any guard failure writes the same results JSON with `status: ABORT`/`PENDING` and a
   `guard` field; unexpected failures write `status: ERROR` (exit 1).

Exit codes: `0` COMPLETE, `2` ABORT, `3` PENDING, `1` ERROR.

`eval/results/ood_frozen/ood_run.json` always carries `n`, `n_bad`, `label_source`,
`protocol`, `frozen_sha256`, `preregistered: true`, `status`, plus
`frozen_sha256_observed`, `frozen_sha256_verified`, `artifact_observations`,
`runtime_config_sha256`, `runtime_overrides` for audit.

Guard proof: `tests/test_ood_frozen_guard.py` (11 tests, ~1.5 s, no network, no model):
missing labels → PENDING/no scoring; wrong/tampered hash → ABORT; wrong count → ABORT;
single class → ABORT; scores from another config → ABORT; happy path → COMPLETE with
hand-checkable metrics; real pod-bundle copy → ABORT (skipped if it ever matches again).

## Claims are frozen until the run completes

- **No OOD claim may be updated until this run completes.** The staged
  `eval/results/ood_frozen/ood_run.json` is `status: PENDING` (labels not delivered).
- **OOD headline claims remain n=60 / n_bad=9**: the first batch
  (`jadal_pod_bundle/labels/ood_labels.json`, read-only) holds 60 labelled windows,
  51 good / 9 bad. Counts verified directly from that file.
- The frozen ensemble's OOD score on the first batch is **0.415 AUROC** (window-bootstrap
  95% CI [0.21, 0.647]; n=60, 51 good / 9 bad, no refit) in
  `eval/results/ensemble/ood_heads.json`, flagged there as EXPLORATORY (n_bad < 10
  pre-registered minimum). The re-derivation on the 100-window `ood_labels2.json` is
  **pending** — the staged run produces it.
- The previously quoted first-batch 0.529 [0.32, 0.74] and "21 of 51 good windows
  rejected" belong to the **PoC cascade**, a different scorer (documented in
  `jadal_pod_bundle/docs/FACTS.md` and reproduced by lead B's re-derivation); the two
  numbers must never be attributed to the same model.
- The second batch (`queues/label_queue_random2.json`, 100 windows; videos present at
  `jadal_pod_bundle/data/ego10k_scale/label_windows2/`) changes the denominator only
  when its labels land and the frozen run completes — and the merged/updated claim must
  say so explicitly.
