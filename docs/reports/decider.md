# decider (L1) — module report

## What exists

- `jadal/decision.py` — `Decider` class, pipeline-compatible (`Decider(config)`,
  `decide(path, signals) -> Decisions` per SPEC). EmbeddingGemma 2 from
  `/home/parshu/projects/awing/data/eg2` via sentence-transformers, bfloat16
  (never fp16), eval mode, deterministic algorithms on, cudnn.benchmark off,
  fixed seeds. Fixed sampling: 4 frames at fractions (0.125, 0.375, 0.625,
  0.875) via OpenCV → shared preprocessing (512-px thumbnail, byte-identical
  to training); per-frame batch size 1 (never batched — see below);
  image token budget configurable, default 140. Heads: sklearn logistic
  regression on frozen embeddings + single temperature each
  (`jadal.certify.fit_temperature` on CALIBRATION only); serving applies heads
  per frame and averages per-frame LOGITS. Heads saved with
  `save_head` to `models/decider_{hands,work}.npz` (git-ignored), sha256 +
  provenance in `model_versions` (DEV_ONLY heads refused unless
  `JADAL_ALLOW_DEV_HEADS=1`). Frame embeddings cached under `models/.embed_cache*/`
  keyed by sha256 of preprocessed bytes + backbone hash + budget. STEP: nearest-prototype head over the 14
  ECU steps (`models/decider_step_prototypes.npz`), wired into
  `step{choice,probs,answer_conf}` when present, else `choice None`.
- `eval/bench_eg2.py` → `eval/results/bench_eg2.json`.
- `eval/train_heads.py` → hands/work heads (`--splits` full mode or `--dev-n`
  DEV_ONLY mode) and `--step` IndEgo prototype mode → `eval/results/decider_step.json`.
- `tests/test_decision.py` — 11 tests (10 CPU-hermetic with injected
  `embed_fn`/`frame_source`, 1 GPU-gated C1 backbone determinism test behind
  `JADAL_TEST_GPU=1`, cache disabled).

## How to run

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python eval/bench_eg2.py --out eval/results/bench_eg2.json
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python eval/train_heads.py --dev-n 1500 --out eval/results/decider_baseline_DEV_ONLY.json
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python eval/train_heads.py --step --out eval/results/decider_step.json
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python -m unittest tests.test_decision -v
```

## Real test output tails (initial run — SUPERSEDED by Fix round 1 below)

> Initial numbers used synthetic benchmark frames, mean-pooled embeddings,
> and test temperatures fit on test labels. Kept for audit trail; current
> numbers are under "Fix round 1".

Benchmark (RTX 4070 Laptop, bf16, run alone on GPU):

```json
{"go_nogo_140": "GO", "median_s_per_frame_at_140": 0.0676,
 "per_budget": [
  {"budget": 70, "median_s_per_frame": 0.0561, "p95_s_per_frame": 0.0689, "peak_vram_mb_batch1": 1458.5, "batch1_vs_batch5_max_abs_diff": 0.001953125, "batch1_vs_batch5_bit_equal": false},
  {"budget": 140, "median_s_per_frame": 0.0676, "p95_s_per_frame": 0.1107, "peak_vram_mb_batch1": 1476.8, "batch1_vs_batch5_max_abs_diff": 0.0009765625, "batch1_vs_batch5_bit_equal": false},
  {"budget": 280, "median_s_per_frame": 0.0941, "p95_s_per_frame": 0.1028, "peak_vram_mb_batch1": 1514.7, "batch1_vs_batch5_max_abs_diff": 0.001953125, "batch1_vs_batch5_bit_equal": false}]}
```

GO at 140 tokens (0.068 s/frame « 1 s). Key finding: batch-1 vs
batch-of-5 embeddings differ by ~1e-3 (not bit-equal), so the Decider never
batches — every frame is encoded batch-size-1, making segments identical
alone or in a batch of 5.

DEV_ONLY heads (1500 frames, 900/300/300, train/cal/test):

```text
hands_ok: cal_acc=0.8867 cal_ece=0.0517 test_acc=0.9067 test_ece=0.0678 bound=None (no threshold certified)
active_work: cal_acc=0.8667 cal_ece=0.0353 test_acc=0.9 test_ece=0.0332 bound=None (no threshold certified)
```

STEP prototypes (ECU clips, 5 frames/segment, budget 140):

```text
cross-clip acc=0.4000 (N=5) loo acc=1.0000 (N=2) T=0.02
```

Unit tests:

```text
Ran 8 tests in 0.052s — OK (skipped=1)
JADAL_TEST_GPU=1 run of TestDeciderGPU: Ran 1 test in 8.887s — OK
```

Full suite `python -m unittest discover -s tests`: 221 tests, 8 failures +
1 error — all pre-existing in `test_service`/`test_voicebank` (verified
identical on the clean tree via `git stash`; unrelated to this module).

## Fix round 1 (review of PR #20)

All rulings addressed in owned files only. Caveats, stated up front:

- **Labels by gemini-2.5-flash, NOT human-checked.** Every accuracy/ECE
  number below is model-vs-model agreement, not ground-truth performance.
  (Also in `decider_baseline*.json` as `label_caveat` / `label_caveat_long`.)
- **Benchmark frames are real**: first 2 frame_ids per eval parquet file
  (sorted), preprocessing byte-identical to serving; frame ids in
  `bench_eg2.json` (`frames`). No labels used for the benchmark.
- **STEP numbers are not accuracy claims**: cross-clip N=5, LOO N=2.

### What changed

- C1 — real determinism test `test_backbone_determinism_cache_disabled`
  (GPU-gated): cache DISABLED (`use_cache=False`), (a) same segment twice
  identical, (b) s0..s4 in sequence then s0 again identical to solo (dict
  equality after 4-decimal rounding AND sha256 of canonical JSON), (c) 10
  repeated runs identical. Raw embeddings differ ~1e-3 between model batch
  sizes, which is why the Decider never batches frames across segments.
- I1 — one shared preprocessing (`jadal.decision.preprocess_bgr` /
  `preprocess_pil`, 512-px thumbnail) imported by training and serving.
  Serving embeds each sampled frame separately, applies each head per frame
  and averages per-frame LOGITS (documented in the module docstring).
- I2 — cache key = sha256(preprocessed PNG bytes) + backbone weight hash +
  image_token_budget + recipe version; test asserts two budgets never share
  an entry. `model_versions` now also records torch/python versions + dtype.
- I3 — temperature fitted on CALIBRATION only; test split untouched until
  the final report; restated test numbers below (note test_ece moved vs the
  leaked values 0.0678/0.0332 — the leak mattered).
- I4 — caveats in report AND JSON (`label_caveat*`, `frames`/`frames_note`).
- I5 — heads carry `provenance` (`DEV_ONLY` or `full_splits:<manifest sha>`);
  `model_versions.heads_provenance` exposes it; Decider REFUSES DEV_ONLY /
  unknown-legacy heads unless `JADAL_ALLOW_DEV_HEADS=1` (verified live).
- I6 — STEP prototypes built with the same 4-frame sampling/preprocessing/
  pooling as serving; temperature grid widened to
  (0.005..1.0), grid-best was the 0.005 edge so T fell back to 1.0 with the
  note recorded in `decider_step.json`; N stated, no accuracy claimed.

### Real outputs (fix round 1)

Benchmark (real frames, alone on GPU, bf16):

```text
140 tokens: median 0.0623 s/frame, p95 0.0897, VRAM 1476.8 MB -> GO
70: median 0.0570 / p95 0.1262 / 1458.5 MB / batch-diff 0.00269 (not bit-equal)
280: median 0.0929 / p95 0.0943 / 1514.7 MB / batch-diff 0.00195 (not bit-equal)
```

DEV_ONLY heads, T from calibration only (900/300/300):

```text
hands_ok: T=0.6834 cal_acc=0.8867 cal_ece=0.0517 test_acc=0.9067 test_ece=0.0591 bound=None
active_work: T=0.6509 cal_acc=0.8667 cal_ece=0.0353 test_acc=0.9000 test_ece=0.0444 bound=None
```

STEP prototypes (same 4-frame recipe, T=1.0 edge_fallback, note in JSON):

```text
cross-clip 2/5 (N=5), LOO 1/2 (N=2)
```

Tests:

```text
CPU:  Ran 11 tests in 1.930s — OK (skipped=1)
GPU (JADAL_TEST_GPU=1): Ran 11 tests in 39.270s — OK (C1 backbone test passed, cache disabled)
```

## Fix round 2 (full-split retrain + disclosures)

Caveats first: **labels by gemini-2.5-flash, NOT human-checked** — all
accuracy/ECE numbers are model-vs-model agreement. **Granularity: per single
frame** (see N1). STEP numbers are not accuracy claims (N=5 / N=2).

Retrain ran under `flock -w 1800 tools/gpu.lock` (repo-root lock file):

```bash
flock -w 1800 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock env \
  PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python \
  eval/train_heads.py --splits eval/results/splits.json \
  --out eval/results/decider_baseline.json
```

Manifest `eval/results/splits.json` (sha256:a315553177948a74, near-duplicate
components kept whole): train 21001 (7001/7001/6999 per source),
calibration 4501, test 4498. Deterministic sha-order subsample
(salt `decider-subsample-v1`): train 2000/source (6000), calibration
500/source (1500), test 500/source (1500). Heads trained on TRAIN only;
temperature + threshold on CALIBRATION only (`jadal.certify`);
metrics + exact-binomial bound on TEST only.

Tables below were GENERATED from `eval/results/decider_baseline.json`
(no hand-copied numbers):

| head | split | n | acc | ECE | T |
|---|---|---|---|---|---|
| hands_ok | calibration | 1500 | 0.9467 | 0.0100 | 0.6237 |
| hands_ok | test | 1500 | 0.9500 | 0.0097 | 0.6237 |
| active_work | calibration | 1500 | 0.9200 | 0.0121 | 0.7245 |
| active_work | test | 1500 | 0.9153 | 0.0160 | 0.7245 |

| head | cal threshold | test accepted | test errors | error upper bound (1-delta, delta=0.1) |
|---|---|---|---|---|
| hands_ok | None | 0 | 0 | None |
| active_work | None | 0 | 0 | None |

No threshold certified at target_err 0.05 on N=1500 calibration (per-frame
error ~5-8% is at the target boundary, so the Clopper-Pearson upper bound
exceeds it) — bound fields are None, stated as-is.
Heads now carry provenance `full_splits:a315553177948a74` (in
`model_versions.heads_provenance`, replacing DEV_ONLY) and load without
`JADAL_ALLOW_DEV_HEADS=1` (verified).

### N1 — unit mismatch (disclosed)

Heads are trained/calibrated on SINGLE frames but serving averages the
per-frame logits of 4 frames: the accuracy/ECE/bound above are per-frame
numbers. `model_versions` carries `granularity: per single frame` and the
JSON carries `granularity` + `granularity_note`. A test-time multi-frame
check from IndEgo segments is NOT possible (no labels there) — stated
plainly instead of measured badly.

### N2 — still-image vs MP4 domain gap (disclosed + measured)

Training frames are parquet stills; serving frames are cv2 MP4 decodes.
Cheap label-free measurement on the first 100 TEST frames (same frame,
JPEG-decoded vs mp4 round-trip through cv2.VideoWriter mp4v, cosine sim of
the two embeddings — codec shift only, NOT accuracy):

```text
codec_shift n=100: mean=0.9913 min=0.9762 p5=0.9844 p50=0.9923
```

Embeddings are near-identical (mean cos-sim 0.9913), so the codec gap looks
small on this probe — but it is one probe, not a guarantee.

### Tests (fix round 2)

```text
CPU:  Ran 11 tests — OK (skipped=1)
GPU (JADAL_TEST_GPU=1, under flock): Ran 11 tests — OK
```

### Gaps / NOT DONE

- No certified accept threshold yet (see bound table above).
- STEP cross-clip 2/5 (N=5), LOO 1/2 (N=2); step T=1.0 edge fallback.
- No Laya fusion (not required for this baseline).
- Proposals needing other owners: none.
