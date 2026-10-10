# decider (L1) — module report

## What exists

- `jadal/decision.py` — `Decider` class, pipeline-compatible (`Decider(config)`,
  `decide(path, signals) -> Decisions` per SPEC). EmbeddingGemma 2 from
  `/home/parshu/projects/awing/data/eg2` via sentence-transformers, bfloat16
  (never fp16), eval mode, deterministic algorithms on, cudnn.benchmark off,
  fixed seeds. Fixed sampling: 4 frames at fractions (0.125, 0.375, 0.625,
  0.875) via OpenCV → PIL; per-frame batch size 1 (never batched — see below);
  image token budget configurable, default 140. Heads: sklearn logistic
  regression on frozen mean-pooled segment embeddings + single temperature
  each (`jadal.certify.fit_temperature`); heads saved with
  `save_head` to `models/decider_{hands,work}.npz` (git-ignored), sha256 in
  `model_versions`. Frame embeddings cached under `models/.embed_cache*/`
  keyed by sha256 of frame bytes. STEP: nearest-prototype head over the 14
  ECU steps (`models/decider_step_prototypes.npz`), wired into
  `step{choice,probs,answer_conf}` when present, else `choice None`.
- `eval/bench_eg2.py` → `eval/results/bench_eg2.json`.
- `eval/train_heads.py` → hands/work heads (`--splits` full mode or `--dev-n`
  DEV_ONLY mode) and `--step` IndEgo prototype mode → `eval/results/decider_step.json`.
- `tests/test_decision.py` — 8 tests (7 CPU-hermetic with injected `embed_fn`,
  1 GPU-gated behind `JADAL_TEST_GPU=1`).

## How to run

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python eval/bench_eg2.py --out eval/results/bench_eg2.json
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python eval/train_heads.py --dev-n 1500 --out eval/results/decider_baseline_DEV_ONLY.json
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python eval/train_heads.py --step --out eval/results/decider_step.json
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python -m unittest tests.test_decision -v
```

## Real test output tails

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

## Gaps / NOT DONE

- Real results (`eval/results/decider_baseline.json`) still need the FIXED
  eval manifests from `feat/evalharness` (near-dup leakage fix not merged at
  time of writing); DEV_ONLY numbers are convenience-subset agreement only.
- No threshold certified on the dev calibration split (N=300 too small at
  target_err 0.05) — expect this to change with the full calibration split.
- STEP cross-clip is 2/5 (N=5); 13/14 train steps are singletons so LOO is
  N=2. More IndEgo ECU clips (same 14-step vocabulary) would fix this.
- No Laya fusion (not required for this baseline).
- Proposals needing other owners: none (no changes to `jadal/config.py`,
  `jadal/__init__.py`, or `docs/SPEC.md` were needed).
