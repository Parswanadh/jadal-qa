# Q3 — Byte-identical receipt proof for the deterministic rule path

**Status: PASS.** On a fixed input set of 180 real windows (150 in-domain gold
+ 30 out-of-domain Egocentric-10K), the deterministic rule scorer emits a
**byte-identical receipt SHA-256 for every window across 10 runs**. No
dictionary-ordering, clock, randomness, or GPU dependence was found in the
decision path.

- Experiment: `Q3.01` (`experiments/Q3.01/`, `experiments/registry.jsonl`)
- Results (10×N hash matrix): `eval/results/exp_phys/Q3.01.json`
- Code: `eval/exp_phys/` (`det_rule_scorer.py`, `run_determinism.py`,
  `build_input.py`, `extract_ood_feats.py`, `rule_table.json`,
  `input_windows.json`)
- Tests: `tests/test_exp_phys_determinism.py` (12 tests, <1 s, no network)
- Regenerate: `OMP_NUM_THREADS=2 python3 eval/exp_phys/run_determinism.py`

## Why this exists

The product requirement is *"receipts are SHA-256 of canonical JSON"*. FACTS
lists the real-module end-to-end 10-identical-receipt proof as **PENDING**.
This report closes that gap for the **deterministic rule path** (the frozen
quality tripwires), on real data.

## What is proven (the matrix)

| metric | value |
|---|---|
| windows | 180 (150 gold IndEgo + 30 OOD Egocentric-10K) |
| runs | 10 |
| windows with 1 unique receipt hash | **180 / 180** |
| receipt **bytes** identical across runs | yes |
| cross-process `PYTHONHASHSEED` ∈ {0,1,42} | identical digest |
| aggregate digest of the sorted per-window hashes | `01c8128aa03ac9750062a73ed2fa2471376c1ce98e56057b00122e6695056812` |

Each of the 10 hashes in `hash_matrix[window_id]` is identical, e.g.:

```
g000 -> 7e491b39eac4acc2c619f989ddc0a3eefb75742f9ac082ec877493444e821f32  (x10)
```

Decision distribution: 158 `pass`, 22 `reject` (gold 139/11, OOD 19/11).
The rule table constants and scorer source are pinned in every receipt:
`rule_table_sha256 = cf3ab7574de2a2780a1dc4e5a6c8cb774ded71ed75ad267d90a985e2d42026f6`,
`code_sha256 = b96818f28d699e3db32086ae1c4bb126292ec1a99e465ddbb5783c8388f8a099`.

## Receipt schema

Hashed body (canonical JSON, **integers/strings only**):

```json
{
  "window_id": "g000",
  "decision": "pass",
  "evidence": [
    {"rule": "lowent", "feature": "hist_ent", "value": 74843, "op": "<", "threshold": 64861},
    {"rule": "blur",   "feature": "blur_norm", "value": 1533710, "op": "<", "threshold": 950933},
    {"rule": "ten",    "feature": "ten_norm", "value": 17903885, "op": "<", "threshold": 3392751}
  ],
  "feature_digest": "sha256(canonical_json({feature: quantised_int}))",
  "rule_table_sha256": "cf3ab757…",
  "code_sha256": "b96818f2…",
  "q": 4
}
```

Envelope (the only non-hashed field): `receipt_sha256 =
sha256(canonical_json(body))`. There are **no** time/random/device fields.

## Canonicalisation rules

1. **Canonical JSON** — `json.dumps(obj, sort_keys=True, separators=(",",":"),
   ensure_ascii=False).encode("utf-8")`. Sorting makes the digest immune to
   dict insertion order (tested).
2. **Q=4 integer quantisation, no float in the decision path** — every feature
   is converted with `decimal` `ROUND_HALF_EVEN`: `round(value * 10**4)` →
   `int`. All comparisons (`value_int < thr_int`) are integer-only. Missing /
   NaN / inf features map to a sentinel (`-99999999`) that can never satisfy a
   tripwire.
3. **Rule thresholds are constants** read from `eval/exp_phys/rule_table.json`
   (frozen from `configs/perpixel_v1.json` `quality_stage.rules`, Q1's rule
   list). The table is hashed into every receipt (`rule_table_sha256`).
4. **Scorer source is pinned** into every receipt (`code_sha256`).

## Rule list used

Q1's frozen rule list from `jadal/perpixel/scorer.py`
(`configs/perpixel_v1.json`, branch `b/ood-main` @ `211ef2a`):

| rule | feature | op | threshold | Q=4 int |
|---|---|---|---|---|
| lowent | hist_ent (physics) | `<` | 6.4861 | 64861 |
| blur | blur_norm (theory) | `<` | 95.0933 | 950933 |
| ten | ten_norm (physics) | `<` | 339.2751 | 3392751 |

`jadal/perpixel/scorer.py`'s **Stage-1** block is exactly *"deterministic
rules, integer compare at Q=4"*; the rule constants and their `thr_int` come
from the config verbatim. Stage 2 (conv1d activity head) is deliberately **out
of scope**: this proof is about the deterministic rule path, and the conv head
needs torch weights.

## Input set (fixed, committed)

`eval/exp_phys/input_windows.json` — features only, no frames/video:

- **Gold (150)**: `jadal_pod_bundle/caches/perpixel/physics_feats.npz`
  (`hist_ent`, `ten_norm`) + `theory_feats.npz` (`blur_norm`) + `gold.npz`
  (human labels, LOCO clips). Real IndEgo windows.
- **OOD (30)**: Egocentric-10K `label_windows/*.mp4`, features extracted
  **once** by `eval/exp_phys/extract_ood_feats.py` using the real modules
  (`jadal.perpixel.{series,physics,theory}`) on branch `b/ood-main`. Labels are
  human (`ood160_labels.json`); third-party footage is never committed.

Labels are carried for context only; the scorer never reads them.

## Checks performed (all true)

- `all_10_receipt_hashes_identical` — 180/180 windows.
- `receipt_bytes_identical_across_runs` — full canonical bytes, not just hash.
- `sorted_keys_immune_to_feature_order` — reversed feature dict → same hash.
- `no_forbidden_fields_no_floats` — recursive scan: no `time/timestamp/
  created_utc/random/nonce/uuid/seed/device/gpu/cuda` keys, no floats.
- `cpu_only_no_gpu_modules` — no `torch`/`cupy`/`tensorflow` imported; `OMP_NUM_THREADS=2`.
- `cross_process_pythonhashseed_deterministic` — 3 subprocesses with
  `PYTHONHASHSEED` 0/1/42 produce the same aggregate digest.

## Nondeterminism found and fixed

**None survived** — the scorer is a pure function by construction. The design
choices that pre-empt the usual sources:

- *Float formatting / comparison*: removed entirely by Q=4 integer
  quantisation using `decimal` (not binary floats) before any comparison.
- *Set/dict iteration order*: canonical JSON sorts keys; the rule order is the
  explicit list order; cross-process `PYTHONHASHSEED` test confirms.
- *Uninitialised / time / random state*: the receipt carries no clock, seed,
  nonce or device field (asserted recursively).
- *ffmpeg-MV run-to-run variance*: noted upstream in Q1 (`B_mv_per_frame`
  excluded from digests); that feature is not in this rule path.

## Scope and honest limitations

- This proves the **rule scorer + canonical receipt** is byte-identical. It does
  **not** re-prove the *feature extraction* determinism: the input features are
  frozen once and committed. Upstream extraction (`ffmpeg`/`cv2`) is Q1's
  concern; one-shot extraction took ~103 s for 30 OOD windows.
- The cascade `S0` probe rules (`jadal/poc/cascade.py`: BLACK/TOO_DARK/
  LENS_BLOCKED/OVEREXPOSED/FROZEN) are **not** included because their probe
  features (`spatial_var`, `temp_e`, 2-s probe) are not in the pod/base frozen
  caches; only the scorer's Stage-1 tripwire list was reusable. Stated, not
  hidden.
- OOD set n=30 (of 160 labelled); gold n=150. Both ≥20 as required. Numbers
  here are a determinism proof, not an accuracy claim.
- Device: CPU only (x86_64), Python 3.12.3.
