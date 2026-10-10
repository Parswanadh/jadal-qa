# Q3 — Byte-identical receipt proof for the deterministic rule path

**Status: PASS.** On a fixed input set of 180 real windows (150 in-domain gold
+ 30 out-of-domain Egocentric-10K), the deterministic rule scorer emits a
**byte-identical receipt SHA-256 for every window across 10 fresh-process runs**
(one interpreter per run) on the same machine. No dictionary-ordering, clock,
randomness, or GPU dependence was found in the decision path. A **negative
control** (a deliberately perturbed feature or threshold) does change the
receipt, and **missing features fail closed** (`reject:missing_feature`).

- Experiment: `Q3.01` (`experiments/Q3.01/`, `experiments/registry.jsonl`)
- Results (10×N hash matrix): `eval/results/exp_phys/Q3.01.json`
- Code: `eval/exp_phys/` (`det_rule_scorer.py`, `run_determinism.py`,
  `build_input.py`, `extract_ood_feats.py`, `rule_table.json`,
  `input_windows.json`)
- Tests: `tests/test_exp_phys_determinism.py` (20 tests, <1 s, no network)
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
| runs | 10, each in a **fresh process** (`PYTHONHASHSEED` cycling 0/1/42) |
| windows with 1 unique receipt hash | **180 / 180** |
| receipt **bytes** identical across runs | yes |
| aggregate digest of the sorted per-window hashes | `5f31996f49c015d471f80b8510fdba0726a4bcf7f868ad960476b362f9ccc8e4` |
| negative control (feature +1e-4) changes receipt | yes |
| negative control (threshold +1) changes rule-table hash | yes |
| missing features in the frozen input | 0 (fail-closed path not triggered) |

Each of the 10 hashes in `hash_matrix[window_id]` is identical, e.g.:

```
g000 -> 7e491b39eac4acc2c619f989ddc0a3eefb75742f9ac082ec877493444e821f32  (x10)
```

The **aggregate digest** is `sha256` of the concatenation, with **no separator**,
of the 180 per-window `receipt_sha256` values in **sorted `window_id` order**;
it is stored in `Q3.01.json` as `aggregate_digest` with its method.

Decision distribution: 158 `pass`, 22 `reject` (gold 139/11, OOD 19/11).
The rule table constants and scorer source are pinned in every receipt:
`rule_table_sha256 = cf3ab7574de2a2780a1dc4e5a6c8cb774ded71ed75ad267d90a985e2d42026f6`,
`code_sha256 = 02486b4f4f416abc94ec2dd08a1fe65d160d0e21a57d8def01a8bf4875a96cee`.

## Receipt schema

Hashed body (canonical JSON, **integers/strings only**):

```json
{
  "window_id": "g000",
  "decision": "pass",
  "evidence": [
    {"rule": "lowent", "feature": "hist_ent", "value": 74843, "op": "<", "threshold": 64861, "missing": false},
    {"rule": "blur",   "feature": "blur_norm", "value": 1533710, "op": "<", "threshold": 950933, "missing": false},
    {"rule": "ten",    "feature": "ten_norm", "value": 17903885, "op": "<", "threshold": 3392751, "missing": false}
  ],
  "missing_features": [],
  "feature_digest": "sha256(canonical_json({feature: quantised_int}))",
  "rule_table_sha256": "cf3ab757…",
  "code_sha256": "02486b4f…",
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
   `int`. All comparisons (`value_int < thr_int`) are integer-only.
3. **Fail closed on missing features** — an absent / `None` / NaN / inf
   required feature maps to a sentinel (`-99999999`) that can never satisfy a
   tripwire, is listed in the body's `missing_features`, flagged in its
   `evidence` entry (`"missing": true`), and forces
   `decision = "reject:missing_feature"`. A missing feature can **never**
   silently produce `pass`. The frozen 180-window input has **zero** missing
   features (`missing_features_in_input = {}`), asserted for PASS.
4. **Rule thresholds are constants** read from `eval/exp_phys/rule_table.json`
   (frozen from `configs/perpixel_v1.json` `quality_stage.rules`, Q1's rule
   list). The table is hashed into every receipt (`rule_table_sha256`).
5. **Scorer source is pinned** into every receipt (`code_sha256`).

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

- **Gold (150)**: `$JADAL_BUNDLE/caches/perpixel/physics_feats.npz`
  (`hist_ent`, `ten_norm`) + `theory_feats.npz` (`blur_norm`) + `gold.npz`
  (human labels, LOCO clips). Real IndEgo windows. The bundle path is read from
  the `JADAL_BUNDLE` environment variable (no hardcoded path).
- **OOD (30)**: Egocentric-10K `label_windows/*.mp4`, features extracted
  **once** by `eval/exp_phys/extract_ood_feats.py` using the real modules
  (`jadal.perpixel.{series,physics,theory}`) on branch `b/ood-main`. Labels are
  human (`ood160_labels.json`); third-party footage is never committed.

Labels are carried for context only; the scorer never reads them.

## What the "10 runs" are (exactly)

The 10 runs in `hash_matrix` are **10 separate fresh processes** — one new
Python interpreter per run (`run_determinism.py:fresh_process_run`), each with
`OMP_NUM_THREADS=2` and `PYTHONHASHSEED` cycling `0,1,42,0,1,42,0,1,42,0`, each
re-scoring all 180 windows and emitting per-window receipt bytes. All 10×180
receipt hashes and bytes are compared against each other. This is **same-machine
determinism only**; no cross-machine / "any CPU" claim is made (the scorer is
pure Python, so thread count is irrelevant to the result, but no thread-count
sweep is claimed). A single in-process pass is also run, but only to read the
schema invariants and the sample receipt — it is not the proof matrix.

## Checks performed (all true)

- `all_10_receipt_hashes_identical` — 180/180 windows across 10 fresh processes.
- `receipt_bytes_identical_across_runs` — full canonical bytes, not just hash.
- `sorted_keys_immune_to_feature_order` — reversed feature dict → same hash.
- `no_forbidden_fields_no_floats` — recursive scan: no `time/timestamp/
  created_utc/random/nonce/uuid/seed/device/gpu/cuda` keys, no floats.
- `cpu_only_no_gpu_modules` — no `torch`/`cupy`/`tensorflow` imported; `OMP_NUM_THREADS=2`.
- `cross_process_pythonhashseed_deterministic` — the 10 fresh processes with
  `PYTHONHASHSEED` 0/1/42 cycling produce the same per-window hashes.
- `negative_control_feature_perturbation_changes_receipt` — adding one Q=4
  quantum (`+1e-4`) to `hist_ent` on window `g000` changes its receipt SHA-256
  (`e1c3ddb6…` → `3062cabb…`).
- `negative_control_threshold_changes_rule_table` /
  `..._changes_receipt` — incrementing one threshold by 1 changes
  `rule_table_sha256` and the receipt scored with it.
- `no_missing_features_in_input` — the frozen 180-window input has zero missing
  features (so the fail-closed branch is not triggered here; it is unit-tested
  separately).

## Data licence

`eval/exp_phys/input_windows.json` carries 30 OOD rows whose features were
derived from **Egocentric-10K** `label_windows/*.mp4` (third-party, builddotai);
only features (no frames/video) are committed. Labels are human
(`ood160_labels.json`). Allowed in this private repo; **must be excluded from
any public snapshot**.

## Nondeterminism found and fixed

**None survived** — the scorer is a pure function by construction. The design
choices that pre-empt the usual sources:

- *Float formatting / comparison*: removed entirely by Q=4 integer
  quantisation using `decimal` (not binary floats) before any comparison.
- *Set/dict iteration order*: canonical JSON sorts keys; the rule order is the
  explicit list order; 10 fresh-process runs with `PYTHONHASHSEED` 0/1/42 confirm.
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
