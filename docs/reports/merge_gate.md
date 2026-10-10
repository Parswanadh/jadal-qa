# Merge gate: full-suite status on `pod/base` and `main`

Claim **LB1-gate** · measurement only, **nothing fixed** · verifier worktree `/workspace/wt/b-verify-gate` (branch `b/verify-gate`)
Machine: pod, **CPU only** (no GPU), `/usr/local/bin/python` 3.12.3, pytest 9.1.1, `OMP_NUM_THREADS=2`, RAM ≤ 1.5 GB, every run wrapped in `timeout 1200`.
Raw logs: `eval/results/pod_verify/logs/`. Machine-readable result: `eval/results/pod_verify/merge_gate.json`.

## 1. What the documented full suite is

| Source | Command |
|---|---|
| `Makefile` target `test` (the only suite definition in the repo) | `timeout 150 env PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests` **and** `-s web` |
| `docs/BASELINE.md` row B1 | same (`timeout 120 … discover -s tests` and `-s web`, `node --check web/*.js`) |
| `.github/workflows` | **does not exist** — no CI enforces the gate |
| pytest config (`pytest.ini`/`pyproject.toml`/`setup.cfg`/`tox.ini`) | **none** — pytest is not the documented runner |

`make test` **cannot run on this pod**: the Makefile hardcodes the laptop interpreter `/home/parshu/envs/jadal/bin/python`, which does not exist here (`env: … No such file or directory`, `Error 127`). The documented suite was therefore run with `/usr/local/bin/python`.

## 2. The mission's exact command runs ZERO tests

`OMP_NUM_THREADS=2 timeout 1200 python -m pytest tests/ -q` aborts on **both** trees:

```
!!!!!!!!!!!!!!!!!!!! Interrupted: 4 errors during collection !!!!!!!!!!!!!!!!!!!!
```

pytest stops at collection, so it reports no test results at all. Counts below come from the same command plus `--continue-on-collection-errors` (added only to obtain a verdict; the flag changes nothing about which tests run).

## 3. Counts

| Tree | SHA | collected | passed | failed | skipped | collection errors | unittest `-s tests` | unittest `-s web` |
|---|---|---|---|---|---|---|---|---|
| `pod/base` | `f20013b` | 468 (+4 error modules) | 463 | **4** | 1 | 4 | Ran 463, 0 failed, 4 errors, 1 skipped | Ran 32, OK |
| `origin/main` | `65f37dc` | 496 | 493 | **0** | 3 | 4 | Ran 500, 0 failed, 4 errors, 3 skipped | Ran 32, OK |

90 pytest subtests passed on both trees; wall time ≈ 16 s each. The suite is **fast and does not hang** — it runs to completion.

## 4. Exact failing / erroring tests

**Assertion failures — `pod/base` only (4):**

1. `tests/test_exp_regimes_router.py::test_clear_stationary`
2. `tests/test_exp_regimes_router.py::test_walking_logistics_by_motion`
3. `tests/test_exp_regimes_router.py::test_blurry_lowlight_by_sharpness`
4. `tests/test_exp_regimes_router.py::test_deterministic_and_receipt_fragment`

**Collection errors — identical on both trees (4):** `tests/test_eval_splits.py`, `tests/test_integration.py`, `tests/test_service.py`, `tests/test_signals.py`.

`main` has **no failing tests**. `pod/base`-only failures all live in a file that exists only on `pod/base` (added by `4c418d4`, the laptop-code port); the failure sets otherwise match exactly.

## 5. Classification

**(a) missing optional dependency — environment-specific, present on both trees**

- `imagehash` → `tests/test_signals.py` (via `jadal/signals.py:51`) and `tests/test_eval_splits.py`. Declared in `requirements-edge.txt` (`ImageHash`), **not installed** on the pod; installs are not permitted in this session.
- `fastapi` → `tests/test_service.py` (`from fastapi.testclient import TestClient`) and `tests/test_integration.py`. `requirements-edge.txt` documents fastapi/uvicorn as CENTRAL-tier, not an edge requirement — a pod environment gap, not a repo defect.

**(b) genuine pre-existing failure on `pod/base` — not environment-specific**

All 4 router failures have one root cause: the tests use the **pre-calibration synthetic descriptor scale**, while `jadal/regimes/router.py` is calibrated v0.2.0 on E4.01 data:

- `BASE` has `sharp_norm=300.0` but `THRESHOLDS.sharp_norm_lo=2500.0` → router returns `blurry_lowlight`, test expects `clear_stationary`;
- `test_blurry_lowlight_by_sharpness` uses `sharp_norm=10.0`, outside `TRAINING_RANGES["sharp_norm"]=[150, 55000]` → router returns `default` (UNSEEN_REGIME), test expects `blurry_lowlight`.

This is pure deterministic stdlib Python (no torch, CUDA, or missing packages involved), so it is a real stale-fixture/calibration mismatch on `pod/base`, reproducible anywhere.

**(c) environment-specific**

`make test` unrunnable (hardcoded interpreter path); the exact pytest command aborting at collection; a CUDA-init `UserWarning` in `tests/test_decision.py::TestDeciderDeterminism::test_alone_vs_in_batch_of_5_identical`, expected with no GPU — that test still passes.

## 6. Blind spot in the documented gate (important for the merge gate)

`tests/test_exp_regimes_router.py` contains **9 module-level pytest test functions**; `unittest discover` only collects `TestCase` methods. On `pod/base`:

- pytest: 459 methods + 9 functions = 468 tests, **4 failed**;
- unittest: 459 methods, **0 failed**, 4 import errors → prints `Ran 463 tests … FAILED (errors=4)`.

So the documented `make test` gate **cannot see any of the 4 real failures**. A merged tree carrying them would look "errors-only, no failures" under the repo's own gate.

## 7. Verdict

| Boolean | Value |
|---|---|
| `gate_green_on_main` | **false** (literal exit code: 4 missing-dependency import errors; **0 test failures**) |
| `gate_green_on_main_if_optional_deps_installed` | **true** (inference: no failures to fix) |
| `gate_green_on_pod_base` | **false** (same 4 import errors **plus** 4 genuine failures) |
| `gate_green_on_pod_base_if_optional_deps_installed` | **false** |

**Main is green at test level; `pod/base` is not.** The merge gate as written is RED on both trees, RED purely for environment reasons on `main` and additionally for a real pre-existing defect on `pod/base`. No code was changed, nothing was installed.
