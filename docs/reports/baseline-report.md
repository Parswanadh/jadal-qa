# Baseline report worker (feat/baseline-report)

One-command scoreboard: `PYTHONNOUSERSITE=1 <env>/bin/python
scripts/baseline_report.py` (exit 0 informational; `--strict` fails on FAIL;
`--clean-clone` runs the B8 fresh-clone quickstart).

## Real output on main @ 6b3f196 (2026-10-10 ~12:33 IST)

| id | status | note |
|----|--------|------|
| B1 | FAIL | 5/6 subchecks green (tests, web, node --check, web w/o signals). `tests` with signals blocked errors in test_integration (dock_sim posts to a dead URL). Voicebank's 32 tests pass on this run — the README "2 fail" note looks stale/environment-dependent. |
| B2 | PENDING | `jadal.signals` / `jadal.decision` not importable on main; no real-pipeline demo possible. |
| B3 | PENDING | no real demo output to parse (follows B2). |
| B4 | PENDING | `decider_baseline.json`, `funnel_calibration.json`, scale/bench latency files all absent. `dataset` numbers (30000 frames, splits, seed 1234, gemini-2.5-flash labels) copied with source/N/label_source. `DEMO_synthetic_*` explicitly excluded (synthetic, not model output). |
| B5 | FAIL | `jadal/pipeline.py` has no mention of funnel (fallback: current inline logic, as BASELINE.md allows). |
| B6 | PENDING | 13 screenshots exist but are the same age as the last `web/` change — re-capture against the live service. |
| B7 | PASS | `/pico/status` -> 200 (`connected:false`, no hardware attached). |
| B8 | FAIL | tag `baseline-v1` absent; clean-clone skipped by default. |

Nothing is claimed beyond what `eval/results/BASELINE.json` contains.
