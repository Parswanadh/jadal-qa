# BASELINE scoreboard

Generated (UTC): 2026-10-10T07:03:03.457788+00:00
Git SHA: 6b3f1965a3632c35f82b4ba5eb33b2fab8be0d2c

| id | check | status | evidence | command output tail |
|---|---|---|---|---|
| B1 | main green with AND without real signals, fast; node --check | FAIL | 1/6 subchecks failed: suite tests (signals unimportable) | [suite tests (signals unimportable)]  <br> FAILED (errors=1) <br> docking 2 clips from /tmp/tmp4i9sbsdq/src -> /tmp/tmp4i9sbsdq/bucket (delay 0.0s), posting to http://127.0.0.1:9 <br> copied a.mp4 -> /tmp/tmp4i9sbsdq/bucket/a.mp4 <br>   ingest failed for a.mp4: <urlopen error [Errno 111] Connection refused> <br> copied b.mp4 -> /tmp/tmp4i9sbsdq/bucket/b.mp4 |
| B2 | real pipeline on real clips via make demo (no stubs) | PENDING | real jadal.signals and/or jadal.decision not importable on main; demo stays on stubs (probe: Traceback (most recent call last): <br>   File "<string>", line 1, in <module> <br> ModuleNotFoundError: No module named 'jadal.signals') | - |
| B3 | determinism: identical receipt_sha256; Re-verify MATCH | PENDING | no real-pipeline demo output (B2 PENDING), nothing to parse | - |
| B4 | measured numbers in BASELINE.json with N and label source | PENDING | missing measured artifacts: eval/results/decider_baseline.json (decider accuracy/ECE/certified bound); eval/results/funnel_calibration.json (funnel split on IndEgo segments); scale bench / bench_eg2.json (per-stage latency, segments/hour) | - |
| B5 | funnel router is the verdict implementation | FAIL | jadal/pipeline.py has no mention of funnel: the funnel router is not the verdict implementation (fallback: current inline logic) | - |
| B6 | dashboard screenshots vs live service newer than web/ | PENDING | screenshots (13 png, last 2026-10-10T12:08:30+05:30) are NOT newer than last web/ change (2026-10-10T12:08:30+05:30): re-capture against the live service | - |
| B7 | /pico/status answers (hardware optional) | PASS | GET /pico/status -> 200: {"connected":false,"port":"/dev/ttyACM0","last_hb_age_s":null,"last_hb_ticks_ms":null,"last_hb_led":null} | - |
| B8 | clean clone runs quickstart; tag baseline-v1 pushed | FAIL | git tag baseline-v1 absent on this repo; clean-clone check skipped (rerun with --clean-clone) | - |

Statuses: PASS / FAIL / PENDING / NOT-APPLICABLE. Missing artifacts are PENDING, never estimated.
Numbers live in eval/results/BASELINE.json with per-number source file, N and label_source.
