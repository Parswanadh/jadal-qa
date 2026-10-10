# Baseline v1 — definition of done

`baseline-v1` is a git tag on `main`. It is set ONLY when every item below passes. After the tag, everything new goes behind flags and must not break these checks. Target: 16:00 IST Sat 10 Oct 2026 (fallbacks noted).

| # | Check | How to verify | Status source |
|---|---|---|---|
| B1 | `main` is green, with AND without the real signals module importable, and finishes fast | `timeout 120 python -m unittest discover -s tests` and `-s web`, `node --check web/*.js` | `scripts/baseline_report.py` |
| B2 | Real pipeline on real clips (no stubs): dock IndEgo clips -> segments -> real L0 signals -> real decider -> routed verdicts -> batches -> receipts | `make demo` with `JADAL_USE_STUBS` unset | `scripts/baseline_report.py` |
| B3 | Determinism: 10 repeated runs and alone-vs-batch-of-5 give identical `receipt_sha256`; Re-verify MATCHES on every receipt; a tampered receipt FAILS | demo script assertions | `scripts/baseline_report.py` |
| B4 | Measured numbers exist in `eval/results/BASELINE.json`, each with N and label source: decider heads (accuracy, ECE, certified bound on the clean splits; labels are Gemini's), funnel split on IndEgo segments, per-stage latency and segments/hour on this laptop | `python scripts/baseline_report.py` | `eval/results/BASELINE.json` |
| B5 | The funnel router is the single verdict implementation (fallback if `feat/funnel` is not merged by 15:00: ship the current inline logic and say so) | `grep -n "funnel" jadal/pipeline.py` | review |
| B6 | Dashboard loads against the live service and shows batches, a receipt tree, Re-verify, and the funnel summary | screenshot in `docs/screens/` | reviewer |
| B7 | `/pico/status` answers (hardware optional) | `curl /pico/status` | report |
| B8 | A clean clone runs: README quickstart works; `make demo` works; backup demo video recorded; tag `baseline-v1` pushed | fresh clone test | orchestrator |

Claims allowed in the pitch come ONLY from `eval/results/BASELINE.json` and the files it points to. Anything not measured is roadmap. Known limits: labels for hands/activity come from gemini-2.5-flash (not human-checked); gold labels (if collected) reflect one labeller on a few IndEgo workers; thresholds are placeholders until calibrated; determinism is claimed for the same machine and pinned stack at batch size 1.
