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

## Amendment (Sat 10 Oct, ~13:45 IST) — Awign's answers: edge-first, autonomous

Awign told us (via the builder, verbally): they record the videos themselves; a decentralised deployment is acceptable **if the algorithm runs on a Raspberry Pi**; their rejection reasons are generic; accept/reject may be fully automated with **no mandatory human in the loop**. Consequences for baseline-v1:

- **baseline-v1 is the EDGE pipeline**: segmenter + L0 signals + `funnel.route` + certify + receipts, REAL modules (no stubs), CPU only, on the IndEgo clips, with 10 repeated runs and alone-vs-batch giving identical `receipt_sha256` and a non-stub backend recorded in `config.versions`. The decider (EmbeddingGemma 2 heads), hub search and dashboard are the optional CENTRAL tier: they enrich and search, they never gate in edge mode, and they merge when their reviews are clean, but they are not blockers.
- **B9 (new): edge feasibility.** `scripts/edge_check.py` proves the edge modules import without torch / sentence-transformers and reports peak RSS; `eval/edge_bench.py` reports segments/hour with the process pinned to 1 and 4 CPU cores. Results measured on this x86 laptop are labelled **"x86 laptop cores as a proxy, NOT measured on a Raspberry Pi"**; a real Pi 4/5 run (same script, `--device-label`) replaces the proxy if hardware is available. We never claim "runs on a Pi" without a Pi run. mediapipe on Pi ARM64 is unverified: the edge path must work with the OpenCV hands fallback and record which backend ran.
- **Cross-architecture determinism is not claimed.** Record platform/arch in `config.versions`; re-verify claims are limited to the same device class.
- **Autonomous mode** (funnel round 3): `mode: autonomous`, no review bucket; accept/reject from calibrated L0 rules only; uncertain => reject (LOW_CONFIDENCE) and counted in the false-reject bound; with no human backstop the two-sided bounds ARE the safety claim. This settles audit item L9 (auto-reject is acceptable to Awign) but not the calibration problem: thresholds remain placeholders until calibrated on in-domain human labels.
- Freshness and voice are roadmap/pilot.
