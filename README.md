# Jadal QA

Jadal QA is a deterministic quality-assurance add-on for crowd-captured head-camera (egocentric) video. Each uploaded clip is split into segments, scored by exact signals and typed decision models, and checked against a calibrated threshold with an exact-binomial error bound. Each segment gets a verdict, each batch gets a decision, and every decision is written into a content-hashed receipt, re-verifiable by re-running the pinned pipeline on the original clip. Results appear on a dashboard, in pre-recorded voice messages, and on a Raspberry Pi Pico status light. Built for the Awign Hackathon: Physical AI (10-11 October 2026).

The build contract (data schemas, module interfaces, determinism rules) is in [`docs/SPEC.md`](docs/SPEC.md). The full architecture, with the status of every stage, is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The docs index is [`docs/README.md`](docs/README.md).

## Data flow

Status below is from `origin/main` (6b3f196) and the PR list, as of 2026-10-10. Stages marked "stub" run on deterministic stand-ins when `JADAL_USE_STUBS=1` is set.

```
 head camera -> auto-sync -> ingestion bucket          (outside this repo)
                                  |
                                  | upload event (POST /ingest/event, or bucket watcher)
                                  v
  segment (split_clip)               open PR #12   | stub on main
  L0 signals (compute_signals)       open PR #12   | stub on main
  L1 decisions (Decider.decide)      open PR #20   | stub on main
  L2 certify (pick_threshold)        merged PR #14
  verdict per segment                merged PR #14
  receipt (make_receipt)             merged PR #14
  batch rule (reject if rejected_count > k)  merged PR #14
                                  |
                 +----------------+----------------+
                 v                v                v
          dashboard (web/)   voice (voicebank/)   Pico status (pico/)
          merged PR #15      merged PR #11        merged PR #10
```

## Modules

| Module | File(s) | Owner issue | Status |
|---|---|---|---|
| Certify and receipts | `jadal/certify.py`, `jadal/receipts.py`, `tests/test_certify.py`, `tests/test_receipts.py` | #1 | merged, PR #14 |
| Pico firmware and host link | `pico/main.py`, `jadal/pico_link.py`, `tests/test_pico_link.py`, `tests/test_pico_protocol.py` | #2 | merged, PR #10. Verified on hardware: Raspberry Pi Pico (RP2040), MicroPython 1.29.0; heartbeat, LED, FLASH and PING/PONG work; offset RTT about 15 ms. |
| Service, SQLite, batches, dock sim | `jadal/service.py`, `jadal/store.py`, `jadal/pipeline.py`, `jadal/stubs.py`, `sim/dock_sim.py`, `tests/test_service.py` | #3 | merged, PR #13 |
| Voice bank and SOP index | `voicebank/`, `sop/`, `tests/test_voicebank.py`, `tests/test_sop_index.py` | #5 | merged, PR #11; network-free test fix merged, PR #22 |
| Dashboard v1 | `web/` (`index.html`, `app.js`, `style.css`, `mock/`, `test_dashboard.py`) | #4 | merged, PR #15 |
| Signals and segmenter (L0) | `jadal/signals.py`, `jadal/segmenter.py` | #6 | open, PR #12 |
| Decider (L1) | `jadal/decision.py`, `tests/test_decision.py` | #7 | open, PR #20. Not on `main`. |
| Eval harness | `eval/` | #8 | merged, PR #16 |
| Integration: real pipeline, API docs, demo | `docs/API.md`, `Makefile`, `scripts/demo.sh` | #17 | merged, PR #19 |
| Dashboard v2 | `web/` (dropdown filters, receipt tree, step timeline) | #18 | merged, PR #21 |
| Freshness (nonce-LED liveness check) | `jadal/freshness.py`, `tests/test_freshness.py`, `scripts/freshness_demo.py` | #24 | open, PR #25. Live trials not yet run. |
| Funnel (accept/reject/review router) | (see PR) | n/a | open, PR #29 |
| Video hub (EmbeddingGemma 2 search and clusters) | (see PR) | n/a | open, PR #28 |
| Baseline scoreboard (B1-B8) | (see PR) | n/a | open, PR #30 |
| Repo docs: README, docs/ | `README.md`, `docs/` | n/a | merged, PR #23 |
| Laya Design 1 fusion | (none) | #9 | planned. Stretch goal. |

## Repo layout

Matches `main`.

```
.
├── README.md
├── .env.example          template; the real .env is git-ignored
├── docs/
│   ├── SPEC.md           build contract
│   ├── ARCHITECTURE.md
│   ├── ATTRIBUTIONS.md
│   ├── README.md         docs index
│   ├── laya_design1_plan.md
│   ├── sarvam_reference.md
│   ├── reports/          certify, dashboard, pico, service, voicebank
│   ├── screens/          dashboard screenshots
│   └── process/          how the repo was built: workflow, reviews, decision log
├── jadal/                core package: certify, config, pico_link, pipeline,
│                         receipts, service, store, stubs
├── models/               README only; model weights are git-ignored
├── pico/                 Raspberry Pi Pico MicroPython firmware (main.py; RP2040 tested)
├── sim/dock_sim.py       copies clips into the bucket and posts ingest events
├── sop/                  sample SOP (labelled SAMPLE) and deterministic index
├── tests/                unittest files (7)
├── voicebank/            messages.json, gen_voicebank.py; generated wavs git-ignored
└── web/                  static dashboard, mock data, dashboard tests
```

## Quick start

The project uses a Python 3.12 conda env with torch+cu130, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Do not `pip install` into it without checking first.

```
export PYTHONNOUSERSITE=1
PY=<env>/bin/python    # the Python 3.12 conda env described above

$PY -m unittest discover -s tests     # module tests
$PY -m unittest discover -s web       # dashboard tests
JADAL_USE_STUBS=1 $PY -m uvicorn jadal.service:app   # service on stubs
make demo                             # coming with the integration PR (#19 / issue #17)
```

Secrets are read from environment variables (`SARVAM_API_KEY`, `CF_API_TOKEN`, `CF_ACCOUNT_ID`). Put them in a local `.env`, which is git-ignored. Never commit a key.

Test status on this branch (based on `main` 6b3f196): `tests/` runs 254 tests, OK. The dashboard suite (`web/`) runs 19 tests, OK.

## Honest claims

- **Measured vs not.** Nothing here is a measured result until a file under `eval/results/` says so. On `main` there is no such file.
- **Determinism.** Determinism is scoped to: same machine, pinned stack, batch size 1; not across GPUs or versions. Current unit tests cover this for pure functions only. The check that a segment gives the same receipt alone and in a batch of five needs the real pipeline, so it awaits integration.
- **Receipts.** A receipt is content-hashed (SHA-256 of its canonical JSON). It is not signed. Re-verification means re-running the pinned pipeline on the original clip and comparing.
- **Labels.** The labels for Egocentric-10K-Evaluation come from gemini-2.5-flash. They are not human-checked.
- **Thresholds.** The answer threshold is a placeholder (0.9) until it is calibrated on our own clips. No threshold on `main` is certified.
- **Batch rule.** The batch rule is a fixed rule (reject if rejected count is greater than k). It has no statistical guarantee at n = 5 to 10.
- **Face blur.** The blur is a colour-based proxy. It is not a privacy guarantee.
- **SOP and voice.** The SOP is a labelled sample. Voice translations are machine drafts.
- **Pico.** Firmware is verified on hardware on a Raspberry Pi Pico (RP2040) only, with MicroPython 1.29.0. RP2350 is not tested.
- **Not yet run on Awign footage, cameras or criteria.**

Licences and attributions are in [`docs/ATTRIBUTIONS.md`](docs/ATTRIBUTIONS.md).
