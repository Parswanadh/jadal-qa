# Jadal QA

Jadal QA is a quality-assurance add-on for crowd-captured head-camera (egocentric) video. For each clip window it decides whether the footage is usable (dark, blocked, frozen, duplicate, hands not visible, no active work), and it writes each decision with its evidence into a content-hashed receipt. Built for the Awign Hackathon: Physical AI (10-11 October 2026).

This page states what is built, what is measured, and what is not yet shown. It is current as of 2026-10-10 and describes `origin/main` at `55b0654`. This branch contains no measured accuracy, speed or throughput numbers. Measured results will be cited from files in the repo once they are merged.

The build contract is [`docs/SPEC.md`](docs/SPEC.md), which is binding. The architecture and the status of each stage is described in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), which is descriptive only. The docs index is [`docs/README.md`](docs/README.md).

## Direction (2026-10-10)

Planned direction, pending a `docs/SPEC.md` amendment. Where this direction and `docs/SPEC.md` disagree, `SPEC.md` governs.

- **Edge-first.** The target is analysis on the capture device, with no mandatory human in the loop. Running on a Raspberry Pi is **not yet measured**. Any speed figures so far are from x86 laptop cores, as a proxy. A real Pi run is pending.
- **Whole-video, per-pixel analysis of every frame.** No sampled frames. Streaming analysis in chunks while the worker records is planned; the chunk size is not measured or backed by a file on this branch. Early rejection of dark, blocked or frozen footage is planned; its timing is not measured.
- **Deterministic, mathematics-first decisions.** The inputs are measured signals: noise level, sharpness, exposure, frame difference against the noise floor, residual motion after camera-motion removal, and rhythm tests. The intended decision is a transparent rule list with exact thresholds, and every reject carries its evidence (value, comparison, threshold). Logistic regression and boosted trees are comparison baselines only. The first threshold rule list rejected nearly everything and was dropped (unverified builder note, no backing file in this repo), so the rule-list form is not yet working.
- **Not in the pipeline.** A generic vision-language model was tried and stopped as too probabilistic. Laya (text-only) and Clef (hosted) are not in the pipeline.
- **EmbeddingGemma 2 is an optional helper**, used for hub search, de-duplication and frame heads. It is not the core.
- **RGB, not hyperspectral.** The cameras record 3 bands (RGB). We do per-pixel frequency and temporal-spectral analysis of RGB frames. This is **not** hyperspectral imaging. Recovering a spectrum from RGB is under-determined, and a hyperspectral sensor would be needed for real spectral data.

## Status

- **Merged to `main`, with its own report:** the spectral-features arm (PR #52). Its measured results are in [`docs/reports/spectral_features.md`](docs/reports/spectral_features.md). Cite that file for numbers.
- **Not merged, no results cited here:** the per-pixel scorer, GPU pipeline and related work are in open PRs #57, #61 and #67-#73. Their measured results are not on this branch; measured results are in those PRs (not yet merged), see `docs/reports/` after they merge.
- **Not measured:** Raspberry Pi runs, end-to-end GPU throughput as one number, and accuracy on Egocentric-10K (which has no human labels).
- **Pending:** a custom hand-region residual-motion rhythm network (result pending; a negative result is acceptable), cross-device determinism of the GPU path, and a batch-independence proof through the real modules.
- **Not run:** freshness nonce-LED live trials.
- **Not done:** hyperspectral imaging (see above).

## What this means

- **Accuracy is not yet shown.** No accuracy number is claimed on this branch.
- **The human set is small.** 150 windows: 100 usable, 50 bad (NO_ACTIVE_WORK 23, HANDS_NOT_VISIBLE 14, DUPLICATE 6, TOO_DARK 6, UNDECODABLE 1) (see `docs/reports/funnel.md:421-422`; `eval/results/gold_labels_snapshot.json`). It comes from 7 IndEgo videos (Fraunhofer IPK), 480p at about 10 fps (see `docs/reports/edge.md:69`). The file names in `eval/gold/segments.json` suggest only a few wearers. It is not Awign footage.
- **Egocentric-10K has no human labels.** Its frame labels come from an AI model. A proof-of-concept run of the pipeline on that footage is descriptive only: its verdicts are not accuracy results. No measurement on this branch shows how the scorer transfers to that footage, so the domain shift is untested.
- **Speed is not measured as one number.** Decode is the expected bottleneck, and no end-to-end throughput figure is cited on this branch.

## What is and is not claimed

- **No certified bounds.** The certification machinery exists in `jadal/certify.py`: `exact_binomial_upper` (Clopper-Pearson), `pick_threshold` (fixed-sequence testing) and `batch_rule`. It does not give a usable guarantee on the current data. With 150 labels from 7 clips, a multiple-testing correction needs about 86 clean windows (see `docs/reports/funnel.md`: support floor 86). Windows within a clip are correlated, so the usable independent sample is smaller than 150. Error rates are therefore reported as observed, with n and label source, and not as certified bounds. A certified bound needs more labelled, independent clips than we have.
- **Receipts are content-hashed, not signed.** A receipt is the SHA-256 of its canonical JSON. Re-verifying means re-running the pinned pipeline on the original clip and comparing hashes. A tampered receipt fails. An optional HMAC sidecar exists (see [`docs/reports/verify.md`](docs/reports/verify.md)), but no receipt is signed.
- **Determinism is narrow.** The CPU feature path's canonical-JSON digest was identical across repeated runs and across 1-core vs 4-core runs (see `docs/reports/edge.md:81-83`). A repeated-run determinism check exists in `scripts/demo.sh`, and the batch-independence proof through the real modules is pending. GPU numerics are not claimed identical across devices, and cross-architecture determinism is not claimed.
- **Raspberry Pi.** A Raspberry Pi Pico (RP2040) drives verdict LEDs, and the Pico firmware is verified on that hardware (MicroPython 1.29.0, `docs/reports/pico_hardware.md`). This is a status light, not the compute device. RP2350 is not tested. Running the analysis on a Pi is not measured.
- **Labels.** Egocentric-10K-Evaluation labels are AI-generated (gemini-2.5-flash, per [`docs/ATTRIBUTIONS.md`](docs/ATTRIBUTIONS.md)) and not human-checked. The 150 human gold labels come from one labeller.
- **Thresholds.** The answer threshold on `main` is a placeholder (`STUB_THRESHOLD = 0.9` in `jadal/stubs.py`). No threshold on `main` is certified.
- **Batch rule.** The batch rule is a fixed rule (reject if the rejected count is greater than k). It has no statistical guarantee at small batch sizes.
- **Face blur** is a colour-based proxy. It is not a privacy guarantee.
- **SOP and voice.** The SOP is a labelled sample. Voice messages are pre-recorded machine-draft translations.
- **Not yet run on Awign footage, cameras or criteria.**

## Data flow (status on `origin/main`, 55b0654)

```
 head camera -> auto-sync -> ingestion bucket          (outside this repo)
                                  |
                                  | upload event (POST /ingest/event, or bucket watcher)
                                  v
  segment (split_clip)               merged PR #12
  L0 signals (compute_signals)       merged PR #12
  L1 decisions (Decider.decide)      merged PR #20 (optional EmbeddingGemma 2 heads)
  L2 certify (pick_threshold)        merged PR #14
  verdict per segment                merged PR #14
  receipt (make_receipt)             merged PR #14 (content hash, not signed)
  batch rule (reject if rejected_count > k)  merged PR #14
                                  |
                 +----------------+----------------+
                 v                v                v
          dashboard (web/)   voice (voicebank/)   Pico status (pico/)
          merged PR #15      merged PR #11        merged PR #10
```

The whole-video, streaming direction above is not yet on `main`. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Modules

| Module | File(s) | Owner issue | Status |
|---|---|---|---|
| Certify and receipts | `jadal/certify.py`, `jadal/receipts.py`, `tests/test_certify.py`, `tests/test_receipts.py` | #1 | merged, PR #14 |
| Pico firmware and host link | `pico/main.py`, `jadal/pico_link.py`, `tests/test_pico_link.py`, `tests/test_pico_protocol.py` | #2 | merged, PR #10. Verified on a Raspberry Pi Pico (RP2040), MicroPython 1.29.0. |
| Service, SQLite, batches, dock sim | `jadal/service.py`, `jadal/store.py`, `jadal/pipeline.py`, `jadal/stubs.py`, `sim/dock_sim.py`, `tests/test_service.py` | #3 | merged, PR #13 |
| Voice bank and SOP index | `voicebank/`, `sop/`, `tests/test_voicebank.py`, `tests/test_sop_index.py` | #5 | merged, PR #11; network-free test fix merged, PR #22 |
| Dashboard v1 | `web/` (`index.html`, `app.js`, `style.css`, `mock/`, `test_dashboard.py`) | #4 | merged, PR #15 |
| Signals and segmenter (L0) | `jadal/signals.py`, `jadal/segmenter.py` | #6 | merged, PR #12 |
| Decider (L1) | `jadal/decision.py`, `tests/test_decision.py` | #7 | merged, PR #20 |
| Eval harness | `eval/` | #8 | merged, PR #16 |
| Integration: real pipeline, API docs, demo | `docs/API.md`, `Makefile`, `scripts/demo.sh` | #17 | merged, PR #19 |
| Dashboard v2 | `web/` (dropdown filters, receipt tree, step timeline) | #18 | merged, PR #21 |
| Funnel (accept/reject/review router) | (see PR) | n/a | merged, PR #29 |
| Repo docs: README, docs/ | `README.md`, `docs/` | n/a | merged, PR #23 |
| PoC demo (one-command launch, stream replay, Pico hook) | `scripts/poc_demo.sh`, `scripts/poc_stream_replay.py`, `jadal/poc_stream/`, `tests/test_poc_demo.py` | n/a | merged, PR #66 |
| Freshness (nonce-LED liveness check) | `jadal/freshness.py`, `tests/test_freshness.py`, `scripts/freshness_demo.py` | #24 | open, PR #25. Live trials not run. |
| Video hub (EmbeddingGemma 2 search and clusters) | (see PR) | n/a | open, PR #28 |
| Baseline scoreboard (B1-B8) | (see PR) | n/a | open, PR #30 |
| Spectral features arm | `jadal/spectral/`, `eval/spectral_features/`, `docs/reports/spectral_features.md`, `tests/test_spectral_*.py` | n/a | merged, PR #52 |
| Per-pixel scorer (rest), GPU pipeline, dashboard | (see PRs) | n/a | open, PRs #57, #61, #67-#73 |
| Laya fusion | (none) | #9 | not in the pipeline. Stretch idea only. |

## Repo layout

Layout as of `origin/main` (55b0654).

```
.
├── README.md
├── .env.example          template; the real .env is git-ignored
├── docs/
│   ├── SPEC.md           build contract (binding; predates the direction above, not yet updated)
│   ├── ARCHITECTURE.md   descriptive architecture and status
│   ├── ATTRIBUTIONS.md
│   ├── README.md         docs index
│   ├── laya_design1_plan.md
│   ├── sarvam_reference.md
│   ├── reports/          certify, verify, edge, funnel, dashboard, pico, service, voicebank, spectral_features, and more
│   ├── screens/          dashboard screenshots
│   └── process/          how the repo was built: workflow, reviews, decision log
├── jadal/                core package (incl. spectral/ arm, PR #52, and poc_stream/, PR #66)
├── eval/                 harness, spectral_features arm (PR #52), gold labels
├── models/               README only; model weights are git-ignored
├── pico/                 Raspberry Pi Pico MicroPython firmware (main.py; RP2040 tested)
├── sim/dock_sim.py       copies clips into the bucket and posts ingest events
├── sop/                  sample SOP (labelled SAMPLE) and deterministic index
├── scripts/              demo and PoC scripts
├── tests/                unit and integration tests
├── voicebank/            messages.json, gen_voicebank.py; generated wavs git-ignored
└── web/                  static dashboard, mock data, dashboard tests
```

## Quick start

The project uses a Python 3.12 conda env with torch+cu130 (see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). Do not `pip install` into it without checking first.

```
export PYTHONNOUSERSITE=1
PY=<env>/bin/python    # the Python 3.12 conda env described above

$PY -m unittest discover -s tests     # module tests
$PY -m unittest discover -s web       # dashboard tests
JADAL_USE_STUBS=1 $PY -m uvicorn jadal.service:app   # service on stubs
```

Secrets are read from environment variables (`SARVAM_API_KEY`, `CF_API_TOKEN`, `CF_ACCOUNT_ID`). Put them in a local `.env`, which is git-ignored. Never commit a key.

## Run the demo

- **On `main`:** `make demo` runs `scripts/demo.sh` on the IndEgo clips. It goes through the service, the batches, the verdict table, re-verification of every receipt, and a repeated-run determinism check.
- **The proof-of-concept demo** (one-command launch, status page, live stream replay and Pico hook) is merged to `main` via PR #66: see `scripts/poc_demo.sh`.

## Licences

Licences and attributions are in [`docs/ATTRIBUTIONS.md`](docs/ATTRIBUTIONS.md).
