# Jadal QA

Jadal QA is a quality-assurance add-on for crowd-captured head-camera (egocentric) video. For each clip window it decides whether the footage is usable (dark, blocked, frozen, duplicate, hands not visible, no active work), and it writes each decision with its evidence into a content-hashed receipt. Built for the Awign Hackathon: Physical AI (10-11 October 2026).

This page states what is built, what is measured, and what is not yet shown. It is current as of 2026-10-10. Measured numbers are in the table below, each with its n and label source.

The build contract is [`docs/SPEC.md`](docs/SPEC.md). The architecture and the status of each stage is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The docs index is [`docs/README.md`](docs/README.md).

## Direction (2026-10-10)

- **Edge-first.** The target is analysis on the capture device, with no mandatory human in the loop. Running on a Raspberry Pi is **not yet measured**. The only speed numbers are on x86 laptop cores, as a proxy. A real Pi run is pending.
- **Whole-video, per-pixel analysis of every frame.** No sampled frames. Streaming: 2-second chunks are analysed while the worker records. Dark, blocked or frozen footage is rejected in the first 1-2 seconds.
- **Deterministic, mathematics-first decisions.** The inputs are measured signals: noise level, sharpness, exposure, frame difference against the noise floor, residual motion after camera-motion removal, and rhythm tests. The intended decision is a transparent rule list with exact thresholds, and every reject carries its evidence (value, comparison, threshold). Logistic regression and boosted trees are comparison baselines only. The current scorer is a logistic head over 26 per-pixel features. The first threshold rule list rejected nearly everything and was dropped, so the rule-list form is not yet working.
- **Not in the pipeline.** A generic vision-language model was tried and stopped as too probabilistic. Interim run: SmolVLM2-2.2B, 79 labels, yes/no AUROC 0.57. At about 1.4 s per segment it would take about 58 GPU-days for 3.6 million 10-second segments. Laya (text-only) and Clef (hosted) are not in the pipeline.
- **EmbeddingGemma 2 is an optional helper**, used for hub search, de-duplication and frame heads. It is not the core.
- **RGB, not hyperspectral.** The cameras record 3 bands (RGB). We do per-pixel frequency and temporal-spectral analysis of RGB frames. This is **not** hyperspectral imaging. Recovering a spectrum from RGB is under-determined, and a hyperspectral sensor would be needed for real spectral data.

## Status and measured results

All AUROCs are leave-one-clip-out (LOCO) on the human gold labels, with clip-bootstrap 95% intervals where given. Source: the builder's measurement notes, 2026-10-10.

| What was measured | Result | n and label source |
|---|---|---|
| 26 per-pixel features + logistic head | AUROC 0.81 (0.71-0.87) | 150 windows from 7 clips, one human labeller |
| EmbeddingGemma 2 frame heads, pooled over every 0.5 s frame (pretrained on AI-labelled frames) | AUROC 0.79 | same 150 windows |
| EmbeddingGemma 2 human-only head | AUROC 0.82 | same 150 windows |
| EmbeddingGemma 2 combined | AUROC 0.83 | same 150 windows |
| 1D-conv on 8 per-frame series | AUROC 0.78 (0.63-0.86) | same 150 windows |
| Boosted trees / physics-only logistic | AUROC 0.76 / 0.72 | same 150 windows |
| Quality problems only (dark, duplicate, undecodable) by physics features | AUROC 0.85-0.90 | same 150 windows; quality labels only |
| Best single feature: residual optical flow in the hand region after ego-motion removal | AUROC 0.81 | same 150 windows |
| 26 features + logistic, deciding every window | balanced accuracy 0.74 (26 of 100 good windows wrongly rejected) | same 150 windows |
| 26 features + logistic, keeping the 80% most confident windows | balanced accuracy 0.84 (15 of 80 good windows wrongly rejected; 88% of bad rejected) | same 150 windows; the curve is not monotone at this n |
| Harness arms (OpenCV + MediaPipe signals; spectral arm B) | AUROC 0.47-0.59, intervals touch 0.5 | 150 windows, each arm's own protocol |
| Egocentric-10K proof of concept (58 clips, 4.404 h, 47 factories, 1,546 windows) | **descriptive only**: every window processed, no accuracy claimed | no human labels exist for this set |

What this means:

- **Accuracy is not yet high.** About 0.8 AUROC is the honest ceiling so far. Two clips (pallet, pallet2) are at chance for every model.
- **The human set is small.** 150 windows: 100 usable, 50 bad (NO_ACTIVE_WORK 23, HANDS_NOT_VISIBLE 14, DUPLICATE 6, TOO_DARK 6, UNDECODABLE 1). It comes from 7 IndEgo videos (Fraunhofer IPK), 480p at about 10 fps. The file names suggest only about 3 wearers. It is not Awign footage.
- **Egocentric-10K has no human labels.** Its frame labels come from an AI model. The proof of concept shows the pipeline runs on real 1080p/30 fps HEVC footage. Its verdicts are not accuracy results. The scorer was calibrated on 480p/10 fps IndEgo, so the domain shift is untested and probably inflates the reject share (an earlier partial run of 1,214 windows had 37% rejects). A 60-window out-of-domain labelling queue is ready for the builder to label.
- **Speed (single laptop core and RTX 4070 laptop GPU, not a Pi).** One CPU core decodes 1080p HEVC at 43-55 fps. NVDEC decodes 620-730 fps but saturates at one engine. GPU kernels alone reach about 700 hours of video per hour. End-to-end throughput is lower and is decode-bound. It is not yet measured as one number.

Still in progress or not claimed:

- Raspberry Pi measurement (pending).
- End-to-end GPU throughput as a single number (pending).
- Accuracy on Egocentric-10K (no labels yet).
- Custom "hand-region residual-motion rhythm network": target LOCO AUROC of at least 0.85, result pending. A negative result is acceptable.
- Cross-device determinism of the GPU path, and 10 identical receipt hashes through the real modules (pending).
- Freshness nonce-LED live trials (not run).
- Hyperspectral imaging (not done; see above).

## What is and is not claimed

- **No certified bounds.** The certification machinery exists in `jadal/certify.py`: `exact_binomial_upper` (Clopper-Pearson), `pick_threshold` (fixed-sequence testing) and `batch_rule`. It does not give a usable guarantee on the current data. With 150 labels from 7 clips, a multiple-testing correction needs about 86 clean windows (59 with fixed-sequence testing alone). Windows within a clip are correlated, so the effective sample size is roughly 20-43. A conformal calibration from the training clips did not hold on held-out clips: a nominal 2.7% false-reject rate became 46%. Error rates are therefore reported as observed, with n and label source, and not as certified bounds. A certified bound needs more labelled, independent clips than we have.
- **Receipts are content-hashed, not signed.** A receipt is the SHA-256 of its canonical JSON. Re-verifying means re-running the pinned pipeline on the original clip and comparing hashes. A tampered receipt fails. An optional HMAC sidecar exists (see [`docs/reports/verify.md`](docs/reports/verify.md)), but no receipt is signed.
- **Determinism is narrow.** The CPU feature path gives an identical quantised digest over 10 repeated runs, on this machine. GPU numerics are not claimed identical across devices, and cross-architecture determinism is not claimed.
- **Raspberry Pi.** A Raspberry Pi Pico (RP2040) drives verdict LEDs, and the Pico firmware is verified on that hardware (MicroPython 1.29.0). This is a status light, not the compute device. RP2350 is not tested. Running the analysis on a Pi is not measured.
- **Labels.** Egocentric-10K-Evaluation labels are AI-generated (gemini-2.5-flash, per [`docs/ATTRIBUTIONS.md`](docs/ATTRIBUTIONS.md)) and not human-checked. The 150 human gold labels come from one labeller.
- **Thresholds.** The answer threshold on `main` is a placeholder (0.9). No threshold on `main` is certified.
- **Batch rule.** The batch rule is a fixed rule (reject if the rejected count is greater than k). It has no statistical guarantee at n = 5 to 10.
- **Face blur** is a colour-based proxy. It is not a privacy guarantee.
- **SOP and voice.** The SOP is a labelled sample. Voice messages are pre-recorded machine-draft translations.
- **Not yet run on Awign footage, cameras or criteria.**

## Data flow (status on `origin/main`, f58e0a1, 2026-10-10)

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
| Freshness (nonce-LED liveness check) | `jadal/freshness.py`, `tests/test_freshness.py`, `scripts/freshness_demo.py` | #24 | open, PR #25. Live trials not run. |
| Video hub (EmbeddingGemma 2 search and clusters) | (see PR) | n/a | open, PR #28 |
| Baseline scoreboard (B1-B8) | (see PR) | n/a | open, PR #30 |
| Per-pixel scorer, GPU pipeline, PoC demo and dashboard | (see PRs) | n/a | open, PRs #52, #57, #61, #66-#73. PoC demo on `integration/poc`, pending merge. |
| Laya fusion | (none) | #9 | not in the pipeline. Stretch idea only. |

## Repo layout

Matches `origin/main`.

```
.
├── README.md
├── .env.example          template; the real .env is git-ignored
├── docs/
│   ├── SPEC.md           build contract (predates the direction above; not yet updated)
│   ├── ARCHITECTURE.md
│   ├── ATTRIBUTIONS.md
│   ├── README.md         docs index
│   ├── laya_design1_plan.md
│   ├── sarvam_reference.md
│   ├── reports/          certify, verify, edge, funnel, dashboard, pico, service, voicebank, and more
│   ├── screens/          dashboard screenshots
│   └── process/          how the repo was built: workflow, reviews, decision log
├── jadal/                core package
├── models/               README only; model weights are git-ignored
├── pico/                 Raspberry Pi Pico MicroPython firmware (main.py; RP2040 tested)
├── sim/dock_sim.py       copies clips into the bucket and posts ingest events
├── sop/                  sample SOP (labelled SAMPLE) and deterministic index
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

- **On `main`:** `make demo` runs `scripts/demo.sh` on the IndEgo clips. It goes through the service, the batches, the verdict table, re-verification of every receipt, and a 10x determinism check.
- **The proof-of-concept demo** (one-command launch, status page, live stream replay and Pico hook) is on branch `integration/poc`, pending merge. `docs/POC_STATUS.md` and `scripts/poc_up.sh` are on that branch, not on this one, so do not look for them here yet.

## Licences

Licences and attributions are in [`docs/ATTRIBUTIONS.md`](docs/ATTRIBUTIONS.md).
