# Jadal QA — build spec (the contract every module codes against)

Project: **Jadal** — a deterministic QA add-on for crowd-captured head-camera video.
Repo root: `/home/parshu/projects/jadal-qa` (new repo; unrelated to the old GitHub repo `jadal`).
Python: `/home/parshu/envs/jadal/bin/python` (3.12). **Never `pip install` into it** (one agent owns it). If you need a missing package, say so in your final message. Pure-Python modules may also be tested with `/home/parshu/miniconda3/bin/python` (numpy/scipy/pandas available).
Style: small modules, type hints, docstrings, **no network calls at import time**, no secrets in code (keys come from env vars: `SARVAM_API_KEY`, `CF_API_TOKEN`, `CF_ACCOUNT_ID`).

## Flow
head camera (unchanged) -> auto-sync -> existing ingestion backend (object store + upload event) -> **upload event** -> Jadal service (floor hub or cloud, same code)
-> segmenter -> L0 signals -> L1 decisions -> L2 certify -> per-segment verdict -> batch rule -> receipt -> outputs (dashboard, voice, Pico LED).
Simulated bucket for the demo: directory `data_runtime/bucket/` watched by the service; `sim/dock_sim.py` copies clips in one by one.

## Determinism rules (all modules)
- Same input + same pinned weights/config => same output. No sampling, no temperature, no randomness without a fixed seed.
- **Every model call is per segment (batch size 1) with fixed frame sampling.** A segment must give the identical result whether it is processed alone or inside a batch of 5 (padding/kernels differ otherwise). Set `torch.use_deterministic_algorithms(True, warn_only=True)`, `torch.backends.cudnn.benchmark = False`, fixed seeds. Required test: process one segment alone and inside a batch of 5 and assert identical `receipt_sha256`.
- **Probabilities and other floats are rounded to 4 decimals before they enter a receipt** (L0 metrics to 4 decimals as well). Sort dict keys and lists where order is not meaningful.
- Record versions: python, torch, model names + weight sha256 (or file size + name if hashing is too slow), pipeline config hash.
- **Which bytes are hashed:** `segment_sha256` is the SHA-256 of the file the pipeline actually READ (the original). L0 signals are computed on the original frames. If faces are blurred, the blurred output is written to a SEPARATE file under `data_runtime/blurred/` and its hash is stored in the receipt as `blurred_sha256`. Re-verify always re-reads the original.
- **Durable outputs only under `/home/parshu/projects/`. Never write results to `/tmp`.**
- `jadal/__init__.py` and `jadal/config.py` already exist and are READ-ONLY for module agents (propose changes in your final message instead).

## Data contracts (JSON; field names are fixed)
Reason codes (exact strings): `UNDECODABLE`, `HANDS_NOT_VISIBLE`, `BLUR`, `TOO_DARK`, `OVEREXPOSED`, `SHAKY`, `DUPLICATE`, `WRONG_STEP`, `NO_ACTIVE_WORK`, `LOW_CONFIDENCE`.
Verdicts: `accept` | `reject` | `review` (review = send to supervisor). Batch decision: `accept` | `reject`.

Segment: `{"segment_id": str, "source_clip": str, "worker_id": str, "site_id": str, "start_s": float, "end_s": float, "path": str, "sha256": str}`

Signals (L0, per segment): `{"decodable": bool, "blur_var": float, "brightness_mean": float, "under_exposed_frac": float, "over_exposed_frac": float, "motion_mean": float, "hands_visible_frac": float, "hand_count_mode": int, "faces_found": int, "faces_blurred": int, "phash": str, "dup_of": str|null}`

Decisions (L1, per segment): `{"hands_ok": {"p_yes": float}, "active_work": {"p_yes": float}, "step": {"choice": str|null, "probs": {str: float}, "answer_conf": float}, "model_versions": {str: str}}`

Verdict (L2, per segment): `{"segment_id": str, "verdict": str, "reasons": [str], "answer_conf": float, "threshold": float}`

Batch: `{"batch_id": str, "site_id": str, "segment_ids": [str], "verdicts": [Verdict], "batch_decision": str, "rule": "reject if rejected_count > k", "k": int, "n": int, "created_utc": str}`
(The batch rule is a FIXED rule. It carries no statistical guarantee for n = 5-10; never describe it as one.)

Receipt: `{"receipt_id": str, "segment_sha256": str, "signals": Signals, "decisions": Decisions, "verdict": Verdict, "config": {"thresholds": {...}, "pipeline_version": str, "versions": {...}}, "receipt_sha256": str, "created_utc": str}`
`receipt_sha256` = SHA-256 of the canonical JSON (sorted keys, separators `(",", ":")`, UTF-8) of the receipt **without** `receipt_sha256`, `receipt_id`, and `created_utc`. `receipt_id` = first 16 hex chars of `receipt_sha256`.
Re-verify = re-run the pipeline on the stored segment with the stored config and compare `receipt_sha256`; return `{"match": bool, "expected": hex, "actual": hex, "diff": [field paths]}`.

### Optional receipt field: `freshness` (added after the research gate; integration wires it in only after baseline-v1 is tagged)
`freshness`: `{"nonce_hash": hex, "status": "fresh"|"stale"|"inconclusive", "lag_ms": float|null, "z": float|null, "bits_matched": int, "bits_total": int}` — produced by `jadal/freshness.py`. When present it is part of the receipt (included in `receipt_sha256`); when absent the hash is unchanged. It only supports the claim "a party that did not know the nonce during its validity window could not have produced a clip that shows it"; it does NOT prove the clip is unforgeable (an attacker who watches the nonce live can render it).

## Module ownership and interfaces
- `jadal/certify.py` — `fit_temperature(logits, y) -> float`; `ece(probs, y, bins=15) -> float`; `exact_binomial_upper(errors, n, delta=0.1) -> float` (Clopper-Pearson); `pick_threshold(conf, correct, target_err, delta=0.1, min_accepted=30) -> {"threshold": float|None, "accepted": int, "errors": int, "upper_bound": float|None, "coverage": float}` using fixed-sequence testing from strictest threshold downward; `risk_coverage(conf, correct) -> list[{"threshold","coverage","error"}]`; `batch_rule(verdicts, k) -> str`.
- `jadal/receipts.py` — `canonical_json(obj) -> bytes`; `make_receipt(segment, signals, decisions, verdict, config) -> Receipt`; `receipt_hash(receipt) -> hex`; `verify_receipt(receipt, recompute_fn) -> dict` (as above).
- `jadal/signals.py` (L0) — `compute_signals(path, config) -> Signals`; `blur_faces(frame) -> (frame, n_found, n_blurred)`; `phash_segment(path) -> str`; `find_duplicate(phash, index) -> str|None`.
- `jadal/segmenter.py` — `split_clip(path, seg_seconds, out_dir) -> list[Segment]` (OpenCV/ffmpeg, deterministic naming).
- `jadal/decision.py` (L1) — `Decider.decide(path, signals) -> Decisions` (baseline: EmbeddingGemma 2 embeddings + small heads; Laya fusion later).
- `jadal/pipeline.py` — `run_batch(segments, config) -> Batch + list[Receipt]` calling the above in order.
- `jadal/service.py` — FastAPI app. Endpoints: `GET /health`; `POST /ingest/event {"path": str, "worker_id": str, "site_id": str}`; `GET /batches`; `GET /batches/{batch_id}`; `GET /receipts/{receipt_id}`; `POST /receipts/{receipt_id}/reverify`; `GET /stats/risk_coverage`; `GET /stats/sop_gaps`; `GET /sop/search?q=&lang=`; `GET /voice/{reason_code}/{lang}` (serves wav); `GET /pico/status`; `WS /ws` (pushes batch events). Serves `web/` at `/`. Config in `jadal/config.py` (batch size N default 5, k default 1, segment seconds default 20). State in SQLite `data_runtime/jadal.db`.
- `web/` — static dashboard (no build step, no CDN): `index.html`, `app.js`, `style.css`.
- `sop/` — `sop_garment_v1.md` (SAMPLE SOP, clearly labelled sample), `sop_index.py` (deterministic retrieval).
- `voicebank/` — `messages.json` (reason code x language -> text), `gen_voicebank.py` (Sarvam Bulbul v3; run once; skips existing files), `{lang}/{code}.wav` generated files. Languages: `hi-IN`, `kn-IN`, `ta-IN`, `te-IN`, `en-IN`.
- `pico/` — `main.py` (MicroPython, Raspberry Pi Pico 2, USB serial only), `jadal/pico_link.py` (host side, with a fake-serial mode for tests).
- `sim/dock_sim.py` — copies clips into `data_runtime/bucket/` one by one with a delay and posts `/ingest/event`.
- `eval/` — dataset split, baselines, calibration curves.
- `tests/` — plain `unittest` files named `test_*.py`.

## Pico serial protocol (115200 baud, newline-terminated ASCII)
Pico -> host every 1.0 s: `HB <ticks_ms> <led_state>`. Host -> Pico: `LED G|A|R|OFF`, `FLASH <8 hex digits>` (blink the 32-bit pattern, one bit per 100 ms, MSB first, LED visible to the camera), `PING`. Pico answers `PONG <ticks_ms>`.

## Done means
Code runs; a `unittest` exists for the module; a short module docstring says what is NOT done. Report only what you ran, with the command and its real output tail. Do not claim accuracy numbers without a measurement file.
