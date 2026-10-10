# Freshness module report (issue #24)

Owner branch: `feat/freshness` (worktree `.../jadal-qa-wt/freshness`).
Nonce-LED liveness check: proves a clip was recorded live in view of our Pico
beacon, so a replayed clip (earlier nonce) FAILS.

## What exists

- `jadal/freshness.py`
  - `issue_nonce(link=None, rng=None)` — 32-bit nonce via `secrets` (injectable
    `rng` for tests), `validity_s=30`. Plaintext nonce is never logged; receipts
    carry only `nonce_hash = sha256(hex)`.
  - `record_live(nonce, link, cam_index=0, seconds=6.0)` — captures
    webcam `/dev/video0` (640x480@30, exposure/gain locked best-effort), sends
    `FLASH <hex>` ~0.5 s in, writes mp4 + timestamps + preview `frame.jpg`
    under `data_runtime/freshness/` (git-ignored).
  - `verify_clip(path_or_frames, nonce, cfg)` — finds LED ROI as peak temporal
    variance (floor 100 gray²; else `inconclusive`), extracts ROI luminance
    series, searches pattern start, decodes 32 bits with 1-D Otsu threshold and
    computes matched-filter Pearson correlation vs the expected pattern;
    `z = (corr - mean_null)/std_null` over 2000 seeded random patterns.
    `fresh` iff `bits_matched >= 30/32` **and** `z >= 4.0`; `inconclusive` when
    LED not found / signal too weak / degenerate nonce; else `stale`.
    Deterministic: same input twice gives the identical dict (tested on
    synthetic frames and on a real mp4).
  - `synthesize_frames(...)` — deterministic synthetic LED clip generator
    (seeded noise, known pattern start) for hardware-free tests.
- `tests/test_freshness.py` — 8 hardware-free unittests, all passing:
  correct nonce => `fresh` with lag ≈ true start; wrong nonce => `stale`;
  no LED => `inconclusive`; determinism; receipt field shape.
- `scripts/freshness_demo.py` — `live` / `replay` / `trials` (N live + N replay,
  Clopper-Pearson bounds via `jadal.certify.exact_binomial_upper`, writes
  `eval/results/freshness_trials.json`) / `synthetic` (SNR sweep, writes
  `eval/results/freshness_synthetic.json`).

## Measured numbers (all I ran — nothing invented)

- Synthetic sweep (`synthetic --per 5`, sigmas 2/5/10/20): **n=20,
  detection 20/20, replay-rejection 20/20**
  (`eval/results/freshness_synthetic.json`).
  True-nonce operating point: z ≈ 5.5, bits 32/32. Wrong-nonce: z ≈ 2.0–2.9
  (inflated by lag-search maximisation), bits 19–23/32. Threshold `z >= 4.0`
  plus `bits >= 30` separates with margin on synthetic data.
- Hardware smoke (Pico on `/dev/ttyACM0`, webcam `/dev/video0`), 2026-10-10:
  - Pico PING/PONG RTT ≈ 15 ms (min 15.04, bound 7.5 ms); `FLASH` accepted.
    NOTE: **no `HB` heartbeat frames arrived** (`connected: False`,
    `last_hb_age_s: None`) — this firmware/build does not appear to emit the
    1 s heartbeat. Flashing/verification do not need HB, but the service's
    `/pico/status` health read will show disconnected; pico owner may want to
    check the firmware HB schedule.
  - 1× `live` (beacon NOT in camera view — see below): `stale`, bits 18/32,
    z=0.25. Correct: no false-fresh without the beacon.
  - 1× `replay` of that clip vs a new nonce: `stale` → REPLAY-REJECTED.
  - Determinism on the real mp4: identical dict twice.

## NOT DONE (needs a human + 5 minutes)

- **10 live + 10 replay hardware trials were NOT run (N=0).** The laptop webcam
  faces the office (see `data_runtime/freshness/frame.jpg`: two people,
  monitors, ceiling spots) and the Pico is nowhere in frame. I cannot move
  hardware. To finish, the user must:
  1. Hold/tape the Pico **30–50 cm in front of the laptop webcam, onboard LED
     facing the lens**, roughly centred; keep it still during each 6 s capture.
  2. Dim the room / turn away from the ceiling spots if detection is weak
     (the ROI finder locks onto the strongest variance blob — a flickering
     monitor in frame competes with the LED).
  3. Run: `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python
     scripts/freshness_demo.py trials --n 10`
  4. Accept if detection 10/10 and false-accept 0/10 (exact upper bounds print
     automatically); otherwise demote to pilot per issue #24.
- Limits (also in module docstring): defeats replay only by parties without the
  nonce inside its 30 s window; does NOT defeat an attacker who watches the
  nonce live and renders it. Deployment needs the beacon in the head camera's
  view at clip start (e.g. at the dock). ROI finder assumes the beacon is the
  dominant variance blob.

## How to run

```bash
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_freshness -v
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/freshness_demo.py synthetic --per 5
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/freshness_demo.py live --seconds 6.0
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/freshness_demo.py replay --clip <clip.mp4>
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/freshness_demo.py trials --n 10  # needs Pico in view
```
