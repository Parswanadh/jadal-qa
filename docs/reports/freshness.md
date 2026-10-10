# Freshness module report (issue #24)

Owner branch: `feat/freshness` (worktree `.../jadal-qa-wt/freshness`).
Nonce-LED liveness check: proves a clip was recorded live in view of our Pico
beacon, so a replayed clip (earlier nonce) FAILS.

## What exists

- `jadal/freshness.py`
  - `issue_nonce(link=None, rng=None)` — 32-bit nonce via `secrets` (injectable
    `rng` for tests), `validity_s=30`. The value NEVER appears in filenames,
    logs, sidecars, or results — clips are named by `nonce_hash[:16]`.
    `nonce_hash` is a SHA-256 COMMITMENT (binding), NOT a hiding one: the
    32-bit preimage is trivially enumerable, so secrecy holds only inside the
    validity window; afterwards the value is worthless (no clip can be
    back-dated into a closed window — `verify_clip` enforces it).
  - `record_live(nonce, link, cam_index=0, seconds=6.0)` — captures
    webcam `/dev/video0` (640x480, exposure/gain locked best-effort), sends
    `FLASH <hex>` ~0.5 s in, measures the real capture rate from per-frame
    times and writes the mp4 at that rate; mp4 + sidecar (frame times +
    wall-clock capture start, hash-named) under `data_runtime/freshness/`
    (git-ignored). Capture loop aborts after 30 consecutive empty reads.
  - `verify_clip(path_or_frames, nonce, cfg)` — resamples to the 100 ms bit
    grid using REAL timestamps (`cfg` > sidecar `frame_times` > container
    presentation times > nominal `fps` fallback); reports `fps_measured` +
    `fps_source`. Finds LED ROI as peak temporal variance (else
    `inconclusive`), searches pattern start on a 10 ms grid, decodes 32 bits
    (Otsu, polarity-free), correlates vs the expected pattern. Null: seeded
    random wrong nonces through EXACTLY the same procedure (same lag search,
    same Pearson normalisation). Decision: `fresh` iff exact 32/32-bit match
    (primary) AND `z >= 3.0` (supporting). Window check first: clip wholly
    outside `[issued_at, issued_at+validity_s]` is `stale`/`expired`.
    Result keys: `nonce_hash, status, lag_ms, z, bits_matched, bits_total,
    reason (ok|mismatch|expired|no-signal), fps_measured, fps_source`.
    Deterministic: same input twice gives the identical dict.
  - `synthesize_frames(...)` — deterministic synthetic LED clip generator
    (seeded noise, true per-frame times, any fps) for hardware-free tests.
- `tests/test_freshness.py` — 14 hardware-free unittests, all passing:
  fresh with lag ≈ true start; wrong nonce => stale; low-entropy `0x0000FFFF`
  => fresh with non-inflated z; no LED => inconclusive; 25 fps + 15 fps +
  container-timestamp path => fresh; expiry in/outside window; no nonce value
  in filenames/sidecar/result; determinism; receipt field shape.
- `scripts/freshness_demo.py` — `live` / `replay` / `trials` (N live + N replay,
  Clopper-Pearson bounds via `jadal.certify.exact_binomial_upper`, writes
  `eval/results/freshness_trials.json`) / `synthetic` (null-tail calibration:
  4 SNR levels x 2 clips x 5000 seeded wrong nonces with full lag search;
  writes `eval/results/freshness_synthetic.json`).

## Measured numbers (all I ran — nothing invented)

- Null-tail calibration (`synthetic`, fix round 1):
  `eval/results/freshness_synthetic.json`. 40000 seeded wrong nonces (levels
  A snr38 / B snr6 / C snr2 / D snr1, full lag search each): pooled p99 = 2.56
  (per-level 2.53–2.68), p999 = 3.53, max = 4.56. Threshold 3.0 sits above the
  pooled p99. Full-rule false accepts (exact 32/32 + z>=3.0 at each pattern's
  own best lag): **0/10000 per level** (Clopper-Pearson upper 0.023% at
  delta=0.1) — FA <= 1% on wrong nonces. True nonces: z 6.6–7.0, bits 32/32
  at every SNR level.
- Hardware smoke (Pico on `/dev/ttyACM0`, webcam `/dev/video0`), 2026-10-10 —
  full outputs in `eval/results/freshness_hw_smoke.json`:
  - Pico PING/PONG RTT min 15.04 / median 15.09 / max 20.13 ms (bound 7.5 ms);
    `FLASH` accepted. `status()` within ~1 s of `connect()` read
    `connected: False` with no HB yet — CORRECTION of the round-0 claim: this
    is host-side timing (autodetect `_probe` swallows the first HB; the
    reviewer reproduced it with `FakePicoSerial`, no hardware), NOT a firmware
    fault — `pico/main.py:205-208` sends HB every 1.0 s. Callers should wait
    >1.2 s after connect before trusting `status()`.
  - 1× `live` with the beacon NOT in frame: `stale`, bits 18/32, z=0.25 —
    correctly not fresh, but carries no detection information.
  - Replay smoke: **REMOVED from the claims (was vacuous)** — a clip with no
    LED is stale for any nonce. Live/replay trials: **NOT MEASURED (N=0 live
    trials)**.
  - Determinism on the real mp4: identical dict twice.
- Privacy note: captures record whatever the camera sees (round-0 preview
  showed office bystanders). Keep beacon framing tight; delete raw clips after
  the trials. Round-0 clip/sidecar renamed to hash-based names; sidecar
  verified to contain no nonce value.

## NOT DONE (needs a human + 5 minutes)

- **10 live + 10 replay hardware trials: NOT MEASURED (N=0 live trials).** The
  laptop webcam faces the office and the Pico is nowhere in frame. I cannot
  move hardware, and no hardware was opened in fix round 1. To finish:
  1. Hold/tape the Pico **30–50 cm in front of the laptop webcam, onboard LED
     facing the lens**, roughly centred; keep it still during each 6 s capture.
  2. Dim the room / turn away from the ceiling spots if detection is weak
     (the ROI finder locks onto the strongest variance blob — a flickering
     monitor in frame competes with the LED).
  3. Run: `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python
     scripts/freshness_demo.py trials --n 10`
     (`scripts/freshness_demo.py trials` is ready; fix I-4 first was done —
     it now records/uses real timestamps, so the trials measure the LED, not
     the fps. Re-check the z=3.0 threshold on hardware before the gate.)
  4. Accept if detection 10/10 and false-accept 0/10 (exact upper bounds print
     automatically); otherwise demote to pilot per issue #24.
- Limits: defeats replay only by parties without the nonce inside its 30 s
  window; does NOT defeat an attacker who watches the nonce live and renders
  it. Deployment needs the beacon in the head camera's view at clip start
  (e.g. at the dock). SPEC's `freshness` receipt field list predates `reason`,
  `fps_measured`, `fps_source` — needs a SPEC amendment (SPEC.md is read-only
  for this worker; proposing, not editing).

## How to run

```bash
PYTHONNOUSERSITE=1 timeout 200 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -p 'test_freshness*'
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/freshness_demo.py synthetic
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/freshness_demo.py live --seconds 6.0
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/freshness_demo.py replay --clip <clip.mp4>
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/freshness_demo.py trials --n 10  # needs Pico in view
```

## Fix round 1 (2026-10-10; no hardware opened)

Rulings addressed: I-1 hash-only filenames + no-value test (old smoke clip
renamed; sidecar grep-clean); I-2/I-3 same-procedure null, 40000 trials,
threshold 3.0 from pooled p99=2.56, exact-32/32 primary, `0x0000FFFF` test;
I-4 real timestamps end-to-end + `fps_measured`, 25/15 fps + container-path
tests; I-5 commitment rewording in docstring/report/demo; I-6/I-9 vacuous
replay claim removed, NOT MEASURED (N=0); I-7 window enforced + tested; I-8
`eval/results/freshness_hw_smoke.json`; HB correction. Real outputs:

```text
$ PYTHONNOUSERSITE=1 timeout 200 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -p 'test_freshness*' 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 14 tests in 13.807s
OK
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/freshness_demo.py synthetic  # saved to eval/results/freshness_synthetic.json
threshold 3.0, pooled null p99 = 2.5642 (n=40000); per-level full-rule FA:
A 0/10000 (CP-upper 0.00023), B 0/10000 (0.00023), C 0/10000 (0.00023),
D 0/10000 (0.00023); true z 6.62-7.02, bits 32/32 every clip
```
