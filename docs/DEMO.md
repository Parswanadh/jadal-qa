# PoC demo script (3 minutes, for judges)

Setup (1 min before): `EVENTS=/home/parshu/projects/jadal-qa-wt/poc_data/stream_events.jsonl
bash scripts/poc_demo.sh` — or play `docs/screens/poc_demo_backup.mp4` (80 s,
no dependencies) if anything is broken.

## What to say/click (times are video time)

0:00 — "Real factory head-camera video (Egocentric-10K, 1080p HEVC) goes
through the quality funnel LIVE, in 2-second chunks, as if a worker were
recording right now."
Point at the terminal: each `chunk` line = one 2 s slice judged in isolation:
`t, running_score, decision`.

0:40 — "First six chunks are clean: score ~0.78, verdict `continue`
(watching) then `accept` when the 10 s window completes above the operating
point." (Amber Pico LED = watching, green = accept.)

1:10 — "Now I cut the feed — a lens-covered blackout at t=12 s."
Watch: the very next chunk flips to `early_reject`, evidence on screen:
`S0 rule black p95=0.0 < 0.03`. Time-to-reject: **0.0 s of video** (same
2 s chunk; freeze degradation also 0.0 s via `frozen temp_e < 5e-4`).
(Time-to-reject is chunk-quantized: a degradation starting mid-chunk is first
seen at the next 2 s boundary, so the true delay can be up to one chunk
larger than the reported video-seconds number.)
Red Pico LED.

1:50 — "Every verdict carries evidence (value, operator, threshold) and the
per-chunk cost: latency ms + RAM + decode backend, all in
`poc_data/stream_events.jsonl`."
`tail -n 3 $EVENTS | python3 -m json.tool`

2:20 — Pico (optional): "The verdict also drives a Pi Pico status LED —
green/amber/red — under `flock tools/pico.lock`. No hardware today? It logs
`pico: not connected` and the demo continues."

2:40 — Close: "This is descriptive behaviour on unlabelled video, not
accuracy. Accuracy is measured only on 150 human-labelled windows
(AUROC ~0.7–0.8, leave-one-clip-out; re-verify at
`/home/parshu/projects/jadal-qa-wt/gpu-pipeline/eval/results/gpu/cascade.json`
(`loco.auc`, `loco.auc_ci95`, n=150) and the label snapshot
`/home/parshu/projects/jadal-qa-wt/spectral_data/GOLD_FINAL_150*`)."

## Expected output

- Clean chunks: `continue` ×4, `accept` at chunk 5 (score ~0.78–0.89).
- Blackout at t=12 s: `early_reject` from chunk 7 on, score → 0.00.
- `time_to_reject_s: 0.0` in the summary JSON.

## Failure fallbacks

| If this breaks | Do this |
|---|---|
| No GPU / NVDEC missing | Replay uses CPU decode already (`"decode": "cpu"`); nothing to change |
| `poc_server.py` missing | `poc_demo.sh` prints a stub line and continues; `tail -f` the events file instead |
| Pico absent | Hook logs `pico: not connected`; demo continues (tested with `FakePicoSerial`) |
| Slow machine (latency >2 s/chunk) | Normal under load: decode of 1080p HEVC dominates; per-chunk latency is printed honestly — never hidden |
| No network/clip missing | Play `docs/screens/poc_demo_backup.mp4` (80 s, pre-rendered from real events) |

## Claims to avoid (binding)

- Do NOT call any number on this video "accuracy": there are NO human labels
  for Egocentric-10K; these are descriptive verdict rates.
- Do NOT say thresholds are certified (n=150 too small) or "real-time
  guaranteed" (per-chunk wall latency 0.6–6 s under load, CPU decode).
- Do NOT say "hyperspectral" (RGB only), "signed receipts" (content-hashed),
  or "tested on Raspberry Pi" (Pico LED only; funnel ran on an RTX 4070 laptop).
