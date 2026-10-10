# Gold-label tool (video)

Stdlib-only labelling page for the ~150 ten-second IndEgo segments in
`eval/gold/segments.json`. Shows **no model output** (unbiased).

## Run (human switchover — labels preserved)

The human is currently labelling on port 8811 with the old tool. To switch
without losing labels, run the new tool against the real labels file on
port 8812:

```sh
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \
  tools/gold_label/label_server.py \
  --labels /home/parshu/projects/jadal-qa-wt/funnel/eval/gold/labels.json \
  --port 8812
# then open http://127.0.0.1:8812/
```

`--labels PATH` / `--manifest PATH` / `--port N` are accepted; defaults are
`eval/gold/labels.json`, `eval/gold/segments.json`, 8811. Existing entries in
`--labels` are always preserved and reused — saves are atomic (temp file +
`os.replace`), append-only `history` + `latest` map, resume-safe.

## What the page shows

- **Video player**: 10 s clip cut on demand with ffmpeg
  (`-ss start -t dur -i clip -an -c:v libx264 -preset veryfast -crf 28
  -pix_fmt yuv420p -movflags +faststart`), cached under
  `data_runtime/gold/video/<segment>.mp4`, served with HTTP Range (206
  partial content). `<video autoplay loop muted playsinline controls>` with a
  speed dropdown (1x / 1.5x / 2x, remembered in localStorage). Next 3
  segments are pre-cut in a background thread.
- **Task context line**: dataset ground truth (not a model verdict) — IndEgo
  VIA annotated key step(s) overlapping the window for assembly clips
  (`Task: ECU assembly · annotated step(s) in this window: 02_attach_ram_chip`,
  or `between annotated steps`), folder/task name for logistics clips,
  `unknown` fallback. Plus clip name, time window, `segment N of 150`.
- **GOOD/BAD definition panel**: collapsible, open on first visit then
  remembered; 8 reasons with one-line examples.
- **Contact sheet**: 6 still frames inside a collapsed `<details>`.
- **Keys**: `g` good, `b` bad, `s` skip, `u` undo, arrows prev/next, space
  play/pause, `1`–`8` pick BAD reason and save.
- **Progress**: `labeled n/150`, real ETA from median seconds-per-label
  (`~? min left` until 2+ labels exist).

## Verification (2026-10-10, port 8812)

- `pytest tools/gold_label/test_label_server.py`: **27 passed**.
- ffmpeg cut verified on the command line (451 KB mp4 for g000).
- Playwright: `video.readyState` 4, `currentTime` 0.54 → 3.05 over 2.5 s
  (advancing); task line, open-by-default definition panel, speed options
  all present. Range `bytes=0-99` → 206, unsatisfiable → 416.

## NOT DONE

- Keyboard `1`–`8` shortcuts and speed persistence are implemented but were
  only code-reviewed, not clicked through in a browser.
- Prefetch endpoint (`/api/prefetch`) likewise untested in-browser.
- ETA shows `~? min left` until the human has labelled 2+ segments in the
  real labels file (median needs one gap).
