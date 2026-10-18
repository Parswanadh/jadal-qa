---
name: jadal-honesty
description: Claims hygiene for this project: what may and may not be said, how to word results, banned phrases, required evidence per number, how to describe small-n and out-of-domain results. Load before writing any report, README, PR body or claim.
license: Apache-2.0
---
- Every number -> a results JSON in the repo (n, label source, protocol, CI, device, command). If you cannot point to the file, delete the number or write "not measured".
- Banned unless proven in-repo: "first", "state of the art", "novel", "zero competitors", "unforgeable", "signed" (receipts are content-hashed, not signed), "certified bound" (not achievable at n=150: clip correlation, effective n ~11-43), "hyperspectral" (RGB), "runs on a Raspberry Pi" (never measured), "high accuracy" (not yet).
- Wording for small data: "on 150 windows from 7 clips and ~3 wearers (one labeller)", "out-of-domain n_bad = 9 (exploratory)", "CI by window is optimistic".
- State what was NOT done and why. Keep negative results next to positive ones. Corrections to earlier claims are logged, not hidden (see `$JADAL_BUNDLE/docs/FACTS.md` updates: spatial-localisation claim withdrawn; motion cue inverted; synthetic negatives harmful).
- The motion cue is described as "motion/stability cues predict usable windows; part of the cue may be head stability", NOT "hand work is detected".
- Descriptive rates on unlabelled video are not accuracy. Verdict rates on Egocentric-10K from IndEgo-trained models are not quality measures (they failed to transfer).
