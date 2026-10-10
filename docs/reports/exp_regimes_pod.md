# E4 regimes + profiles + router (pod re-run) — experiment report

Track E4 (`feat/exp-regimes-pod`). Builder intent: "for blurry data one
algorithm/model, for clear data another; the product must be customisable and
adapt to whatever data we use." Motivation: every IndEgo-trained scorer failed
on real Egocentric-10K windows (AUROC 0.4–0.53) — a data-type problem.

All numbers come from the results JSONs in `eval/results/exp_regimes/`
(`E4.01.json` … `E4.04.json`), each with n, label source, protocol, CI, device,
cost, command and git_sha. Device: **CPU (pod), no GPU used**. Data regime is
stated per result.

## E4.01 — deterministic regime descriptors (domain-shift diagnosis) — adopt

`eval/exp_regimes/descriptors.py` (no GPU, 8 sampled frames/window): container
(resolution, fps, codec, bitrate, bytes/frame) + sharpness (Laplacian var and
normalised), noise sigma (Immerkaer), luma mean/p05/p50/p95, exposure clipping,
edge density, blockiness (8-px boundary ratio), chroma spread, ego-motion
(phase-correlation shift, static fraction, temporal energy, duplicate fraction).

**n = 730** descriptor records: 150 IndEgo gold (human) · 60 OOD-60 on the
480p/**30fps** H.264 transcode · 60 OOD-60 on the 1080p/30 HEVC source · 100
queue-2 480p/**10fps** transcodes (unlabelled) · 300 seed-7 Ego10K sample
(unlabelled) · 60 new IndEgo-spec re-encodes (E4.03). `ci95 = n/a —
deterministic descriptors`.

**Ranked separators, IndEgo vs OOD-source (Cohen's d).** Container descriptors
separate *exactly by construction* (width/height/fps `d = +inf`, codec
H.264↔HEVC). The OOD side is the human-labelled OOD-60 set (**n = 60,
n_bad = 9 — exploratory**); these are *descriptor* differences, not evidence of
a transfer improvement. The interesting separators are content:

| descriptor | d(IndEgo vs OOD-src) | d(IndEgo vs Ego10K) |
|---|---|---|
| chroma_spread | −3.19 | −2.39 |
| bright_frac | −2.65 | −1.76 |
| luma_mean | −2.57 | −2.80 |
| luma_p95 | −2.54 | −3.00 |
| temp_energy | −2.37 | −1.77 |
| luma_p50 | −1.72 | −1.69 |
| luma_p05 | −1.34 | −1.07 |
| dark_frac | +1.02 | +1.45 |
| noise_sigma | +0.99 | +1.04 |
| sharp_norm | +0.61 | +0.83 |

IndEgo is darker and more saturated (chroma) than Egocentric-10K, which is
brighter with more temporal energy; sharpness separates only weakly — the
distributions overlap. `eg2_clusters.json` could **not** be recomputed
(`caches/ensemble/ood_frames.npz` is absent from the trimmed bundle); the older
partial result is deliberately not used as a headline number.

**Correction vs the task text:** the queue-1 "480p transcode" is **480p/30fps**,
not 480p/10fps; only queue-2 is 480p/10fps. IndEgo is 480×480/10fps H.264.

## E4.02 — regime-specific vs global models (nested LOCO) — drop

150 IndEgo gold windows (human), leave-one-clip-out over 7 clips; logistic
(C=0.01, balanced) on the 26 per-pixel features and the EG2 view; terciles of
sharpness / lighting / motion; 2000× clip-bootstrap CI.

Global AUROC: **feat26 0.712 [0.634, 0.811]**, **EG2 0.848 [0.719, 0.912]**.

No split is adopted. No regime-only model beat global by >0.03 with a CI
supporting it **under the same folds** (feat26 best cell `lighting_lo` +0.125
[−0.10, 0.35]; `sharpness_hi` +0.094 [−0.17, 0.28]). All EG2 regime models were
*worse* than global (δ from −0.02 to −0.32). Several terciles are thin:
`motion_lo` has only n_bad = 4; clips rarely hold both classes in a motion
tercile. Decision: **one global model per view, no per-regime model.**

Each tercile has n = 50 windows; the three splitters give n_bad
sharpness 28 / 10 / 12, lighting 18 / 10 / 22, motion 4 / 16 / 30 (lo / mid /
hi). Per the pre-declared rule (n_bad < 10 is exploratory), `motion_lo` and the
two mid cells with n_bad = 10 are labelled exploratory.

**Correction (2026-10-10):** the mid-tercile selection was
`vals <= t2` (lo + mid) instead of `(vals > t1) & (vals <= t2)` (mid only), so
the earlier JSON reported every `*_mid` cell with n = 100. The selection is
fixed and `regime_vs_global.json` / `E4.02.json` were regenerated: all cells now
have n = 50, and the EG2 δ range moved from −0.01…−0.18 to −0.02…−0.32
(the `lighting_mid` EG2 cell is the new worst at δ = −0.32). The adopt/drop
decision is unchanged — still **drop**.

## E4.03 — domain-gap control (re-encode to the IndEgo spec) — drop

60 OOD-60 sources re-encoded with ffmpeg to the IndEgo spec (480×480, 10 fps,
H.264) and re-described; compared to the 1080p/30 HEVC source and the
480p/30 H.264 transcode. Matching the container shrinks the IndEgo content gap
for only **6/18** content descriptors; several grow (e.g. `luma_p05` d from
−1.34 to +5.02, `sharp_lapvar` −0.06→−3.05). Frozen-ensemble OOD baseline on the
480p/30 transcode: AUROC **0.415** (n = 60, **n_bad = 9 — exploratory**); the
re-encode was **not** re-scored (frozen model pipeline unavailable). Matching
the container does not fix the content gap.

## E4.04 — profiles product layer — adopt

`profiles/schema.json` + 5 profiles (`clear_stationary`, `blurry_lowlight`,
`walking_logistics`, `factory_1080p30`, `default`) and the deterministic router
`jadal/regimes/router.py::choose_profile(descriptors) -> {profile, margin,
reasons, unseen_regime}`. Any descriptor outside the E4.01-observed ranges →
`UNSEEN_REGIME` → `default`; the checked ranges now cover the keys the rules
decide on (`clip_frac` ≤ 0.85, `temp_energy` ≤ 0.40, `static_frac` ∈ [0, 1]),
so out-of-pool values such as `clip_frac = 0.9` or `temp_energy = 2.0` fall
back instead of silently routing. **`receipt_fragment()` returns a helper dict
(profile, margin, reasons, unseen flag, descriptor `_sha256`) but is NOT yet
wired into the product QA receipt builder** — no receipt records it today.

**How the product selects a profile per data type (builder requirement d):**
each 10-s window is described once (E4.01 descriptors, ~8 frames, no GPU); the
router then applies ordered rules — (1) native high-res/high-fps
(`width≥1280` or `fps≥25`) → `factory_1080p30`; (2) high ego-motion
(`shift_mean≥28` or `temp_energy≥0.26`) → `walking_logistics`; (3) low quality
(`sharp_norm≤2500` or `luma_mean≤0.24` or `clip_frac≥0.30`) → `blurry_lowlight`;
(4) otherwise → `clear_stationary`; any descriptor outside the observed ranges →
`default` + `UNSEEN_REGIME`. `margin` reports the distance to the deciding
boundary. Thresholds were calibrated on the E4.01 window pool, **n = 730**
(from `E4.01.json`; that count includes the 60 E4.03 IndEgo-spec re-encodes).

Unit tests: **13/13 pass in 0.08 s** (`python3 -m pytest
tests/test_exp_regimes_router.py -q`; also `python3 -m unittest discover -s
tests` collects them — `unittest.TestCase` classes, paths anchored to
`__file__`; synthetic descriptors, determinism, out-of-range `clip_frac` /
`temp_energy` fallback, schema conformance; no video, no network).

## Verdicts

| id | verdict | headline |
|----|---------|----------|
| E4.01 | adopt | content separators exist but sharpness overlaps; container is by construction |
| E4.02 | drop | no regime split beats global under the same folds (best +0.125 [−0.10,0.35]) |
| E4.03 | drop | IndEgo-spec re-encode shrinks only 6/18 content gaps; frozen OOD AUROC 0.415 (n=60, n_bad=9, exploratory) |
| E4.04 | adopt | 5 profiles + deterministic router + 13/13 tests (receipt fragment not yet wired) |

## Limitations

- OOD set is small (n = 60, n_bad = 9) and window-level CIs there would be
  optimistic; no OOD AUROC is claimed as a quality measure, and descriptor
  differences are not claimed as a transfer improvement.
- The E4.02 global AUROC (feat26 0.712) uses this experiment's fixed C=0.01 and
  is not the same setting as the documented 0.810 baseline (CI overlaps it).
- `ood_frames.npz` is missing, so the EG2 cluster-distance view was not rerun.
- Ego10K-300 and OOD-2-100 are unlabelled: descriptor distributions only.
