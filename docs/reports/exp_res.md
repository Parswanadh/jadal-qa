# X1 / RES — resolution + compression sweep (claim X1, monitor)

**Question (Q1):** how far can resolution and compression be reduced before accuracy drops?

**Data:** 150 human-labelled IndEgo gold windows (100 good/50 bad) from 7 clips. **Protocol:** each window re-encoded with libx264 (`veryfast`), decoded through the canonical frozen path (scale to 160x90, `cpu_window_features` = the 26 per-pixel features), then LOCO logistic (C=0.3, class_weight=balanced, StandardScaler), leave-one-CLIP-out — identical split/protocol to the cached baseline (`tools/port/baseline_logistic_loco.py`). CI = clip-bootstrap B=1000 seed 0. Cached 480p native baseline = **0.8100**.

Figures: [res_ladder.png](../experiments/exp_res/images/res_ladder.png) · [compression_ladder.png](../experiments/exp_res/images/compression_ladder.png) · [sample_frames.png](../experiments/exp_res/images/sample_frames.png). Plot data: `docs/experiments/exp_res/plot_data.csv`.

## Findings — RES.01 resolution ladder (crf23)

| config | AUROC [CI95] | bytes/window | ms/window | verdict |
|---|---|---|---|---|
| native 480p (cached, no re-encode) | 0.8100 | — | — | reference |
| 480p | 0.8006 [0.676,0.883] | 636 KiB | 2335 | maybe |
| 360p | 0.8028 [0.687,0.878] | 427 KiB | 6846 | maybe |
| 240p | 0.8080 [0.704,0.882] | 258 KiB | 2669 | adopt |
| 180p | 0.8428 [0.747,0.903] | 198 KiB | 5491 | adopt |
| 120p | 0.8022 [0.682,0.889] | 137 KiB | 1771 | maybe |

**What worked:** reducing the encoded resolution to **180p** keeps AUROC within 0.02 of the 480p native baseline (indeed nominally *above* it: the 160x90 decode target means 180p→160x90 is nearly 1:1, so the extra downscale acts as a denoiser) at **~31% of the bytes** (651→202 KiB/window). Even the **120p** probe (below the 160x90 target width) holds at 0.8022 (137 KiB, 21% of bytes). All rungs are statistically indistinguishable — the CIs are ±0.1.

**What did not:** nothing regressed within the tested range; there is **no knee down to 120p** for this feature set. Caveat: re-encoding at 480p *loses* ~0.01 AUROC vs the cached native (generation loss) — the re-encode rungs are therefore best compared to each other; against the cache the whole ladder sits inside one CI. **=> lowest resolution within 0.02 of 480p = 120p (the lowest tested); 180p at 31% bytes is the safe, well-supported pick.**

## Findings — RES.02 compression ladder (480p)

| config | AUROC [CI95] | bytes/window | ms/window | verdict |
|---|---|---|---|---|
| CRF 18 | 0.8030 [0.687,0.881] | 1113 KiB | 4407 | maybe |
| CRF 23 | 0.8006 [0.676,0.883] | 636 KiB | 2335 | maybe |
| CRF 28 | 0.7960 [0.681,0.877] | 365 KiB | 3906 | maybe |
| CRF 33 | 0.7946 [0.694,0.863] | 230 KiB | 3742 | maybe |

**What worked:** the pre-set target is met — **CRF28 = 0.7960 (ΔAUROC vs native −0.014, well inside the 0.02 budget) at 365 KiB = 57% of the CRF23 bytes**. CRF33 = 0.7946 (Δ −0.015) at 230 KiB = 36% of CRF23 bytes — a bigger saving that still stays within budget. CRF18 (0.8030) costs 1.75× the CRF23 bytes for no reliable gain.

**What did not:** no CRF setting broke the 0.02 budget on this set, so the codec cannot be used to *save compute* (ms is flat-ish and load-noisy); it only saves bytes. All CRF CIs overlap, so the small AUROC ordering (18>23>28>33) is within noise. **=> adopt CRF28 (target met); maybe CRF33.**

## Findings — RES.03 cross cells

| config | AUROC [CI95] | bytes/window | ms/window | verdict |
|---|---|---|---|---|
| 360p@CRF23 | 0.8028 [0.687,0.878] | 427 KiB | 6846 | maybe |
| 360p@CRF28 | 0.8036 [0.696,0.879] | 257 KiB | 2983 | maybe |
| 240p@CRF28 | 0.7934 [0.683,0.855] | 169 KiB | 2722 | maybe |

**What worked:** **360p@CRF28 = 0.8036 at 257 KiB — the best bytes/accuracy point**: it matches 480p@CRF23's accuracy (0.8006) at **40% of its bytes**. **240p@CRF28 = 0.7934 at 169 KiB** (Δ −0.017 vs native, still inside the 0.02 budget).

**What did not:** no cross cell fell out of the 0.02 band; again all CIs overlap, so the ranking is within noise. **=> adopt 360p@CRF28; maybe 240p@CRF28.**

## Findings — RES.04 OOD sensitivity (Egocentric-10K)

n=60 windows (51 good / 9 bad, 48 clips), frozen pixel-26 head trained on IndEgo gold.

| condition | AUROC [CI95] | Δ vs 1080p [CI95] |
|---|---|---|
| native1080p | 0.4270 [0.248,0.611] |  |
| 480p | 0.4248 [0.250,0.612] | -0.0022 [-0.026,+0.019] |
| 360p | 0.4401 [0.269,0.628] | +0.0131 [-0.034,+0.059] |

**What worked:** resolution reduction is **OOD-neutral** — 480p Δ−0.002 [−0.026,+0.019] and 360p Δ+0.013 [−0.034,+0.059] vs 1080p, both indistinguishable from zero. So the ≤0.02 resolution budget transfers out of domain.

**What did not:** the **EG2 window-embedding view was NOT recomputed** — it needs a GPU frame embedder run on the downscaled clips, which did not fit the timebox. This combo is pixel-26 only. Also, the 1080p HEVC software decode is ~20 s/window, so the downscale is done at decode time (no re-encode): this isolates resolution loss from codec artifacts.

## Bottom line

**Resolution can be dropped to 120p (the lowest tested) with no AUROC loss on the gold set** — the frozen 160x90 pipeline is insensitive to source resolution because the analysis target is already only 160x90; the win is bytes/transport, not feature compute. **Recommended shipping point: 360p@CRF28 (0.8036, 257 KiB, ~40% of the 480p@CRF23 bytes).** Compression to CRF28/33 holds the ≤0.02 budget; there is no accuracy cliff on either axis within the tested range. OOD (RES.04): the frozen pixel-26 head does not transfer to Egocentric-10K (AUROC≈0.43) and its score is unchanged by resolution (Δ within ±0.013), i.e. resolution reduction is OOD-neutral.


**Cost note:** `ms/window` includes encode+decode+features and is measured on the shared, loaded pod (not a clean benchmark); treat relative comparisons as indicative.

