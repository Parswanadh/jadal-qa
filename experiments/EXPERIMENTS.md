# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| RES.01 | RES | done | adopt | Resolution ladder: IndEgo pixel-26 at 480/360/240/180/120p | IndEgo gold 150 windows human labels 100good/50bad / 480p 10fps H264 in-domain |
| RES.02 | RES | done | adopt | Compression ladder: H.264 CRF 18/23/28/33 at 480p | IndEgo gold 150 windows human labels / 480p 10fps H264 in-domain |
| RES.03 | RES | done | adopt | Cross cells 360p@CRF23/28, 240p@CRF28 | IndEgo gold 150 windows human labels / 480p 10fps H264 in-domain |
| RES.04 | RES | done | maybe | OOD resolution sensitivity: frozen pixel-26 on Ego10K 1080p/480p/360p | Egocentric-10K OOD 60 windows (51 good/9 bad) human labels / 1080p30 HEVC OOD |

## RES.01 — Resolution ladder: IndEgo pixel-26 at 480/360/240/180/120p
- **Why:** how far can encode resolution drop before the frozen 26-feature LOCO AUROC falls
- **Hypothesis:** AUROC stays within 0.02 of 480p down to 120p
- **Data / label source:** IndEgo gold 150 windows human labels 100good/50bad
- **Regime:** 480p 10fps H264 in-domain
- **Method:** re-encode each window libx264 crf23 -> decode 160x90 -> cpu_window_features 26 -> LOCO C=0.3 clip-bootstrap CI
- **Owner / started / finished:** monitor / 2026-10-10 16:48 / 2026-10-10 16:49
- **Verdict:** adopt — Encoded resolution can drop to 180p (0.8428, 198 KiB) and 120p (0.8022, 137 KiB) with no AUROC loss vs 0.810 native; no knee down to 120p; all rungs within 0.02 (one CI).
- **Decision:** 
- **Cost:** encode~0.6s+dec0.2s+feat0.5s per 10s window (shared pod)
- **Results:** eval/results/exp_res/RES.01.json

## RES.02 — Compression ladder: H.264 CRF 18/23/28/33 at 480p
- **Why:** CRF28 gives ~50% bytes with dAUROC<0.02
- **Hypothesis:** CRF28 within 0.02 and ~57% bytes; CRF33 36%
- **Data / label source:** IndEgo gold 150 windows human labels
- **Regime:** 480p 10fps H264 in-domain
- **Method:** re-encode crf ladder -> 160x90 -> 26 feats -> LOCO
- **Owner / started / finished:** monitor / 2026-10-10 16:48 / 2026-10-10 16:49
- **Verdict:** adopt — Target met: CRF28=0.7960 (dAUROC -0.014) at 57% of CRF23 bytes; CRF33=0.7946 (-0.015) at 36%; no CRF broke 0.02.
- **Decision:** 
- **Cost:** flat ms; bytes 1.14MB->230KB
- **Results:** eval/results/exp_res/RES.02.json

## RES.03 — Cross cells 360p@CRF23/28, 240p@CRF28
- **Why:** combine best of both axes
- **Hypothesis:** 360p@CRF28 matches 480p accuracy at 40% bytes
- **Data / label source:** IndEgo gold 150 windows human labels
- **Regime:** 480p 10fps H264 in-domain
- **Method:** re-encode -> 160x90 -> 26 feats -> LOCO
- **Owner / started / finished:** monitor / 2026-10-10 16:48 / 2026-10-10 16:49
- **Verdict:** adopt — 360p@CRF28=0.8036 at 257 KiB (best bytes/accuracy, 40% of 480p@CRF23 bytes); 240p@CRF28=0.7934 at 169 KiB inside budget.
- **Decision:** 
- **Cost:** 257 KiB/window @ ~3.0 s/window
- **Results:** eval/results/exp_res/RES.03.json

## RES.04 — OOD resolution sensitivity: frozen pixel-26 on Ego10K 1080p/480p/360p
- **Why:** resolution reduction is OOD-neutral
- **Hypothesis:** delta within +/-0.013 of 1080p
- **Data / label source:** Egocentric-10K OOD 60 windows (51 good/9 bad) human labels
- **Regime:** 1080p30 HEVC OOD
- **Method:** NVDEC decode -> one-pass downscale -> frozen pixel-26 head -> AUROC + bootstrap CI
- **Owner / started / finished:** monitor / 2026-10-10 16:48 / 2026-10-10 16:49
- **Verdict:** maybe — Pixel-26 OOD AUROC~0.43 (IndEgo head does not transfer) and is resolution-neutral: 480p d-0.002 [-0.026,+0.019], 360p d+0.013 [-0.034,+0.059]. EG2 view NOT recomputed.
- **Decision:** 
- **Cost:** NVDEC; 60 windows
- **Results:** eval/results/exp_res/RES.04.json
