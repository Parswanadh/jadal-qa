# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E8.01 | E8 | done | maybe | E8.01 re-score 1,546 Egocentric-10K windows with the frozen two-view recipe + profile router | queues/manifest_scale.json: 1546 windows / 58 clips / 4.404  / factory Egocentric-10K, 1080p, 30 fps, HEVC, handheld egocentric manufacturing |

## E8.01 — E8.01 re-score 1,546 Egocentric-10K windows with the frozen two-view recipe + profile router
- **Why:** B2/B4 PoC needs one contract JSONL over the whole 4.404 h scale corpus: per-window verdict, evidence op-threshold, features digest and content-hashed receipt, produced by the frozen in-domain recipe and the deterministic router
- **Hypothesis:** The frozen ensemble_v1 (EG2+pixel, equal-weight) + router produce a deterministic, reproducible contract file for all 1,546 windows; the router routes every 1920x1080/30 window to factory_1080p30; verdict shares are descriptive only (IndEgo-trained models did not transfer out-of-domain, 0.4-0.53 AUROC)
- **Data / label source:** queues/manifest_scale.json: 1546 windows / 58 clips / 4.404 h, 1080p30 HEVC; label_source none (no human labels)
- **Regime:** factory Egocentric-10K, 1080p, 30 fps, HEVC, handheld egocentric manufacturing
- **Method:** per window: ffmpeg/NVDEC decode -> (a) 160x90 rgb24 pixel26 features at native fps, (b) 512-side frames every 0.5 s for EmbeddingGemma-2 mean embedding, (c) 8 native frames for E4.01 regime descriptors; frozen logistic heads + equal-weight probability average; deterministic profile router choose_profile(descriptors); contract.jsonl with receipt = sha256(canonical JSON)
- **Owner / started / finished:** experimenter-e8-rescore / 2026-10-10 15:42 / 2026-10-10 16:27
- **Verdict:** maybe — n=1546/1546 windows (58 clips, 4.404 h) scored, 0 decode failures, label_source none, protocol descriptive re-score + 6-window decode-path control. Verdicts (descriptive, not accuracy): accept 296 (19.15%), reject 83 (5.37%), review 1167 (75.49%); accept share 95% CI [0.136,0.254] by clip bootstrap over 58 clips (window Wilson [0.173,0.212] is anti-conservative). Profile: factory_1080p30 1540, default 6 (UNSEEN_REGIME: 3 extreme ego-motion shift_mean 245-642, 3 overexposed luma_mean 0.66-0.67). Cost: 0.683 windows/s, 5761 ms/window mean (p50 5195, p95 9371), wall 2263 s with 4 workers on a SHARED A40 (load ~15); decode 2354 ms, pixel26 1154 ms, descriptors 1055 ms, EG2 embed 1196 ms. 1546/1546 receipts + features_digest re-verified. Recipe ensemble_v1 (0.8534 full-OOF in-domain AUROC, 150 windows/7 clips, one labeller) did NOT transfer out-of-domain (0.4-0.53 AUROC, n_bad=9), so the accept/reject share is NOT a quality measure.
- **Decision:** Ship /workspace/artifacts/windows_rescored.jsonl as the B2 PoC contract (not committed, artifacts/ ignored); keep the frozen recipe; do not read accept/reject shares as quality; next: E5 domain adaptation on the same corpus, and re-run this runner if the frozen decode path (bit-identical) is needed for the final demo receipts.
- **Cost:** 4 workers x ~2263 s wall (37.7 min) on a shared A40; 1546 windows x (300 pixel frames @160x90 + 20 EG2 frames @512x288 + 8 native descriptor frames) = 1 ffmpeg/NVDEC pass and 1 EG2 forward per window; 0.68 windows/s, 5.76 s/window mean; no extra storage (pipes only); E3.04 every-2nd+4s policy deliberately NOT applied (see report).
- **Results:** eval/results/exp_rescore/E8.01.json
