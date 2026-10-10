# EG2 frame-head + transfer verification (claim LB1-heads)

Verifier: LEAD B verification worker, branch `b/verify-heads` (worktree off `origin/main`).
Source of claims: `/workspace/jadal_pod_bundle/docs/FACTS.md` (sha256 `8031ba05d3832de5…`),
sections **UPDATE 2026-10-10 17:40 IST** and **17:50 IST** (quoted verbatim in the JSON).
Full machine-readable record: `eval/results/pod_verify/frame_heads_verification.json`.

Legend — **REPRODUCED**: recomputed on this pod from cached inputs/features (or metric
re-derived from stored per-window scores). **NOT REPRODUCED**: available artifacts or my
recomputation contradict the value. **CANNOT VERIFY**: recomputation impossible here
(missing inputs), no contradiction found.

| # | FACTS claim (verbatim, abridged) | Published | Recomputed | n | Verdict | Basis |
|---|---|---|---|---|---|---|
| c1 | "EmbeddingGemma-2 frame heads pooled over every 0.5 s frame: AUROC 0.808 (0.70-0.86)" | 0.808 (0.70–0.86) | **0.8078** [0.6995, 0.8616] (proper cluster boot [0.679, 0.869]) | 150 (100/50, 7 clips) | **REPRODUCED** | independent recompute (8k heads) |
| c2 | "EG2 human-only head 0.816 (0.71-0.89)" | 0.816 (0.71–0.89) | **0.8164** [0.7101, 0.8924] (proper [0.678, 0.902]) | 150 | **REPRODUCED** | independent recompute — but this arm is a 768-d *window-mean embedding* logistic, **not a frame head** |
| c3 | "pretrained-on-AI-labelled-frames pooled AUROC 0.79; human-only head 0.82; combined 0.83" | 0.79 / 0.82 / 0.83 | **0.7914** [0.688, 0.863] / **0.8164** [0.710, 0.892] / **0.8286** [0.735, 0.900] | 150 | **REPRODUCED** | independent recompute (10k heads) |
| c4 | "hands-head 0.67 (0.52-0.81)" | 0.67 (0.52–0.81) | **0.6632** [0.5258, 0.8039] (forced published config 0.6484) | 150 | **REPRODUCED** | independent CPU+GPU re-run; config selection differs (hidden 64 vs 256), AUROC within 0.02 |
| c5 | "'work' head 0.41 (no transfer)" | 0.41 | **0.4942** [0.4025, 0.6447] auto; **0.5176** [0.405, 0.642] forced config; GPU identical | 150 | **NOT REPRODUCED** (value) — "no transfer" **REPRODUCED** (all CIs span 0.5) | independent CPU+GPU re-run + forced-config run |
| c6 | "Harness arms … 0.47-0.59 with intervals touching 0.5" | 0.47–0.59 | artifact n=150 arms **0.468–0.594** (8 arms) | 150 (some 79) | **REPRODUCED** | artifact transcription; no per-window scores → CIs not recomputable. A-full CI [0.509, 0.641] does not touch 0.5 |
| c7 | "EmbeddingGemma-2 head 0.60" (earlier arm) | 0.60 | final arm E **0.496** [0.335, 0.639]; superseded 79-window run **0.596** [0.505, 0.817] | 150 vs 79 | **NOT REPRODUCED** (stale) | artifact transcription (both runs in git history) |
| c8 | "1D-conv on 8 per-frame series 0.782 (0.63-0.86)" | 0.782 (0.63–0.86) | **0.78243** [0.6273, 0.8598] | **137** (100/37) | **REPRODUCED** | independent retraining from cached series |
| c9 | "boosted trees 0.758" | 0.758 | **0.75784** [0.5731, 0.8739] | **137** | **REPRODUCED** | metric re-derived from stored OOF scores |
| c10 | "physics-only logistic 0.715" | 0.715 | **0.71541** [0.5726, 0.8104] | **137** | **REPRODUCED** | independent recompute from cached features |
| c11 | "on the 150 human windows" (n + label source) | 150 human | 150 = 100 good/50 bad, 7 clips; labels match the restored `manifest_gold.json` **150/150** (clip, label, reason); declared labels_sha256 `43ab44e5…` | 150 | **REPRODUCED** | independent label cross-check |

## Where the numbers live (provenance)

* The EG2 frame-head transfer artifacts are **not on `origin/main`**: `eval/results/frames10k/`
  exists only on the unmerged `origin/feat/frames10k` branch (`segments.npz` blob `fd1903bd`,
  `transfer.json` blob `d9b91411`, 8k-run `transfer.json` at `62e09e3` blob `eaaaff3b`,
  heads blobs `80a393ef` / `e34dc530`). `docs/FACTS.md` is likewise absent from `origin/main`.
* Pod caches only hold the **10k AI-labelled factory-frame** embeddings/heads
  (`/workspace/jadal_pod_bundle/caches/frames10k/egocentric_10k_frame_embeddings/`), not the
  150-window IndEgo per-frame embeddings.
* GPU multi-head M2 JSONs live on `origin/feat/gpu-heads` (blobs `5ae88a24`, `ce186c9b`);
  harness arms on `origin/feat/spectral-harness` (blob `cedfa527`).

## Commands

```bash
# frame-head transfer (8k + 10k arms), from the unmerged producing branch:
bash eval/pod_verify/fetch_frames10k_artifacts.sh /tmp/f10k
OMP_NUM_THREADS=2 python3 eval/pod_verify/recompute_frame_heads.py /tmp/f10k

# perpixel trio (conv1d / HGB / physics), all inputs on origin/main:
OMP_NUM_THREADS=2 python3 eval/pod_verify/recompute_perpixel.py

# M2 GPU multi-head: extract origin/feat/gpu-heads:{jadal/gpuheads/*,eval/gpu_heads/run_m2.py}
# + origin/feat/frames-spectral:{egocentric_10k.npz,human_windows_raw.npz}, path-patch the two
# constants, then: python3 run_m2.py --device {cpu,cuda} --epochs 30 --seed 0 --target {hands,work}
```

## Findings that change how the claims should be read

1. **Two runs behind two numbers.** 17:40's 0.808 is the **8k**-head run; 17:50's 0.79 is the
   **10k**-head run. Same nominal protocol, different training sets; neither line says which.
2. **"Human-only head" is not a frame head.** The 0.816 arm is a logistic on the 768-d
   L2-normalised window-mean EG2 embedding; it uses no AI frame labels and is identical in the
   8k and 10k runs.
3. **The perpixel trio is n=137**, not 150 (activity reasons only), despite the FACTS header
   "on the 150 human windows" and `compare.json`'s own "150 windows" protocol line.
4. **Bootstrap asymmetry.** `analyze_transfer.py`'s clip bootstrap resamples the 7 clips but
   takes the union of their windows (no duplication) → slightly narrow CIs
   (e.g. 0.8164: [0.710, 0.892] vs proper [0.678, 0.902]). The perpixel scripts use the proper
   duplicating variant.
5. **The exact M2 "work" 0.41 is unpinned.** CPU and GPU re-runs both give ≈0.49 (auto) / ≈0.52
   (forced published config); the producer reads training features from another worker's live
   worktree path and that file was regenerated after the run (`52ac9c99 → f91385e2`), so no
   committed input pins 0.41. The "no transfer" conclusion holds either way.
6. **0.60 is stale.** The final 150-window harness EG2 head scores 0.496; 0.60 matches the
   superseded 79-window partial (0.596). `legacy_models.md` L1.03 already flags the conflict.
7. **Adjacent arm not in FACTS:** `origin/feat/frames-spectral`'s own window transfer
   (`full_pipeline`) is **0.7472** [0.6107, 0.8241], n=150 — outside the 0.47–0.59 harness
   sentence, and the registry backfill wrongly says its window numbers were "not extracted".

No secrets were read, printed, or committed. `labels/manifest_gold.json` was only read.
