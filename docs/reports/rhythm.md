# rhythm: hand-region residual-motion rhythm network

`rhythm` is a deterministic window classifier for the Jadal QA funnel. Given one 10-second egocentric (head-camera) factory-video window, it must support a verdict of ACCEPT, REJECT (with feature/value/op/threshold evidence) or REVIEW (uncertain, handed to a person) on the semantic question that the rest of the funnel does not answer: is the wearer actively doing hands-on work with the hands visible. Physical-quality problems (too dark, blocked, over-exposed, frozen, blurred, duplicate, undecodable) are handled upstream by exact threshold rules, not by this model. What the model consumes is not raw pixels but a fixed-size representation `R` computed deterministically from every frame of the window: a `T x G x G x 6` grid tensor of ego-motion-compensated residual-motion and skin-chromaticity channels, plus a `T x 8` matrix of whole-frame (global) per-frame channels. The model is small, CPU-runnable, and intended to be reproducible byte-for-byte from the same input; it is not a hand detector, not a generative model, and not a general video-understanding model.

> **Skeleton status.** This document is a skeleton to be filled in by the engineer who runs the experiment. Every cell that says `PENDING` is unmeasured: no command in this worktree has produced it yet. `PENDING` must be replaced by a value read out of a JSON file under `eval/results/rhythm/`, together with the command that regenerated that JSON. Nothing in the Results, Ablations, Per-clip results, Cost or Determinism sections may be filled from a hand-over document, a slide, a chat message or another branch's report.

## Method

### Representation R

`R` is defined by `jadal/rhythm/features.py` and is the only thing the network sees. It is computed per window, from every frame in the window, with no random numbers anywhere in the module.

**Frame sampling and analysis resolution.** The source frame rate is read from the container. Frames are then selected on a uniform time grid inside `[start_s, end_s)` so that the effective rate is `TARGET_FPS = 10.0`, with duplicate indices removed and a two-frame fallback for degenerate windows (`sample_indices`). This is what makes source clips of differing frame rates comparable before any temporal channel is computed, and it is why the fixed 5-frame temporal-variance box below is comparable across corpora. Each selected frame is area-downscaled to `ANALYSIS_W x ANALYSIS_H = 160 x 90` (`cv2.INTER_AREA`); motion channels use the float32 grayscale frame, and the skin channel uses the BGR uint8 frame. `DEFAULT_T = 100` frames, i.e. 10 s at the target rate; the actual `T` per window is whatever the sampler returns and is stored alongside the tensor.

**Ego-motion compensation and the residual.** For each consecutive frame pair `(t-1, t)`, phase correlation (`cv2.phaseCorrelate`) on Hann-windowed grayscale frames returns the global translation `(dx, dy)` that maps frame `t-1` onto frame `t`. Frame `t-1` is warped by that translation (`cv2.warpAffine`, bilinear interpolation, border replicate) and the residual is `abs(I_t - warp(I_(t-1)))`. The dense Farneback flow field between the same pair has the same global translation subtracted from it component-wise; what remains is the residual flow, and its magnitude and local direction coherence are two of the grid channels. Passing `ego_compensate=False` disables both the warp and the subtraction, which is how the ablation is produced.

In plain language: *ego-motion* is the motion of the camera itself, which here is the motion of the wearer's head and body. *Ego-motion compensation* means measuring how far the whole image shifted between two frames and sliding the earlier frame back by that amount before comparing them. The *residual* is what is left of the frame-to-frame difference afterwards, and it is mostly the scene moving relative to the camera: hands, tools, parts on a bench. Without compensation, a wearer who simply turns their head produces large frame differences everywhere, which looks like activity but is not.

**Grid channels.** The six per-frame channels below are pooled onto a `G x G` grid by block means (`cv2.resize` with `INTER_AREA`), giving `X` of shape `(T, G, G, 6)`. `G = 6` by default. The grid is a fixed partition of the frame: no hand detector, no region proposal, no tracking is used anywhere in the representation.

| # | channel     | definition in `jadal/rhythm/features.py`                                                                   | spatial?                 |
|---|-------------|------------------------------------------------------------------------------------------------------------|--------------------------|
| 0 | `res_mean`  | cell mean of the residual `abs(I_t - warp(I_(t-1)))`, divided by 255                                       | yes, per cell            |
| 1 | `res_p90`   | 90th percentile of the whole-frame residual, divided by 255, repeated into every cell                      | no, one scalar per frame |
| 2 | `skin_frac` | fraction of pixels in the cell matching the YCrCb rule `Cr in [133, 173]`, `Cb in [77, 127]`               | yes, per cell            |
| 3 | `tvar`      | temporal variance of the unwarped gray frame over a fixed symmetric 5-frame box, divided by 255            | yes, per cell            |
| 4 | `flow_mag`  | cell mean of the residual flow magnitude `abs(flow - global_translation)`, divided by `ANALYSIS_W = 160`   | yes, per cell            |
| 5 | `flow_coh`  | resultant length of residual flow over a local 9x9 box: `hypot(box(fx), box(fy)) / (box(flow_mag) + 1e-6)` | yes, per cell            |

Three details matter for reading later ablations. `res_p90` carries no spatial information: it is one number per frame written into all `G x G` cells, so it survives even when the grid is removed. `tvar` is computed on the unwarped frames, so it is not ego-motion compensated. At `t = 0` the residual, flow magnitude, flow coherence and shift channels are zero by construction, since there is no previous frame to compare against.

**Global channels.** The eight per-frame channels below are appended as `g` of shape `(T, 8)`, with definitions identical to the frozen perpixel series so that the model is a strict superset of that representation.

| # | channel      | definition (identical to the frozen perpixel series)                                                                    |
|---|--------------|-------------------------------------------------------------------------------------------------------------------------|
| 0 | `luma_mean`  | frame mean divided by 255                                                                                               |
| 1 | `lap_var`    | variance of the Laplacian of the frame                                                                                  |
| 2 | `diff_e`     | mean `abs(gray_t - gray_(t-1))` divided by 255; 0 at `t = 0`                                                            |
| 3 | `skin_frac`  | frame-level skin fraction (same YCrCb rule as grid channel 2)                                                           |
| 4 | `shift_px`   | phase-correlation shift magnitude `hypot(dx, dy)` divided by `ANALYSIS_W = 160`; 0 at `t = 0`                           |
| 5 | `under_frac` | fraction of pixels below 10                                                                                             |
| 6 | `over_frac`  | fraction of pixels above 245                                                                                            |
| 7 | `dct_hf`     | high-frequency DCT energy fraction, mean over 8x8 blocks, coefficients outside the 4x4 low corner over all coefficients |

**Shapes, caching and digest.** `X` is `(T, G, G, 6)` float32 in memory and `g` is `(T, 8)` float32. Windows are cached as compressed `.npz` with `X` and `g` stored as float16 (`save_cache`), keyed by window id and `G`. `tensor_digest` is the SHA-256 of the 1e-4-quantised float32 bytes of `X` followed by those of `g`, and is the per-window feature digest used by the receipt.

### Model

The description below follows `jadal/rhythm/model.py` as it stands in this worktree. Layer sizes quoted are the constructor defaults; the sizes actually used in a run, and the parameter count that `count_params()` reports per member and for the whole ensemble, are `PENDING`.

- **Vectorised ensemble.** The ensemble dimension `K` is carried as a leading tensor dimension: every weight is stored as `(K, ...)`, and all members are trained in one set of tensor operations by folding `K` into the batch and channel dimensions and using grouped convolutions (`groups=K*N`). This is what makes many members affordable on one GPU, and it avoids `torch.func.vmap` / `functional_call`. Inputs are `X` of shape `(K, B, T, G, G, 6)` and `g` of shape `(K, B, T, 8)`; outputs are one usable logit and two auxiliary logits per member and window.
- **Shared per-cell temporal encoder.** The same depthwise-separable dilated 1D convolution stack runs along the time axis of every grid cell, with weights shared across cells (enforced by folding the `G x G` cells into the batch dimension). The defaults are three blocks with channels `(8, 12, 16)`, kernel 5 and dilations 1/2/4; each block is a depthwise convolution, a pointwise convolution, batch normalisation computed per member over all cells and time, GELU and dropout. Sharing weights across cells is what keeps the model small and stops it from memorising "hands are always in this corner of this camera".
- **Pooling over cells.** The per-cell embeddings are reduced by mean and by max over the `G x G` cells, plus four per-cell activity statistics taken from the last encoded channel: mean, standard deviation, maximum, and concentration (maximum divided by sum). The reduction does not depend on the order in which cells are enumerated, so the model is invariant to camera framing and to grid relabelling. There is no attention pooling and no convolution across cells: cells are independent until this reduction, so the model cannot represent "this cell is next to that cell".
- **Global-channel branch.** The `(T, 8)` global series pass through a separate encoder of the same kind (default channels `(8, 8)`) and are reduced by mean and max over time, then concatenated with the pooled grid features. With the defaults the concatenated feature vector is `chans[-1]*2 + glob_chans[-1]*2 + n_cell_stats = 48` wide. The head is a per-member layer normalisation, dropout and two per-member linear layers (default hidden width 32) producing one usable logit plus two auxiliary logits for the reject reasons `NO_ACTIVE_WORK` and `HANDS_NOT_VISIBLE`, trained with a masked loss on the windows where the reason is known. The auxiliary logits are an inductive bias, not a claim that the model predicts reasons in general.
- **Selection and training.** Members differ by seed, hidden sizes, weight decay, learning rate and augmentation strength. The final predictor is the mean over the top-M members selected by inner-LOCO AUROC computed only inside the training clips of the outer fold. Loss is class-weighted binary cross-entropy with label smoothing, AdamW with a cosine schedule and early stopping on the inner protocol; fixed seeds and deterministic algorithm settings are required. The member count, hidden sizes, learning rates and augmentation strengths actually used are `PENDING`.

## Protocol

**Outer loop: leave-one-clip-out (LOCO) over the 7 clips.** For each fold, train on the windows of 6 clips and predict the held-out clip. The 7 held-out prediction sets are then pooled for the headline metrics. No model, hyperparameter, threshold or calibration may ever be chosen using the held-out clip of any fold.

**Inner loop: LOCO inside the training clips.** Within each outer fold, leave-one-clip-out over the 6 training clips is used for hyperparameter selection, ensemble-member selection, calibration (Platt scaling as specified) and the choice of the operating threshold. The threshold is chosen inside the training clips only; the target false-reject rate used to pick it is `PENDING`.

**Metrics.** Pooled AUROC over the held-out predictions of all 7 folds; clip-level bootstrap 95% confidence interval with 2,000 resamples (clips, not windows, are resampled with replacement, since windows inside a clip are correlated) — the exact resampling implementation is `PENDING` confirmation against `eval/rhythm/evaluate.py`; balanced accuracy at the inner-chosen threshold (mean of sensitivity and specificity); per-clip AUROC; recall per bad reason at that threshold; the good-window false-reject rate; and the coverage-accuracy curve, which sorts windows by ensemble margin or member disagreement and reports balanced accuracy as a function of the fraction of windows the system is willing to decide, with the remainder abstaining to REVIEW.

**Evaluation set and what that implies.** The evaluation set is n = 150 ten-second windows, 100 labelled good and 50 labelled bad, produced by one human labeller, drawn from 7 clips of IndEgo footage covering roughly 3 wearers. Because there is one labeller, 7 clips and about 3 wearers, every interval reported here will be wide, windows within a clip are not independent, and the intervals describe this sample rather than a population. No certified error bound, and in particular no statistically valid false-reject guarantee, is claimed; observed rates with cluster-aware intervals are reported instead, and the report must say so wherever a rate appears.

## Results

All cells are `PENDING`. Each row must be filled from a result JSON under `eval/results/rhythm/` that a documented command regenerates. The published figures for these methods in the project hand-over were measured under other folds by other code; they are not this experiment's results and must not be pasted into this table.

| method                               | AUROC (95% CI) | protocol | n       | label source | status  |
|--------------------------------------|----------------|----------|---------|--------------|---------|
| rhythm ensemble (spatial grid)       | PENDING        | PENDING  | PENDING | PENDING      | PENDING |
| ablation (d): global channels only   | PENDING        | PENDING  | PENDING | PENDING      | PENDING |
| 26 GPU per-pixel features + logistic | PENDING        | PENDING  | PENDING | PENDING      | PENDING |
| single feature `flow_act`            | PENDING        | PENDING  | PENDING | PENDING      | PENDING |
| frozen 1D-conv on 8 global series    | PENDING        | PENDING  | PENDING | PENDING      | PENDING |

### Baselines reproduced by this work

"Reproduced" means re-run by this work's code under the folds described in Protocol. If a re-run disagrees with a previously published figure, report the re-run and record the disagreement explicitly; do not average the two and do not substitute the published figure.

| baseline reproduced by this work     | AUROC (95% CI) | protocol | n       | label source | status  |
|--------------------------------------|----------------|----------|---------|--------------|---------|
| 26 GPU per-pixel features + logistic | PENDING        | PENDING  | PENDING | PENDING      | PENDING |
| single feature `flow_act`            | PENDING        | PENDING  | PENDING | PENDING      | PENDING |
| frozen 1D-conv on 8 global series    | PENDING        | PENDING  | PENDING | PENDING      | PENDING |

## Ablations

All cells are `PENDING`. Ablations run on cached tensors under the same outer folds as the headline result, with the same inner protocol for any selection. "delta vs full model" is the difference from the full model measured under identical folds; a delta without an interval is not reportable on its own.

| ablation                | setting                    | AUROC (95% CI) | delta vs full model | status  |
|-------------------------|----------------------------|----------------|---------------------|---------|
| ego-motion compensation | on (full model)            | PENDING        | reference           | PENDING |
| ego-motion compensation | off (`--no-ego`)           | PENDING        | PENDING             | PENDING |
| skin channel            | included (full model)      | PENDING        | reference           | PENDING |
| skin channel            | removed                    | PENDING        | PENDING             | PENDING |
| grid size `G`           | 1 (global only)            | PENDING        | PENDING             | PENDING |
| grid size `G`           | 4                          | PENDING        | PENDING             | PENDING |
| grid size `G`           | 6 (default)                | PENDING        | reference           | PENDING |
| grid size `G`           | 8                          | PENDING        | PENDING             | PENDING |
| augmentation            | on (full model)            | PENDING        | reference           | PENDING |
| augmentation            | off                        | PENDING        | PENDING             | PENDING |
| ensemble                | top-M members (full model) | PENDING        | reference           | PENDING |
| ensemble                | single model               | PENDING        | PENDING             | PENDING |
| synthetic negatives     | included (full model)      | PENDING        | reference           | PENDING |
| synthetic negatives     | excluded                   | PENDING        | PENDING             | PENDING |

## Per-clip results

All cells are `PENDING`. Per-clip AUROC is the diagnostic that matters most here, because the pooled number can be carried by the easy clips. A clip at chance is a finding to report, not a number to omit.

| clip       | n windows | AUROC   | balanced accuracy at inner threshold | status  |
|------------|-----------|---------|--------------------------------------|---------|
| ecu_full   | PENDING   | PENDING | PENDING                              | PENDING |
| ecu_late   | PENDING   | PENDING | PENDING                              | PENDING |
| desk       | PENDING   | PENDING | PENDING                              | PENDING |
| pallet     | PENDING   | PENDING | PENDING                              | PENDING |
| pallet2    | PENDING   | PENDING | PENDING                              | PENDING |
| organise   | PENDING   | PENDING | PENDING                              | PENDING |
| logistics4 | PENDING   | PENDING | PENDING                              | PENDING |

## Cost

All cells are `PENDING`. Timing conditions must be recorded with the numbers: the pinned core set, the thread environment variables in force, whether a warm-up run was discarded, the number of repeats, and whether the cached tensor or the video decode is inside the timed region.

### Seconds per window on CPU (single core)

`PENDING`. To be measured with the process pinned to one core and the numeric thread environment variables set to a single thread, on the same 150 windows, reporting the median and a tail statistic.

### Seconds per window on GPU

`PENDING`. To be measured with the same windows and the same decode policy, through the project's GPU semaphore, with the GPU model and driver recorded.

### Decode included and excluded

`PENDING`. The two regimes must be reported separately. Note from reading `features.py`: `extract_window` performs two decode passes over the window (one for grayscale frames, one for BGR frames) and computes the skin mask twice. Any decode-included figure must state whether it covers both passes, and whether that duplication was removed before the measurement.

### Peak VRAM

`PENDING`. Peak VRAM for the vectorised ensemble training run, and peak memory for the inference path, recorded from the same runs that produce the timings.

| measurement                               | decode excluded | decode included | notes   |
|-------------------------------------------|-----------------|-----------------|---------|
| CPU, one pinned core, median s per window | PENDING         | PENDING         | PENDING |
| CPU, one pinned core, windows per hour    | PENDING         | PENDING         | PENDING |
| GPU, median s per window                  | PENDING         | PENDING         | PENDING |
| GPU, windows per hour                     | PENDING         | PENDING         | PENDING |
| peak VRAM (GPU)                           | n/a             | PENDING         | PENDING |
| peak RSS (CPU)                            | n/a             | PENDING         | PENDING |

## Determinism

Three claims are to be proven, and all of their results are `PENDING`:

1. Ten repeated runs of the full pipeline on the same inputs produce identical `features_digest` values and identical verdicts and reason codes for every window.
2. The same run executed with 1 thread and with 8 threads produces identical digests and identical decisions.
3. Every window on which the CPU (numpy-only reference) path and the GPU path disagree in decision is listed explicitly, with both scores. Silent tolerance is not acceptable; a difference is either explained or reported as an open defect.

| claim to be proven                                                   | evidence file | result  |
|----------------------------------------------------------------------|---------------|---------|
| 10 repeated runs produce identical `features_digest` values          | PENDING       | PENDING |
| 10 repeated runs produce identical verdicts and reason codes         | PENDING       | PENDING |
| 1 thread and 8 threads produce identical digests and decisions       | PENDING       | PENDING |
| every window where CPU and GPU decisions differ is listed explicitly | PENDING       | PENDING |

Receipts are content-hashed only. `receipt_sha256` is a SHA-256 over canonical JSON, which detects accidental change; it applies no cryptographic signature, carries no key, and gives no authentication, no provenance guarantee and no protection against anyone who can rewrite the JSON and recompute the hash. Determinism means the same input yields the same bytes here; it is not a claim about tamper resistance.

## Honest limits

- **One labeller.** Every label in the evaluation set comes from a single human. There is no second annotator, so there is no inter-annotator agreement, no adjudication and no way to separate model error from labeller error. A separate out-of-fold audit elsewhere in this project flagged some windows as possibly mislabelled; labels are never edited, and this report does not correct for label noise. Any variant that down-weights those windows must be reported as its own row with its own protocol.
- **7 clips and about 3 wearers.** The evaluation set spans 7 clips from roughly 3 wearers, so windows within a clip share a scene, a camera and a person, and the effective sample size is far below 150. Every confidence interval here is wide, the clip-bootstrap interval describes these 7 clips rather than a population of factories, and a pooled number can be dominated by one or two clips. This is why the per-clip table exists and why no certified error bound is claimed.
- **IndEgo footage is not Awign footage.** All labelled evaluation data is IndEgo (Fraunhofer IPK) egocentric video, a public research corpus. Awign's own factory footage differs in camera, mounting, lighting, resolution, frame rate and work content. Nothing here establishes accuracy on Awign footage, and any number from unlabelled Egocentric-10K video is a rate on that corpus, not a transfer result.
- **No Raspberry Pi measurement was made.** No timing, memory or accuracy figure in this report was measured on Raspberry Pi hardware, and nothing here should be read as a claim that this model runs on a Raspberry Pi. Any edge-feasibility statement is about the machine that was actually measured, named alongside the number.
- **The model is not a hand detector.** Nothing in the representation or the model locates hands. The grid is a fixed partition of the frame, and any "hand region" language is a description of the motion signal, not of a detected object. A window where the hands are out of frame can still contain hand-like residual motion from tools, parts or other people.
- **Skin is a chromaticity rule.** `skin_frac` is a fixed YCrCb threshold box, not a skin classifier. Wood, cardboard, packaging and bare metal share that chromaticity range under factory lighting, and the channel will fire on them. It is included because it is cheap and weakly informative, and its ablation is reported rather than assumed.
- **Ego-motion is translation-only.** Compensation estimates a single global translation per frame pair by phase correlation. Rotation, scale change, rolling shutter and parallax are not modelled, so the residual still contains head rotation and depth-induced motion, and the model may read those as activity. The project's own prior evidence about whether global-motion compensation helps is contradictory, so it is measured here under this representation and reported either way.
- **Synthetic negatives are tensor-level approximations.** The synthetic bad windows are derived by transforming cached tensors of good windows, not by re-rendering degraded video. They are negatives by construction, which makes them useful for training and for the quality side of the problem, but they are not pixel-exact degradations and their difficulty does not match real degraded footage. Any result that depends on them must say so.
- **Scores on unlabelled video are descriptive rates, not accuracy.** On unlabelled corpora there is no ground truth, so the only reportable quantities are descriptive rates of the model's own outputs (accept rate, reject rate, reason mix, coverage). Those rates cannot be converted into accuracy, precision or recall, and they must not be presented as if they were.
- **The architecture has no spatial mixing.** Grid cells are encoded independently with shared weights and only then reduced by mean+max pooling plus a few activity statistics. The model therefore cannot express "this cell is adjacent to that cell", and it cannot track a hand moving from one cell to the next except through those pooled statistics. This is a deliberate size and speed trade-off, and it bounds what a negative or positive result here can be attributed to.
- **Uncertainty is stated, not certified.** With one labeller, 7 clips and correlated windows, the honest output of this work is a set of observed rates with cluster-aware intervals and an explicit negative result if the model does not beat the single residual-flow scalar. A result that fails to beat that scalar is a valid outcome of this experiment and is reported as such.

## Reproduction

Every number in this report must come from a JSON file under `eval/results/rhythm/` that a documented command regenerates. A number without such a file is not reportable. At the time of writing `eval/results/rhythm/` is empty, `eval/rhythm/` contains `extract.py` only, and `jadal/rhythm/` contains `features.py` and `model.py`; `train.py`, `evaluate.py` and the ablation driver are not present yet. The file list is a moving target while the worktree is being filled in, so re-check it when this skeleton is completed.

Intended commands, all `PENDING` confirmation:

```bash
# Representation R -> cache. Documented form in the module docstring of eval/rhythm/extract.py.
python eval/rhythm/extract.py --kind gold --shard 0 --nshards 4 --G 6      # PENDING confirmation
python eval/rhythm/extract.py --kind synth --G 6                           # PENDING confirmation

# Training. Script and flags PENDING; not present in this worktree yet.
python eval/rhythm/train.py ...                                            # PENDING confirmation

# Evaluation and ablations. Script and flags PENDING; not present in this worktree yet.
python eval/rhythm/evaluate.py ...                                         # PENDING confirmation
```

Notes for whoever fills this in:

- Record the exact command, the git revision of the worktree, the `G` and `--no-ego`/`--no-flow`/`--tag` settings used for each cache, and the output JSON path next to every number.
- `extract.py` writes a per-shard manifest with per-window digests and per-window extraction seconds; those manifests are the raw evidence for both the determinism section and the decode-included cost, and should be cited by path.
- Cached tensors are float16, so any number derived from them must state that precision; re-running extraction overwrites the cache with byte-identical tensors by design, which is itself part of the determinism claim.
- A single-command end-to-end reproduction target, and the runtime it takes, are `PENDING`.

## Per-reason breakdown (headline run)

| reason | n | AUROC good-vs-this | recall at 0.5 |
|---|---|---|---|
| DUPLICATE | 6 | 0.657 | 1.000 |
| HANDS_NOT_VISIBLE | 14 | 0.804 | 1.000 |
| NO_ACTIVE_WORK | 23 | 0.679 | 1.000 |
| TOO_DARK | 6 | 0.990 | 1.000 |


## The decisive negative result: synthetic negatives destroyed real accuracy

Two full 7-fold LOCO runs, differing ONLY in whether tensor-level
synthetic-by-construction negatives were added to training:

| run | training set | AUROC on the 150 REAL human windows |
|---|---|---|
| with synthetic negatives | 150 real + 300 synthetic | **0.351** (inverted; CI 0.23-0.49) |
| no synthetic negatives | 150 real only | **0.753** (CI 0.65-0.81) |

With synthetic negatives the pooled score on real windows is *below chance*: the
model separates the synthetic artifacts, not real defects. Mean prediction was
0.131 for real good windows, 0.158 for real bad windows and 0.084 for synthetic
negatives -- it had collapsed toward the negative class and ranked real windows
by how synthetic-like they looked. The tensor-level synthetic approximations
(blur = grid smoothing, dark = channel scaling, frozen = first frame repeated)
are NOT pixel-accurate renderings of degraded video, and a model trained on them
learns the approximation, not the defect.

**This is the single most important result in this report.** Any pipeline that
trains on tensor-level synthetic degradations for this task should verify on
real labels before trusting the synthetic-inclusive score.

## Honest verdict against the success criterion

The stated success bar was LOCO AUROC >= 0.85 with CI lower bound >= 0.72 and no
clip below 0.60, OR a clearly documented negative result.

- Headline LOCO AUROC **0.753**, CI [0.65, 0.81].
  The point estimate is BELOW 0.85 and the CI lower bound is BELOW 0.72.
- `pallet` (0.515) and `pallet2` (0.500) are at chance,
  so the "no clip below 0.60" condition fails.
- The model does NOT beat the 26-GPU-feature logistic baseline (0.810)
  under identical folds, and only marginally exceeds the flow_act-like scalar
  family (0.764) and the global-channels-only ablation
  (0.724).

**Therefore the outcome is the second acceptable one: a documented negative
result.** The purpose-built spatial rhythm network, at this sample size and with
this training budget, does not beat a single residual-motion magnitude scalar.

## Ablations actually run

| ablation | AUROC (95% CI) | n | status |
|---|---|---|---|
| spatial grid present, K=32 ensemble, no synthetic | 0.753 (0.65-0.81) | 150 | measured |
| global channels only (no grid), same folds | 0.724 (0.59-0.80) | 150 | measured |
| single feature family (residual flow magnitude) | 0.764 (0.59-0.88) | 150 | measured |
| with synthetic negatives | 0.351 (0.23-0.49) | 150 | measured |
| no ego-motion compensation | see note | - | not measured |
| no skin channel | see note | - | not measured |
| G in {1,4,8} | see note | - | not measured |
| no augmentation | see note | - | not measured |
| single model vs ensemble | see note | - | not measured |

Note: the un-run ablations above are listed for completeness. They were NOT
executed inside the time box, and no number is reported for them. The
`custom_algo_data/research/ablations/` track independently measured the
spatial-vs-global question with hand-built summaries of the same cached tensors
(see below).

## Independent tracks (not this model)

Three parallel sub-agent tracks wrote their own results under
`custom_algo_data/research/` (outside git, large artefacts):

- **`ablations/`** -- hand-built spatial summaries of the same cached tensor under
  the same LOCO protocol. `sp_cellmax` (6 features: per-channel maximum over grid
  cells) reached pooled AUROC 0.799 (CI 0.690-0.849) against a size-matched
  global-only control whose 30 random draws averaged 0.725; the spatial set beat
  96.7% of matched global draws. Paired difference +0.074. This is the strongest
  evidence in the whole effort that the spatial grid carries information the
  global channels do not -- but it comes from hand-built features, not from the
  learned network.
- **`rhythm_analysis/`** -- temporal-spectrum features (band powers, spectral
  entropy, autocorrelation periodicity), evaluated raw and after variance
  normalisation to separate "how much motion" from "what shape the motion has".
- **`math_approaches/`** -- six framings (information-theoretic, spectral/graph,
  sparse coding, stochastic process, low-rank, distributional). Verdict: after
  removing magnitude, NO framing gave a statistically defensible gain over "how
  much residual motion there is". The spectral/graph framing beat the
  magnitude-only control on a paired clip bootstrap (+0.064, CI [+0.014,+0.097])
  but its edge vanished under variance normalisation (+0.012, CI [-0.051,+0.048]).
  Naive concatenation of all 206 features HURT (0.629 vs 0.738 for magnitude
  alone). 412 evaluations were performed with no multiple-comparison correction,
  so the best-of-N number is optimistically biased.

Taken together: the representation contains a real spatial signal (the cellmax
result), but neither the learned network nor any of the six mathematical framings
converted it into a gain over the single magnitude scalar at n=150.

## CORRECTION: the motion cue points the OPPOSITE way to the premise

Independently confirmed here with separate code (`eval/rhythm/sign_robust.py`,
`eval/results/rhythm/sign_robust.json`), and originally found by the
`rhythm_analysis` track. Every residual-motion MAGNITUDE scalar is strongly
INVERTED: **good windows have LESS residual flow than bad ones.**

| raw scalar | signed AUROC | orientation-free | direction |
|---|---|---|---|
| `flow_top_mean` (top-25% cells) | 0.196 | 0.804 | good < bad |
| `flow_mean` (whole window) | 0.197 | 0.803 | good < bad |
| `flow_cell_max` | 0.200 | 0.800 | good < bad |
| `coh_mean` (flow coherence) | 0.218 | 0.782 | good < bad |
| `coh_std` | 0.721 | 0.721 | good > bad |
| `concentration` (max cell / sum) | 0.759 | 0.759 | good > bad |

**Interpretation.** Ego-motion compensation here is TRANSLATION ONLY. When the
wearer walks or turns, the global translation model fails to explain the frame,
so the residual rises everywhere -- not because hands are working, but because
the camera is moving. The strongest cue in this representation is therefore
closer to a **"the wearer is not walking / hands are not dominant" detector than
a "work is happening" detector.** Any judge-facing explanation of this pipeline
must be corrected to say that.

Note also that `concentration` (good > bad, 0.759) points the expected way and
is the one spatial summary that is both orientation-correct and competitive.

## pallet is a SIGN INVERSION, not missing signal

Within `pallet` the magnitude scalar scores 0.273 (good windows there have MORE
flow -- the reverse of the other six clips). `pallet2` behaves like the majority
(0.750). This is a per-clip sign flip, not an absence of signal.

Label-free per-clip transforms (using only that clip's own median/MAD, so legal
under LOCO) were tested:

| pooling rule | pooled AUROC | pallet | pallet2 |
|---|---|---|---|
| raw | 0.766 | 0.273 | 0.750 |
| signed (negated) | 0.766 | 0.273 | 0.750 |
| `abs_z` within clip | 0.555 | 0.606 | 1.000 |
| distance from clip median | 0.735 | 0.515 | 0.750 |
| two-sided (raw + abs_z) | 0.752 | 0.242 | 0.750 |

Sign-robust pooling **does** lift pallet (0.273 -> 0.606 for `abs_z`, 0.515 for
distance-from-median) but costs the pooled score (0.766 -> 0.555 / 0.735). No
rule here dominates the raw baseline. **The live lead is per-clip sign
calibration, not a new feature** -- and it cannot be tuned on the held-out clip,
so it needs either more clips or a label-free sign decision rule that this
sample (7 clips) cannot validate.

## CORRECTION 2: the spatial LAYOUT carries no information beyond pooling

Found by the `ablations` track and **reproduced here on this code path**
(`eval/rhythm/evaluate.py --baselines`, `eval/results/rhythm/baselines.json`).
The decisive control uses the SAME grid channels with the spatial axis summed
away (mean and std over time x cells per channel), so it is like-for-like:

| feature set | d | AUROC (95% CI) |
|---|---|---|
| pooled control, no layout | 12 | **0.790 (0.60-0.90)** |
| global channels only (8 ch, mean/std/p10/p90) | 32 | 0.724 (0.59-0.80) |
| **rhythm ensemble (this model, spatial grid)** | - | **0.753 (0.65-0.81)** |

The learned spatial network scores **0.753**, which is BELOW the
12-feature pooled control at
**0.790** that has no layout information at all.

The `ablations` track measured the same thing with hand-built summaries:
`sp_cellmax - poolmean_only = -0.0018 [-0.077, +0.071]`, and adding spatial
features on top of pooled features was null or significantly WORSE
(`sp_flatmap` -0.062, `sp_motionconc` -0.023). Their size-matched check found a
5-feature *global-only* subset reaching 0.830.

**Conclusion: the grid's value was that it introduced `flow_mag`/`flow_coh` as
channels, not that it localised anything. Spatial localisation is a dead end for
this task at this sample size.** The earlier claim in this report that "the
representation contains a real spatial signal (the cellmax result)" was based on
comparing spatial features against a WEAKER global-only bank, not against the
correct pooled control. That claim is withdrawn.

## CORRECTION 3: pallet does NOT reproduce as "chance" on the 26-feature baseline

The handover states pallet 0.50 / pallet2 0.50 (chance). Independently
reproduced here for the 26 GPU per-pixel features: **pallet 0.500, pallet2 0.500**
-- so those handover numbers DO reproduce for that feature set.

The `ablations` track reports pallet **0.2879** and pallet2 **0.7500** for their
pooled grid-channel control. These are not contradictory: they are DIFFERENT
representations. pallet sits at chance on the 26 per-pixel features and below
chance on pooled grid channels, while pallet2 is at chance on the former and
above chance on the latter. The lesson is that "pallet is hard" is
representation-dependent, and quoting a single per-clip number without naming the
representation is meaningless.

The `ablations` track also tested and FALSIFIED the reason-mix explanation: pallet
is 45% NO_ACTIVE_WORK vs 48.6% elsewhere (Fisher p=0.758), and HANDS_NOT_VISIBLE
is the third-EASIEST reason (0.852) yet pallet scores far below the other five
clips. **pallet is a clip-level (scene/wearer) shift, not a reason-level effect.**

## Full ablation matrix (measured, not estimated)

All rows share IDENTICAL settings except the named change: K=32, top-M=16, 4
epochs, batch 8, no synthetic negatives, LOCO over 7 clips, pooled AUROC with
clip-bootstrap 95% CI (2000). They are directly comparable to each other.

| ablation | pooled AUROC | 95% CI | pallet |
|---|---|---|---|
| control_4ep | 0.799 | 0.703-0.847 | 0.530 |
| no_ego_motion | 0.794 | 0.693-0.845 | 0.530 |
| no_skin_channel | 0.795 | 0.695-0.845 | 0.530 |
| G=8 | 0.790 | 0.674-0.846 | 0.530 |
| G=4 | 0.785 | 0.683-0.837 | 0.530 |
| single_member | 0.783 | 0.649-0.831 | 0.515 |
| no_augmentation | 0.772 | 0.663-0.830 | 0.530 |
| single_no_aug | 0.768 | 0.638-0.818 | 0.515 |
| **reference: headline run, 25 epochs** | 0.753 | 0.651-0.813 | - |

### What the matrix says

1. **Every CI overlaps every other CI.** No ablation is distinguishable from the
   control at n=150. Removing ego-motion compensation, removing the skin channel,
   changing the grid from 4x4 to 8x8, removing augmentation, and collapsing the
   ensemble to a single member all land between 0.768 and 0.799 against a control
   of 0.799. **Nothing in the architecture is individually load-bearing**, which
   is consistent with the model not having learned a spatial-rhythm signal in the
   first place.
2. **The control at 4 epochs (0.799) scores ABOVE the
   25-epoch headline run (0.753).** The extra epochs
   did not help; if anything the longer schedule overfits. This is a training-budget
   artefact, not a modelling insight, and it means the 0.753 headline is not the
   best number this architecture produced under this protocol.
3. **pallet is stuck at 0.515-0.530 in every configuration.** No ablation moves it.
   Combined with the finding that pallet is a clip-level scene/wearer shift, this
   confirms it needs cross-clip adaptation, not a better feature or architecture.
4. **Ablations that should matter, do not.** `no_ego_motion` (0.794) is
   statistically indistinguishable from the control (0.799) -- despite
   ego-motion compensation being the design's stated core contribution. Given the
   motion-cue inversion finding (translation-only warping fails when the wearer
   walks), this is coherent: the compensation is not doing what it was intended to do.

## Determinism proof (measured)

`eval/results/rhythm/determinism.json`, reproducible with
`eval/rhythm/determinism.py`. Five windows spanning clips and labels:

- **10 repeated `score()` calls per window**: identical `features_digest`,
  `usable_prob` and `verdict` every time. 5/5 windows stable.
- **Representation byte-stability**: extracting the same window twice from scratch
  gives identical tensor digests. 5/5 identical.
- **OMP_NUM_THREADS=1 vs 8**: **0 differing fields across 5 windows**
  (`n_differing_fields: 0`). Verdict: IDENTICAL across thread counts.

Scope of the claim: **same machine, pinned stack, CPU reference path only.**
Cross-device identity and GPU-vs-CPU identity are NOT claimed and were not
measured. Receipts in this project are content-hashed (SHA-256 of canonical
JSON); they are **not** cryptographically signed. This proves reproducibility,
not authenticity.
