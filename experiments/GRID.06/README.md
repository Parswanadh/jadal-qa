# GRID.06 — Out-of-domain transfer of the mosaic head on the 160 labelled factory windows

**Why:** The customer's real footage is factory footage. In-domain AUROC is not the question that decides the product; transfer is. The shipped IndEgo-trained 26-number head is near chance on factory windows, so the mosaic head must be tested on the same windows with the same folds and a factory-clustered interval.

**Hypothesis:** The IndEgo-trained mosaic head will NOT transfer (the frozen per-pixel head scored 0.342 and the EG2 head 0.586 out of domain), but the mosaic-native spatial features are less tied to IndEgo photometry than the shipped temporal features, so an OOD-trained mosaic recipe may reach the same 0.6-0.7 band as the pod's OOD recipes.

**Data / label source:** OOD-160: 160 human-labelled Egocentric-10K windows, 138 good / 22 bad, 47 factories, ONE labeller, 19 of 22 bad are NO_ACTIVE_WORK; 480p window clips. In-domain training labels: the 150 IndEgo windows.

**Data regime:** 480p letterboxed 16:9 factory footage (source 1080p HEVC transcoded to 480p), CPU only

**Method:** frozen transfer (head trained on the 150 IndEgo windows only, applied unchanged) and 5-fold factory-grouped CV on the OOD set itself; factory-grouped bootstrap CIs (2000, seed 0); paired factory-clustered comparison against the frozen 26-number shipped-head scores on the same windows; reason-stratified result for NO_ACTIVE_WORK only

**Owner:** dsh  **Started:** 2026-10-18 06:49

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:49): drop.** OOD-160 (n=160, n_bad=22, 47 factories, ONE labeller; 19/22 bad are NO_ACTIVE_WORK). FROZEN IndEgo-trained mosaic head: AUROC 0.3557 [0.224,0.515] factory-grouped = INVERTED; paired vs the frozen 26-number shipped head (0.5878) delta -0.2320 [-0.380,-0.077], EXCLUDES ZERO. Trained on factory footage with 5-fold factory-grouped CV the same mosaic features reach 0.6798 [0.487,0.845], paired vs 26-number +0.0921 [-0.149,+0.296] = NOT established. Window-bootstrap intervals are optimistic and are reported only next to the grouped ones. NO_ACTIVE_WORK-stratified numbers are in the JSON.

Results file: eval/results/exp_grid/GRID.06.json

Decision: DROP the mosaic as a decision input out of domain: the IndEgo-trained mosaic head does not transfer (it inverts, like the frozen per-pixel head at 0.342 before it). The grouped-CV number 0.68 is the honest statement that factory labels are the missing ingredient, not the picture format.

Cost: 319 ms/window CPU for OOD mosaic features; 0 GPU
