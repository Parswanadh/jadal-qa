# TASK.00 — Do NO_ACTIVE_WORK reason labels predict the shipped 26-feature verdicts (160 Egocentric-10K windows)?

**Why:** The brief asks whether the free human reason label NO_ACTIVE_WORK already predicts what the shipped 26-number head decides on the 160 out-of-domain windows, and whether per-factory NO_ACTIVE_WORK share predicts per-factory reject+review share. If it does, the head adds little on this set; if it does not, the head is separating something the reason label does not capture.

**Hypothesis:** NO_ACTIVE_WORK windows fall mostly in the shipped reject/review bands and per-factory NO_ACTIVE_WORK share correlates positively with per-factory reject+review share.

**Data / label source:** 160 human-labelled Egocentric-10K 10-s windows (138 good / 22 bad; bad reasons NO_ACTIVE_WORK 19, HANDS_NOT_VISIBLE 2, SHAKY 1), ONE labeller, 47 factories; frozen old26 scores+verdicts from interview/dsh_pack/ood160_heads_old26_vs_new22.json

**Data regime:** out-of-domain (Egocentric-10K vs an IndEgo-trained head); factory is the grouping unit for CIs

**Method:** Verdict derived from the frozen score with jadal/poc/cascade.py::decide_window bands (review iff |p-0.5|<MARGIN=0.25, so accept iff p>=0.75, reject iff p<=0.25); the derivation is checked against the file's own verdict field. (i) accept/review/reject shares for NO_ACTIVE_WORK vs all other windows; (ii) per-factory NO_ACTIVE_WORK share vs reject+review share, Spearman and Pearson, n=47; (iii) AUROC of the frozen score for NO_ACTIVE_WORK vs good (label 1 = good) with factory-grouped bootstrap CI. Zero training, zero new inference.

**Owner:** S5-task  **Started:** 2026-10-18 06:37

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:42): inconclusive.** NO_ACTIVE_WORK labels only weakly track the shipped verdict. Band check: the band re-derived from the frozen old26 score (review iff |p-0.5|<0.25) reproduces the shipped verdict field on 160/160 windows. Shares: NO_ACTIVE_WORK (n=19) 7 accept / 2 review / 10 reject = 63.2% reject+review; all other 141 windows 64 accept / 20 review / 57 reject = 54.6% reject+review. 7 of 19 NO_ACTIVE_WORK windows are ACCEPTED by the shipped head. AUROC of the frozen score for NO_ACTIVE_WORK (0) vs good (1): 0.5755 [0.4275, 0.7386] factory-grouped bootstrap, n=157, n_bad=19, 47 factories, n_boot_valid=2000. Per-factory NO_ACTIVE_WORK share vs reject+review share: Spearman rho=0.1217, Pearson r=0.2040, permutation p=0.1708, n=47 factories of which only 11 have any NO_ACTIVE_WORK and 41 have <=5 windows -- a weak test, as the brief anticipated. Decision: NO_ACTIVE_WORK is NOT a usable stand-in for the shipped verdict (37% of those windows are accepted); treat it as a weak, noisy correlate only. All rates are descriptive agreement on 160 windows labelled by one human, not accuracy on unlabelled video.

Results file: eval/results/task/TASK.00.json

Decision: Do not substitute the reason label for the head; do not report NO_ACTIVE_WORK share as a proxy for reject rate. The association question stays open at n_bad=19.

Cost: 0 GPU, cached scores, <5 s CPU
