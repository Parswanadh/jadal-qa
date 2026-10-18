# AUTO.01 — Second-view resolver: EmbeddingGemma-2 window embeddings decide the review band

**Why:** If the shipped band (|p-0.5|<0.25) is resolved by a different view instead of a human, the error at 100% coverage may improve; the frozen EG2 embeddings are already cached so this costs no GPU.

**Hypothesis:** EG2 inside the band is above chance (Y3 measured EG2 nested AUROC 0.846 overall) and the hybrid beats the forced 0.5 rule on the same windows, but the paired clip-bootstrap interval may include zero.

**Data / label source:** IndEgo 150 windows, ONE human labeller; 26-feature cache + caches/frames10k/segments.npz (150 windows x 21 frames x 768-d EmbeddingGemma-2, ~2 fps)

**Data regime:** cached embeddings, CPU, no GPU, no new downloads

**Method:** mean-pool the 21 frame embeddings per window, same StandardScaler+LogisticRegression recipe, LOCO; conditional-on-band AUROC; hybrid = S1 outside band, EG2 (and rank-average) inside band; paired clip-bootstrap vs forced 0.5

**Owner:** dsh-s4-auto  **Started:** 2026-10-18 06:42

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:58): drop.** Second view = cached EmbeddingGemma-2 window embeddings (mean over the valid frames of 150x21x768; the 21st slot is NaN padding on 145/150 windows). 72 of 150 IndEgo windows are in the shipped band. In-band AUROC: S1 0.820 vs EG2 0.811 [0.683,0.882] clip-grouped, paired delta -0.010 [-0.115,+0.052] -> NOT established. At 100 % coverage the EG2-in-band hybrid gives bal-acc 0.740 (delta 0.000 [-0.044,+0.059] vs forced 0.5) -> the resolver does not improve the error. The band rank-average arm is the only one whose paired clip interval excludes zero: 0.800 vs 0.740, +0.060 [+0.015,+0.136]; but at equal coverage it beats S1's own ranks by +0.025 [+0.007,+0.050] and does NOT beat EG2's ranks (+0.025 [-0.028,+0.077]), so the gain is largely an operating-point change (FAR 0.26->0.12, FRR 0.26->0.28). Frozen cross-check: the repo's nested-LOCO EG2 view is 0.846 overall / 0.848 in-band, r=0.909 vs this fit.

Results file: eval/results/auto/AUTO.01.json

Decision: Do not adopt EG2 as the band resolver. Keep the rank-average only as a candidate operating point, to be tested against equal-coverage controls on fresh labels.

Cost: 0 GPU, no downloads, ~2.3 s CPU
