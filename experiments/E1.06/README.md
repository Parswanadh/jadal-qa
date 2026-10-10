# E1.06 — fps x resolution Pareto cross (IndEgo gold, pixel-26)

**Why:** Find cheapest (fps,res) cell within 0.02 AUROC of 480p/10fps baseline; prior E1.01 (fps) and X1/RES (res/CRF) tested one axis each, interaction (low fps + low res) unknown

**Hypothesis:** A mid cell (e.g. 360p/3.33fps or 240p/5fps) keeps AUROC within 0.02 of 0.809 at a fraction of bytes; very-low+very-low (240p/1fps) may interact downward

**Data / label source:** IndEgo gold manifest_gold.json, human labels, n=150 (100 good/50 bad), 7 clips

**Data regime:** 480p~10fps H.264 source; re-encoded grid fps {1,2,3.33,5} x res {480,360,240}px CRF23

**Method:** re-encode each gold window at (fps,res) libx264 veryfast CRF23 -> decode 160x90 -> 26 cpu per-pixel features -> LOCO logistic C=0.3 balanced + StandardScaler; CI=clip-bootstrap B=1000 seed0

**Owner:** H3  **Started:** 2026-10-10 17:43

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
