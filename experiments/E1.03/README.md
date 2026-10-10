# E1.03 — OOD 1080p30-source fps sweep {30,15,10,5,3,2,1}

**Why:** The OOD regime is original 1080p/30fps HEVC, not IndEgo 480p/10fps; the fps answer may differ per regime and the frozen IndEgo model is known not to transfer, so we also need label-free score stability across fps.

**Hypothesis:** Frozen IndEgo model AUROC stays 0.4-0.53 at every fps (no transfer); score rank-correlation across fps is high (>=0.8) so the fps choice does not change OOD ranking; per-feature direct AUROC is more stable than the frozen model.

**Data / label source:** 60 human-labelled Egocentric-10K OOD windows (labels/ood_labels.json: 51 good / 9 bad, human, 1 labeller) decoded from queues/label_queue_random.json source_path+source_start_s (original 1080p HEVC 30fps); plus 100 UNLABELLED queue-2 windows (labels/ood_labels2.json absent) for descriptive score stability only.

**Data regime:** out-of-domain Egocentric-10K 1080p 30fps HEVC, 47 factories, 1 window each

**Method:** decode each window at 160x90 with cv2 (CPU), index rule idx=arange(0,T,step), steps {30:1,15:2,10:3,5:6,3:10,2:15,1:30}, 26 cpu_window_features with TRUE effective fps=30/step; frozen logistic fit on ALL 150 IndEgo windows at 10fps (C=0.3, balanced, StandardScaler); window-bootstrap CI95 B=2000; spearman score rank-correlation across fps; per-feature signed AUROC.

**Owner:** experimenter  **Started:** 2026-10-10 15:17

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
