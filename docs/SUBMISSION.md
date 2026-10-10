# Submission pack — Jadal (Awign “Physical AI”, team submission)

**Status documented:** 2026-10-10 · **Branch:** `b/submission-main` (re-landed on `main`;
supersedes #93) · **Sources of record:** `docs/FACTS.md` (verbatim snapshot of the pod
fact sheet, sha256 `77c7331b5e9ca8e8…`) and `experiments/registry.jsonl` (union, 36 entries:
28 done / 7 running / 1 pending; verdicts 10 adopt, 9 drop, 8 maybe, 2 inconclusive, 7 none;
append-only, latest line per id wins → `docs/EXPERIMENTS.md`).

## 1 · One paragraph

Jadal is a deterministic QA funnel for crowd-captured head-camera factory video. Every
10-second window is analysed per-pixel on every frame (S0 exact rules for dark/blocked/
over-exposed/frozen, S1 streaming spatial + temporal spectrum, noise model, motion after
camera-motion removal) and gets ACCEPT / REJECT-with-evidence / REVIEW plus a
content-hashed receipt (SHA-256 of canonical JSON, **not signed**). No generative model is
in the decision path; Logistic regression / boosted trees are comparison baselines only.
Humans audit samples; the funnel can run fully automated.

## 2 · The number to read before all others (out-of-domain failure)

On **60 real Egocentric-10K windows** with human labels (**51 good / 9 bad**, one
labeller), the shipped IndEgo-calibrated cascade is **at chance**: score AUROC **0.529**
[0.32, 0.74], **21/51 good windows rejected (41%)** and 5/9 bad, balanced accuracy
**0.51**. The dashboard’s ~40% reject share on that footage is **mostly false rejects**
(domain shift 480p/10 fps IndEgo → 1080p/30 fps HEVC) and is **not** a quality measure.
`labels/ood_labels2.json` is **not delivered**, so all OOD claims stay at n=60 / n_bad=9.
Source: FACTS §“UPDATE 18:40 IST”; corroborated by registry `E6.03` (0.281 [0.145, 0.434]),
`E4.03` (0.415), `E1.03` (0.36–0.55).

**Verification note (LEAD B, claim LB1, `b/verify`, + independent verifier PR #100):** these
are **two different scorers and both numbers are correct**. **0.529** is the **PoC cascade as
shipped** (`eval/results/poc/ood_check_60.json`): AUROC 0.529 [0.316, 0.728], 21 of 51 good
windows falsely rejected, 5 of 9 bad, balanced accuracy 0.513 — it reproduces exactly.
**0.415** [0.207, 0.630] is the **frozen two-view ensemble** (`caches/ensemble/ood_scores.json`,
column `usable_prob`): 26 of 51 good windows rejected, 3 of 9 bad. Both are at chance with
heavy false rejects, so the out-of-domain failure is confirmed for both. **Do not merge the
two figures into one.** No artifact drift: the shipped OOD score file is byte-identical to
`origin/main` (sha256 `300c1eda…`).

## 3 · What to open

| Artifact | Path | What it is |
|---|---|---|
| Explainer | `docs/explainer.html` | self-contained page; every table carries its source line |
| Deck | `docs/deck/DECK.md` | slide source (Marp-compatible); OOD failure is slide 3 |
| PoC status | `docs/POC_STATUS.md` | current honest status, what runs, what is in progress |
| Experiment log | `docs/EXPERIMENTS.md` | full render of `experiments/registry.jsonl` (36 entries) |
| Fact sheet | `docs/FACTS.md` | verbatim snapshot the numbers come from |
| Registry | `experiments/registry.jsonl` | union, 36 entries; append-only; latest line per id wins |
| Independent verification | branch `b/verify`, `docs/reports/pod_verification.md` | LEAD B claim LB1: 20/20 published quantitative claims reproduced exactly (A40 pod, CPU-only); same verification filed findings — 14/316 checksum mismatches incl. the frozen scorer + the OOD two-scorer attribution correction — and the GPU feature-extraction half and OOD-labels half remain unverified |

## 4 · Claims we make (and where each number lives)

| Claim | Number | Source |
|---|---|---|
| 26-feature pixel view, LOCO n=150 | AUROC 0.810 [0.69, 0.89] | FACTS §18:25; registry `E1.01` |
| EG2 window view, LOCO n=150 | AUROC 0.848 [0.73, 0.91] | FACTS §18:25; registry `E1.02` |
| Probability average of the two views | AUROC 0.853 [0.73, 0.92] | FACTS §18:25 |
| Nested validation (inner-LOCO C) | EG2 0.846, pixel 0.792, prob-avg 0.834, rank-avg 0.844 — **gates ≥0.85 / CI-low ≥0.72 NOT met** | FACTS §18:40 |
| Selective accuracy | 0.74 at full coverage → 0.84 on the 80% most confident | FACTS §17:40 |
| Sampling policy adopted (IndEgo) | every 2nd window + 4 s exit ≈ 0.20 footage (×5) | registry `E3.01`–`E3.05` |
| Early exit | saturates at 4–6 s: 2 s 0.804, 4 s 0.860, 6 s 0.869, 8 s 0.857, full 0.848 | registry `E3.03` |
| fps sweep (pixel view) | 10 fps 0.809; 3.33 fps within 0.02 using 34 vs 100 frames | registry `E1.01` |
| EG2 fewer frames | 2/1/0.5 fps 0.848/0.860/0.870 (fewer frames not worse) | registry `E1.02` |
| PoC run on real footage | 1,546 windows, 58 clips, 4.404 h, 47 factories, descriptive verdicts | FACTS §17:40, §17:50 |
| Kernel-only GPU speed | ~700 video-hours per hour at 4.6 GiB (supersedes 1,800–2,000) | FACTS §17:50 |
| Receipts | SHA-256 of canonical JSON, re-verify by re-running the pinned pipeline; not signed | FACTS §“Measured results” |
| Determinism | identical quantised digest over 10 CPU runs; GPU/cross-device not claimed | FACTS §“Measured results” |

**Independent verification:** 20/20 exact on the A40 pod (CPU-only re-runs; E1.02
re-implemented independently), per `b/verify` `docs/reports/pod_verification.md`. The same
verification also **produced findings**: 14/316 checksum mismatches (including the frozen scorer)
and the OOD two-scorer attribution correction; the GPU feature-extraction half and the OOD-labels
half remain **unverified**.

## 5 · In progress — shown as such, never as results

Raspberry Pi measurement · Egocentric-10K accuracy (no human labels) · hyperspectral
(cameras are RGB; we use RGB temporal-spectral analysis) · freshness nonce-LED · voice ·
end-to-end GPU number · certified error bounds (needs ~86 clean fires; split has 84) ·
end-to-end determinism proof (10 receipt hashes) · spectral arm B re-run (stale 79-label
snapshot) · duplicates on the 6 real human DUPLICATE labels · `ood_labels2.json`
(not delivered → OOD stays n=60 / n_bad=9). Registry status (union registry, 36 ids): **28 done / 7 running / 1 pending**; verdicts
**10 adopt, 9 drop, 8 maybe, 2 inconclusive, 7 none**.

## 6 · Data, licences, honesty constraints

- IndEgo (Fraunhofer IPK) 150 human-labelled windows: 480p ~10 fps, 7 videos, **one
  labeller**, 100 usable / 50 bad; ~3 wearers. Not Awign footage.
- Egocentric-10K (builddotai): 58 clips / 4.404 h / 47 factories / 1,546 windows on disk,
  1080p HEVC 30 fps — **no human labels**; the full 85-factory corpus is download-blocked.
- Egocentric-10K-Evaluation frames: 10,000 single JPEGs with **AI** labels (not human);
  single frames, no temporal info.
- Videos/frames are never committed (licences); only code, small results JSON, labels and
  docs.
- OOD set is small (n_bad = 9) and every OOD number in this pack says so.

## 7 · Reproduce / verify locally (this branch)

```bash
python3 tools/exp.py render                 # regenerates experiments/EXPERIMENTS.md from the registry
python3 -m unittest discover -s tests       # unit/integration suite (stubs; no hardware, no network)
python3 -m unittest discover -s web         # dashboard tests
```

The PoC batch runner + dashboard (`poc_run.py`, `poc_server.py`, `poc_up.sh`) live on the
`integration/poc` branch, together with `eval/results/poc/`. The regression harness for
PR comparisons is registry `E9.01`; the independent verification is registry `LB1`.

## 8 · Limits we state, not hide

150 windows / 7 clips / one labeller; OOD n_bad = 9; correlated windows (effective n
≈ 20–43); no certified bounds; `pallet`/`pallet2` at chance for every model; the shipped
scorer does not transfer to 1080p/30 fps HEVC; thresholds not certified; receipts not
signed; no Raspberry Pi measurement; GPU numerics not claimed identical across devices.
