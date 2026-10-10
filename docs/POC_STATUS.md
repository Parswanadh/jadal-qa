# POC status — Jadal QA funnel (honest state, 2026-10-10)

> **Read first.** On 60 human-labelled Egocentric-10K windows (51 good / 9 bad) the
> shipped IndEgo-calibrated cascade is **at chance**: score AUROC **0.529** (95% CI
> 0.32–0.74, optimistic), **21 of 51 good windows falsely rejected (41%)** and 5 of 9
> bad, balanced accuracy **0.51**. The dashboard’s ~40% reject share on that footage is
> therefore **mostly false rejects from domain shift** (480p/10 fps IndEgo → 1080p/30 fps
> HEVC), **not a quality measure**. Two **different** scorers are both at chance here: the
> **PoC cascade as shipped** 0.529 [0.316, 0.728] (reproduces exactly) and the **frozen
> two-view ensemble** 0.415 [0.207, 0.630] — see the verification note in §3. In-domain
> numbers (below) are ~0.8 AUROC and the accuracy gates are **not met**. Nothing here is
> measured on a Raspberry Pi.
>
> Sources for this page: `docs/FACTS.md` (verbatim snapshot, sha256 `77c7331b5e9ca8e8…`)
> and `experiments/registry.jsonl` (34 entries, rendered to `docs/EXPERIMENTS.md`).

## 1 · What exists and runs

| Piece | State | Where |
|---|---|---|
| Deterministic S0/S1/S2 funnel (rules + per-pixel features + score) | runs | `jadal/`, `eval/`, `scripts/` on this branch |
| PoC batch run on real footage | ran (descriptive) | all **1,546** windows / 58 clips / 4.404 h / 47 factories of Egocentric-10K on disk |
| Dashboard on the real run (`:8840`) and OOD labelling tool (`:8814`) | ran | PoC branch `integration/poc` (not this docs branch); `scripts/poc_demo.sh`, `scripts/poc_stream_replay.py` are here |
| Content-hashed receipts (SHA-256 of canonical JSON, **not signed**) | runs | verdict path; re-verify = re-run the pinned pipeline on the original clip |
| Pico verdict LEDs | hardware demo | nonce-LED freshness check is **in progress** (roadmap/pilot) |
| PoC run/dashboard code (`poc_run.py`, `poc_server.py`, `poc_up.sh`) | exists on `integration/poc` | **not** on this docs branch |

## 2 · Measured accuracy (in-domain, IndEgo only)

150 ten-second windows, 7 IndEgo clips, 480p ~10 fps, **one human labeller** (100 usable /
50 bad). Protocol: leave-one-clip-out, clip-bootstrap 95% CI. Not Awign footage; file names
suggest only ~3 wearers.

| Model (LOCO, n=150) | AUROC | 95% CI |
|---|---|---|
| 26 per-pixel features + logistic | 0.810 | [0.69, 0.89] |
| EmbeddingGemma-2 window view (EG2) | 0.848 | [0.73, 0.91] |
| Probability average of the two views | 0.853 | [0.73, 0.92] |

- **Nested validation** (C chosen by inner LOCO): EG2 0.846, pixel 0.792, prob-avg 0.834,
  rank-avg 0.844. **Gates AUROC ≥ 0.85 and CI-low ≥ 0.72: NOT met.** Weakest clip 0.606.
- Selective accuracy: 0.74 deciding all 150 windows → 0.84 on the 80% most confident
  (26 of 100 good windows wrongly rejected at full coverage). The curve is not monotone.
- **~0.8 AUROC is the honest ceiling so far; accuracy is not yet high.**
- Independent verification (LEAD B, claim LB1, `b/verify`): **20/20 published quantitative
  claims reproduced exactly** — the raw/non-nested baselines (0.810 / 0.8480 [0.734, 0.909]
  / 0.853 / 0.856), the E1.01 fps sweep, E3.01 clip-error 0.0486 [0.0189, 0.0872], E3.03
  4 s 0.8596.

Sources: `docs/FACTS.md` §“UPDATE 18:25”, §“UPDATE 18:40”; registry `E1.01`, `E1.02`,
`E3.01`–`E3.05`, `LB1`.

## 3 · Out-of-domain (the failure)

60 random real Egocentric-10K windows, human-labelled by the builder: 51 good / 9 bad
(8 “no active work”, 1 shaky). Window bootstrap is optimistic.

| Metric | Value |
|---|---|
| score AUROC (shipped IndEgo-calibrated cascade) | **0.529** [0.32, 0.74] |
| good windows rejected | **21 / 51 = 41%** |
| bad windows rejected | 5 / 9 |
| balanced accuracy | **0.51** |

Corroborating registry results (different protocols, all exploratory at n_bad = 9):
`E6.03` frozen motion-cue direction AUROC 0.281 [0.145, 0.434]; `E4.03` frozen-ensemble
AUROC 0.415 on the 480p/30 transcode; `E1.03` frozen IndEgo logistic 0.36–0.55 across
source fps. **Domain shift is the finding; verdict shares are descriptive.**

> **Verification note (LEAD B, claim LB1, `b/verify`, + independent verifier PR #100):** these
> are **two different scorers and both numbers are correct**. **0.529** is the **PoC cascade as
> shipped** (`eval/results/poc/ood_check_60.json`): AUROC 0.529 [0.316, 0.728], 21 of 51 good
> windows falsely rejected, 5 of 9 bad, balanced accuracy 0.513 — it reproduces exactly.
> **0.415** [0.207, 0.630] is the **frozen two-view ensemble** (`caches/ensemble/ood_scores.json`,
> column `usable_prob`): 26 of 51 good windows rejected, 3 of 9 bad. Both are at chance with
> heavy false rejects, so the out-of-domain failure is confirmed for both. **Do not merge the
> two figures into one.** (Registry `E4.03` independently records the frozen-ensemble 0.415 as
> exploratory; no artifact drift: the shipped OOD score file is byte-identical to `origin/main`,
> sha256 `300c1eda…`.)

## 4 · In progress (never reported as results)

Raspberry Pi measurement · Egocentric-10K accuracy (no human labels on the 1,546 windows) ·
hyperspectral (cameras are RGB; we do RGB temporal-spectral analysis) · freshness
nonce-LED · voice · end-to-end GPU number · certified error bounds (not achievable at
n=150: correction needs ~86 clean fires, certification split has 84) · end-to-end
determinism proof (10 identical receipt hashes) · spectral arm B re-run (stale 79-label
snapshot) · duplicates on the 6 real human DUPLICATE labels · second OOD batch
`labels/ood_labels2.json` — **not delivered** (`E5.04` records `labels_missing=true`,
n=0), so **OOD claims stay n=60 / n_bad=9**.

## 5 · Registry snapshot (`experiments/registry.jsonl`)

34 entries: **24 done**, 10 running. Verdicts: 11 adopt, 8 drop, 6 maybe, 1 inconclusive,
8 none yet. Rendered in full at `docs/EXPERIMENTS.md` (append-only; latest line per id wins).

Highlights: `E1.01` fps sweep (10 fps 0.809; 3.33 fps within 0.02 at 34 vs 100 frames);
`E1.02` EG2 0.848/0.860/0.870 at 2/1/0.5 fps (fewer frames not worse); `E3.01`–`E3.05`
sampling policy adopted (every 2nd window + 4 s exit ≈ 0.20 footage, ×5; skip-the-start
dropped); `E1.04` frame selection dropped (uniform wins at equal budget); `E4.02` no
regime split; `E4.03` re-encoding does not fix transfer; `E5.02` ~12-window OOD adaptation
≈ 0.7 exploratory; `E5.04` blocked on `ood_labels2.json`; `E7.01` A40 cost model;
`E9.01`/`LB1` verification harness and independent re-run.

## 6 · Known limitations

Small n (150 windows / 7 clips, one labeller; OOD n_bad = 9) · windows within a clip are
correlated (effective n ≈ 20–43) · no certified error bounds · `pallet` / `pallet2` are at
chance for every model (clip-level sign inversion, open lead, not a result) · the pipeline
was calibrated on 480p/10 fps IndEgo and does **not** transfer to 1080p/30 fps HEVC as
shipped · thresholds not certified · receipts not signed · no Raspberry Pi measurement ·
GPU numerics not claimed identical across devices.

## 7 · Pointers

- Explainer (sources next to every table): `docs/explainer.html`
- Deck: `docs/deck/DECK.md`
- Submission pack index: `docs/SUBMISSION.md`
- Experiment log: `docs/EXPERIMENTS.md` (source: `experiments/registry.jsonl`)
- Fact sheet: `docs/FACTS.md`
