# Documentation index

Start with the [project README](../README.md) for the overview and the honest-claims box.

## Design

- [`SPEC.md`](SPEC.md): the build contract. Data schemas, module interfaces, determinism rules and the Pico serial protocol. Every module codes against it.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): the pipeline as it is today. Layers L0, L1 and L2, receipts and re-verify, determinism rules, failure modes, and what we do not claim.
- [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md): datasets, models, services and their licences.
- [`laya_design1_plan.md`](laya_design1_plan.md): the Laya Design 1 fusion plan. Planned, not built (issue #9).
- [`sarvam_reference.md`](sarvam_reference.md): notes on the Sarvam AI API used for the voice bank.

## API

- `API.md`: **planned.** The HTTP API (issue #17). It arrives with the integration PR (#19, open). Until it merges, `jadal/service.py` and `SPEC.md` are the reference.

## Module reports

Reports on `main`, one per module:

- [`reports/certify.md`](reports/certify.md): calibration, exact binomial bound, content-hashed receipts, re-verify.
- [`reports/service.md`](reports/service.md): FastAPI service, SQLite, batch pipeline, bucket watcher.
- [`reports/pico.md`](reports/pico.md): Pico 2 firmware and the host serial link.
- [`reports/voicebank.md`](reports/voicebank.md): voice bank message catalogue and generator.
- [`reports/dashboard.md`](reports/dashboard.md): judge-facing dashboard.

Reports still on open branches (not on `main` yet): `signals` (PR #12), `evalharness` (PR #16), `integration` (PR #19).

Screenshots of the dashboard are in [`screens/`](screens/).

## Process

- [`process/README.md`](process/README.md): how the repo was built. Workflow, the reviews, and the five bugs they caught.
- [`process/decision-log.md`](process/decision-log.md): the ledger of rulings, scope cuts and incidents. Redacted for paths and session IDs.
- [`process/reviews/`](process/reviews/): every review and scoped re-review, round 0 through round 2, plus the round-0 review of the integration PR (15 files).
