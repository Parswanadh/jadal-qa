# Jadal QA

A deterministic QA add-on for crowd-captured head-camera (egocentric) video.
Built for the Awign Hackathon: Physical AI (Oct 10-11, 2026). Work in this repo started at the event.

Head-camera upload event -> segment -> exact signals (OpenCV, mediapipe) -> typed decision models
(Laya, EmbeddingGemma 2 embeddings) -> calibrated threshold with an exact-binomial error bound ->
per-segment verdict -> batch decision -> signed, re-verifiable receipt -> dashboard, pre-recorded voice
messages and a Raspberry Pi Pico status light.

The build contract (data schemas, module interfaces, determinism rules) is in [`docs/SPEC.md`](docs/SPEC.md).
Python: 3.12. Secrets come from environment variables, never from the repo.

Status: early build. Nothing here is a measured result until a file under `eval/results/` says so.
