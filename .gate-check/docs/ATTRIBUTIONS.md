# Attributions

Every dataset, model, service and external package that Jadal QA uses or depends on, with its licence and how we use it.

**Verification status.** The licences below are as stated in the project brief. They have not been checked against each source for this document. Confirm each one against its source before submission. Items marked **check** are known to need it.

| Name | Type | Source and owner | Licence | How we use it | In this repo? |
|---|---|---|---|---|---|
| IndEgo | Dataset (egocentric video) | Fraunhofer IPK. Dataset: <https://huggingface.co/datasets/FraunhoferIPK/IndEgo> | CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>) | Demo footage: ECU-assembly clips, 14 key steps, VIA step timestamps. Used for the demo and to draft the step labels. Changes made: downscaled and segmented clips are used for the local demo; clips are not redistributed in this repo. | **No.** Clips stay out of git. |
| Egocentric-10K-Evaluation | Dataset (egocentric video) and labels | Build AI | Apache-2.0 (brief) | Evaluation split and calibration source. Gated shard access is needed for some files. | **No** media. Derived split manifests and calibration outputs may be added by PR #16. |
| gemini-2.5-flash labels for Egocentric-10K-Evaluation | Generated labels | Google (Gemini API) | Google API terms. **check** whether the labels may be redistributed. | Automatic labels for the evaluation set. **Not human-checked.** | Only in derived artifacts. |
| EmbeddingGemma 2 | Model (text and image embeddings) | Google | Apache-2.0 per Google's announcement; verify on the model card (Gemma weights are sometimes under the Gemma Terms). | Optional helper, not the core: frame heads, hub search and de-duplication. | **No.** Weights are not committed. |
| MediaPipe model files: `hand_landmarker.task`, `blaze_face_short_range.tflite` | Model (hand and face detection) | Google | Google MediaPipe model files (downloaded from storage.googleapis.com/mediapipe-models; licence not verified here; not redistributed). | L0 hand and face detection (`jadal/signals.py`, PR #12). | **No.** Downloaded into `models/`, which is git-ignored. |
| Laya (Design 1 fusion) | Model and code | NandhaKishorM/laya | Apache-2.0 | Not in the pipeline. Stretch idea only (issue #9). | No. |
| Sarvam AI Bulbul v3 (TTS API) | Hosted API | Sarvam AI | Sarvam service terms. **check** the terms for generated audio. | Generates the pre-recorded voice messages (`voicebank/gen_voicebank.py`). | Generated `.wav` files are git-ignored. The API key is read from `SARVAM_API_KEY`. |
| MicroPython | Firmware | MicroPython project | MIT | Raspberry Pi Pico (RP2040 tested) firmware (`pico/main.py`), v1.29.0 RPI_PICO build. | Not committed as a binary. |

## Other dependencies

Python packages such as OpenCV, NumPy, SciPy, FastAPI, Uvicorn, imagehash, Pillow, pyserial and the `sarvamai` SDK are used but are not covered by the brief. Their licences have not been audited for this document. Audit them from the lock or environment before submission.
