# Sarvam AI integration reference (verified from official docs)

Research date: 2026-10-10. All facts are from docs.sarvam.ai pages fetched in this session unless marked otherwise. Quotes are the fetch-tool's extraction of the page, not raw HTML, so re-check exact spelling before shipping code.

Legend: `[verified: URL]` = stated on the cited page. `[not found]` = not stated on any page checked.

---

## 1. Python SDK and Text-to-Speech (Bulbul v3)

- Package: `sarvamai` [verified: https://docs.sarvam.ai/api/getting-started/sdks] ("[`sarvamai` on PyPI](https://pypi.org/project/sarvamai/)")
- Install: `pip install -U sarvamai` [verified: https://docs.sarvam.ai/api/getting-started/quickstart/llms.txt] (the SDKs page shows `pip install sarvamai`)
- Import: `from sarvamai import SarvamAI` and `from sarvamai.play import save` [verified: https://docs.sarvam.ai/api/getting-started/quickstart/llms.txt]
- Client: `client = SarvamAI(api_subscription_key="YOUR_SARVAM_API_KEY")` [verified: https://docs.sarvam.ai/api/getting-started/quickstart/llms.txt]
- Minimal TTS call (quickstart, verbatim):
  ```python
  audio = client.text_to_speech.convert(
      text="Welcome to Sarvam AI!",
      language_code="hi-IN",
      model="bulbul:v4-flash",
      speaker="simran_en_customer",
  )
  ```
  [verified: https://docs.sarvam.ai/api/getting-started/quickstart/llms.txt]
- Parameter name: the quickstart uses `language_code` for TTS. The REST TTS page also uses `language_code` [verified: https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/rest-api]. A search snippet mentioned `target_language_code` for an older TTS version. That name is used for the Translate API, not TTS [verified: https://docs.sarvam.ai/api/getting-started/quickstart/llms.txt]. Treat `target_language_code` for TTS as unconfirmed.
- Model names: `bulbul:v4-flash` (latest, recommended), `bulbul:v3` (stable; API default when `model` is omitted), `bulbul:v2` (legacy) [verified: https://docs.sarvam.ai/api/getting-started/models/bulbul.md]
- Note: the user asked for Bulbul v3. The quickstart example uses `bulbul:v4-flash`. Pass `model="bulbul:v3"` explicitly.
- REST endpoint: `POST https://api.sarvam.ai/text-to-speech` [verified: https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/rest-api]
- REST headers: `api-subscription-key: <key>`, `Content-Type: application/json` [verified: same URL]
- REST request fields seen in examples: `text`, `language_code`, `speaker`, `model`, `pace`, `speech_sample_rate` [verified: same URL]. `output_audio_codec`, `sample_rate`, `temperature`, `enable_preprocessing` are [not found] on the pages checked.
- Response: `request_id` (string) and `audios` (array of base64 strings, one per input text). The page says the strings are "base64-encoded WAV strings, not raw binary" [verified: same URL]
- Saving to file, Python: `save(audio, "output.wav")` from `sarvamai.play` [verified: quickstart llms.txt]. Manual decode, cURL example: `base64.b64decode` [verified: quickstart llms.txt]. JS example: `Buffer.from(response.audios.join(""), "base64")` then `fs.writeFileSync("output.wav", audio)`. The page warns that writing the raw string gives a corrupted file [verified: quickstart llms.txt].
- Manual decode in Python, based on the documented response shape (`audios[0]` is base64 WAV): `base64.b64decode(resp.audios[0])` written in binary mode. This is an inference from the documented shape, not a copied doc snippet.
- Pace: 0.5 to 2.0 for `bulbul:v3` and `bulbul:v4-flash` [verified: https://docs.sarvam.ai/api/getting-started/models/bulbul.md]
- Output formats: `wav`, `mp3`, `aac`, `flac`, `linear16` work with `bulbul:v3` and `bulbul:v4-flash`. `opus` returns an error on `/text-to-speech` [verified: same URL]. A TTS overview page also lists `mulaw`, `alaw`, and `opus` as codecs [verified: https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/overview]. The REST page lists `Mulaw`, `Alaw`, `Opus` as supported [verified: rest-api URL]. Codec field name: [not found] (the page does not give the field name for codec).
- Sample rate: REST supports `8000 / 16000 / 22050 / 24000 / 32000 / 44100 / 48000`. Default when omitted: 22050 Hz [verified: https://docs.sarvam.ai/api/getting-started/models/bulbul.md]. The field name is `speech_sample_rate` in the REST examples [verified: rest-api URL].
- Max characters per request: 2,500 for Bulbul v3 [verified: https://docs.sarvam.ai/api/getting-started/models/bulbul.md, "2,500 characters per REST request"]. The REST TTS page says "2500 chars (v3)" [verified: rest-api URL].
- Language codes for Bulbul v3 [verified: https://docs.sarvam.ai/api/getting-started/models/bulbul.md]:
  `hi-IN` (Hindi), `bn-IN` (Bengali), `ta-IN` (Tamil), `te-IN` (Telugu), `gu-IN` (Gujarati), `kn-IN` (Kannada), `ml-IN` (Malayalam), `mr-IN` (Marathi), `pa-IN` (Punjabi), `od-IN` (Odia), `en-IN` (English, Indian accent)
- Speakers for Bulbul v3 [verified: https://docs.sarvam.ai/api/getting-started/models/bulbul.md]: Shubh (default), Aditya, Ritu, Priya, Neha, Rahul, Pooja, Rohan, Simran, Kavya, Amit, Dev, Ishita, Shreya, Ratan, Varun, Manan, Sumit, Roopa, Kabir, Aayan, Ashutosh, Advait, Anand, Tanya, Tarun, Sunny, Mani, Gokul, Vijay, Shruti, Suhani, Mohit, Kavitha, Rehan, Soham, Rupali
  - Caution: the quickstart uses `simran_en_customer` as the speaker. That string is not in the Bulbul v3 list. Use the list above and do not assume the quickstart value works with v3.
- Pricing: Bulbul v3 is ₹30 per 10K characters, rounded to the nearest character [verified: https://docs.sarvam.ai/api/getting-started/pricing]
- Determinism: [not found] on all TTS pages checked.

## 2. Speech-to-Text

### 2a. REST (Saaras)
- Endpoint: `POST https://api.sarvam.ai/speech-to-text` [verified: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/rest-api]
- Header: `api-subscription-key: YOUR_SARVAM_API_KEY`; `Content-Type: multipart/form-data` [verified: same URL]
- Form fields [verified: same URL]:
  - `file`: audio file object (the page says "Pass a **file object**, not a path string")
  - `model`: `saaras:v4` (default, recommended) or `saaras:v3` [verified: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/overview]
  - `mode`: `transcribe` (default), `translate`, `verbatim`, `translit`, `codemix` [verified: rest-api URL and overview URL]
  - `keyterms`: JSON array string, up to 50 terms, 64 characters each, supported only on `saaras:v4` [verified: rest-api URL]
  - `language_code`: [not found] as a request field on the REST page. Auto-detect is supported per the overview, but no request value for auto-detect is documented on the page checked. The response has `language_code` (null if not detected).
- Max audio duration: 30 seconds per REST request [verified: rest-api URL and overview URL]
- Response fields: `request_id` (string), `transcript` (string), `language_code` (string, BCP-47, null if none detected) [verified: rest-api URL]
- Python: `client.speech_to_text.transcribe(file=open("audio.wav","rb"), model="saaras:v4", mode="transcribe")` [verified: rest-api URL]

### 2b. Realtime WebSocket streaming
- Hostname: `api.sarvam.ai`. Path: `/speech-to-text-realtime/ws`. The page's cURL test uses `https://api.sarvam.ai/speech-to-text-realtime/ws?language_code=hi-IN&stream_type=fast`. The page does not name a `wss://` scheme [verified: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/realtime-streaming]
- Auth: header `api-subscription-key: YOUR_SARVAM_API_KEY`. The JS example uses `"Api-Subscription-Key"` in connect options. Query-param auth: [not found] [verified: realtime-streaming URL]
- Query params [verified: realtime-streaming URL]:
  - `language_code` (required; BCP-47 or `auto`)
  - `model`: default `saaras:v3-realtime`; also `saaras:v4`
  - `stream_type`: default `balanced`; `fast`, `balanced`, `simulated`
  - `mode`: default `transcribe`; `translate`, `verbatim`, `translit`, `codemix`
  - `endpointing`: default `vad`; or `manual`
  - `encoding`: default `linear16`; `linear32`, `mulaw`, `alaw`
  - `sample_rate`: default `16000`; `8000` or `16000` only; other values close the connection with code `4000`
  - `threshold`, `silence_duration_ms`, `min_speech_duration_ms`, `return_timestamps`, `prompt`, `keyterms`
  - Not found: `input_audio_codec`, `high_vad_sensitivity`, `vad_signals`. Earlier notes that mentioned these are not confirmed by this page.
- Client-to-server message: `{"event": "audio_input", "audio": "<base64 chunk>"}` [verified: same URL]. Python: `base64.b64encode(chunk).decode("utf-8")`. End-of-stream: `ws.send_realtime_end(RealtimeEnd())` in the Python SDK [verified: same URL]
- Server-to-client messages [verified: same URL]: `event` = `transcript.partial` or `transcript.final`. Text in `text` field (Python example reads `message.text`). `transcript.final` also has `language_confidence`, and `start_s`/`end_s` when `return_timestamps` is true. With `language_code="auto"`, both include `language`. Errors: `error` event with `code`, `is_fatal`, `message`.
- Python SDK: `client.speech_to_text_realtime_streaming.connect(language_code="hi-IN", stream_type="fast")` (async context manager) [verified: same URL]
- Streaming limits: concurrent WebSocket STT, 20 (Starter), 100 (Pro/Business) [verified: https://docs.sarvam.ai/api/getting-started/ratelimits]

## 3. Translation (Mayura / sarvam-translate)
- Endpoint: `POST https://api.sarvam.ai/translate` [verified: https://docs.sarvam.ai/api/api-guides-tutorials/text-processing/overview]
- Headers: `Content-Type: application/json`, `api-subscription-key: <key>` [verified: same URL]
- Request fields [verified: same URL]:
  - `input`: text to translate
  - `source_language_code`: e.g., `hi-IN`, `en-IN`, `auto`
  - `target_language_code`: e.g., `gu-IN`, `hi-IN`
  - `model`: `sarvam-translate:v1` or `mayura:v1`
  - `mode`: `formal` (default), `modern-colloquial`, `classic-colloquial`, `code-mixed`
  - `speaker_gender`: `Male` or `Female`
  - `enable_preprocessing`: [not found]
- Languages: Mayura supports "10 Indic Languages and English". Sarvam-Translate supports "all 22 official Indian languages" [verified: same URL]. Full code list: [not found] on this page.
- Character limit per request: 1,000 (`mayura:v1`), 2,000 (`sarvam-translate:v1`) [verified: same URL]
- Response field: `translated_text` (Python: `response.translated_text`; JS: `response.translatedText`) [verified: same URL]

## 4. Language identification
- No standalone endpoint found [not found]. Automatic detection is done via `source_language_code="auto"` on `/translate` [verified: text-processing URL]. Language detection on STT is through `language_code` in the response, and `auto` for realtime streaming [verified: realtime-streaming URL].

## 5. Rate limits and free credit
- Free credit: "Every new user receives ₹100 in credits." [verified: https://docs.sarvam.ai/api/getting-started/pricing]. Also "Sarvam offers ₹100 worth of free credits for every user on signup." and "Credits are universal and never expire." [verified: https://docs.sarvam.ai/api/getting-started/ratelimits]. The ₹1,000 figure was not found on any page checked. Confirmed amount: ₹100.
- Rate limits (per account; Starter / Pro / Business) [verified: https://docs.sarvam.ai/api/getting-started/ratelimits]:
  - STT REST: 60 / 100 / 4,000 req/min
  - STT WebSocket: 20 / 100 / 100 concurrent
  - STT batch: 20 / 100 / 500 req/min
  - TTS REST: 60 / 200 / 1,000 req/min (bulbul:v3 on Starter: 30 req/min, 30 concurrent)
  - TTS WebSocket: 60 / 200 / 1,000 concurrent
  - Text translation: 60 / 200 / 1,000 req/min

## 6. TTS determinism
- [not found]. No statement on determinism or reproducibility on the TTS overview, REST, model, or pricing pages checked.

## 7. Outbound calls (telephony)
- Exists: yes. Outbound calling is covered by the Voice Agents (Conversations) docs, with Campaigns and a developer API for triggering outbound calls [verified: docs.sarvam.ai search snippet, https://docs.sarvam.ai/conversations/overview and https://docs.sarvam.ai/conversations/deploy/telephony/exotel]. The snippet was not opened in full; verify the steps directly at https://docs.sarvam.ai/conversations/deploy/deploy-with-code before relying on them.
- The Twilio guide at https://docs.sarvam.ai/api/integration/build-voice-agent-with-twilio is inbound-only [verified: same URL].

---

## Conflicts and open items
1. TTS language field: `language_code` (quickstart, REST) vs `target_language_code` (older search snippet). Translate uses `target_language_code`.
2. Speaker name: quickstart uses `simran_en_customer`, which is not in the Bulbul v3 list.
3. Default TTS model: docs say `bulbul:v3` is the default when `model` is omitted, but the examples use `bulbul:v4-flash`.
4. STT REST `language_code` request field and auto-detect value: not found.
5. STT REST response vs realtime: `text` vs `transcript` naming differs.
6. TTS codec field name, sample-rate field name in the Python SDK: not confirmed in SDK code. The REST field name `speech_sample_rate` is from examples only.
7. TTS determinism: not found.
8. Realtime WS auth via query param: not found. `wss://` scheme: not stated.
9. Outbound telephony steps: not read in full.
10. Translate `enable_preprocessing`: not found.
